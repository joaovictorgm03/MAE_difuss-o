"""Ajuste fino LoRA do checkpoint MAM-E sobre lesoes BI-RADS 4/5.

Conteudo:

* :class:`PromptPolicy` e :class:`PasteDataset` -- amostras de treino sob demanda;
* :func:`add_lora`, :func:`save_lora`, :func:`apply_lora` -- adaptadores no UNet,
  gravados no formato que ``diffusers`` le com ``load_lora_weights``;
* :func:`load_run` -- le a pasta de um treino e devolve a geometria gravada nele.

Por que LoRA e nao ajuste completo
----------------------------------
Os autores ajustaram UNet e codificador de texto inteiros com dezenas de
milhares de mamografias. Com 59 recortes de treino, o ajuste completo decora as
lesoes e desloca o modelo para longe do tecido normal que ele ja reconstroi bem.
O LoRA restringe a mudanca a matrizes de posto baixo nas camadas de atencao do
UNet; VAE e codificador de texto ficam congelados.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .birads_data import CLASS_PROMPTS, GENERIC_PROMPT, LesionCrop
from .config import NativeWindowConfig
from .paste import Host, PasteConfig, make_sample

#: camadas de atencao do UNet que recebem adaptadores
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
LORA_WEIGHTS = "pytorch_lora_weights.safetensors"
RUN_FILE = "treino.json"


@dataclass
class PromptPolicy:
    """Qual texto acompanha cada amostra.

    * prompt da classe (padrao): o que a inferencia usa;
    * prompt generico do checkpoint (``p_generic``): mantem o LoRA ancorado no
      condicionamento com que o modelo foi treinado;
    * prompt vazio (``p_empty``): sem ele o ramo incondicional da orientacao
      livre de classificador (guidance) nunca ve lesoes, e a guidance passa a
      empurrar para longe delas.
    """

    p_empty: float = 0.10
    p_generic: float = 0.20

    def choose(self, lesion_class: str, rng: np.random.Generator) -> str:
        r = rng.random()
        if r < self.p_empty:
            return ""
        if r < self.p_empty + self.p_generic:
            return GENERIC_PROMPT
        return CLASS_PROMPTS[lesion_class]

    def prompts(self) -> List[str]:
        return ["", GENERIC_PROMPT] + [CLASS_PROMPTS[k] for k in sorted(CLASS_PROMPTS)]


class PasteDataset:
    """Dataset ``torch`` de amostras por colagem, deterministico por indice.

    Cada indice produz sempre a mesma amostra (``rng = [seed, indice]``),
    independentemente de qual worker a gera -- o treino e reproduzivel e pode
    ser retomado a partir de um passo. ``length`` e o numero total de amostras
    (passos x lote): nao ha epocas, cada amostra e uma colagem nova.
    """

    def __init__(self, crops: Sequence[LesionCrop], hosts: Sequence[Host], paste_cfg: PasteConfig,
                 policy: PromptPolicy, class_probs: Dict[str, float], length: int, seed: int,
                 start_index: int = 0) -> None:
        self.by_class: Dict[str, List[LesionCrop]] = {}
        for c in crops:
            self.by_class.setdefault(c.lesion_class, []).append(c)
        missing = [k for k in class_probs if class_probs[k] > 0 and k not in self.by_class]
        if missing:
            raise ValueError("classes sem recortes de treino: {}".format(missing))
        self.classes = sorted(k for k in class_probs if class_probs[k] > 0)
        total = sum(class_probs[k] for k in self.classes)
        self.probs = np.array([class_probs[k] / total for k in self.classes])
        self.hosts = list(hosts)
        self.paste_cfg = paste_cfg
        self.policy = policy
        self.length = int(length)
        self.seed = int(seed)
        self.start_index = int(start_index)

    def __len__(self) -> int:
        return self.length - self.start_index

    def __getitem__(self, i: int) -> Dict[str, object]:
        import torch

        index = self.start_index + int(i)
        rng = np.random.default_rng([self.seed, index])
        error = None
        for _ in range(8):
            # um recorte que nao cabe no hospedeiro sorteado nao pode derrubar o worker:
            # sorteia de novo com o mesmo gerador (continua deterministico por indice)
            lesion_class = self.classes[int(rng.choice(len(self.classes), p=self.probs))]
            pool = self.by_class[lesion_class]
            crop = pool[int(rng.integers(len(pool)))]
            host = self.hosts[int(rng.integers(len(self.hosts)))]
            try:
                sample = make_sample(crop, host, rng, self.paste_cfg)
                break
            except ValueError as exc:
                error = exc
        else:
            raise RuntimeError("amostra {} falhou 8 vezes: {}".format(index, error))
        prompt = self.policy.choose(lesion_class, rng)

        image = torch.from_numpy(sample.image_u8.astype(np.float32) / 127.5 - 1.0)[None].repeat(3, 1, 1)
        mask = torch.from_numpy((sample.mask_u8 > 127).astype(np.float32))[None]
        return {"pixel_values": image, "mask": mask, "prompt": prompt, "index": index,
                "crop": crop.name, "classe": lesion_class, "host": host.name,
                "posicionamento": sample.placement}


def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    import torch

    out: Dict[str, object] = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }
    for key in ("prompt", "index", "crop", "classe", "host", "posicionamento"):
        out[key] = [b[key] for b in batch]
    return out


# ---------------------------------------------------------------------------
# Adaptadores
# ---------------------------------------------------------------------------


def add_lora(unet, rank: int = 16, alpha: int = 16, dropout: float = 0.0) -> list:
    """Injeta LoRA nas camadas de atencao e devolve os parametros treinaveis."""
    from peft import LoraConfig

    for p in unet.parameters():
        p.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                                init_lora_weights="gaussian", target_modules=list(LORA_TARGETS)))
    return [p for p in unet.parameters() if p.requires_grad]


def save_lora(unet, out_dir: Path) -> Path:
    """Grava so os adaptadores (dezenas de MB), no formato do ``diffusers``."""
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionInpaintPipeline.save_lora_weights(save_directory=str(out_dir), unet_lora_layers=state,
                                                     safe_serialization=True)
    return out_dir / LORA_WEIGHTS


def apply_lora(pipe, lora_path: str, scale: float = 1.0, adapter_name: str = "birads") -> None:
    """Carrega um adaptador num pipeline de inferencia ja montado."""
    lora_dir = resolve_lora_dir(Path(lora_path))
    pipe.load_lora_weights(str(lora_dir), weight_name=LORA_WEIGHTS, adapter_name=adapter_name)
    pipe.set_adapters([adapter_name], adapter_weights=[float(scale)])


# ---------------------------------------------------------------------------
# Pasta de treino
# ---------------------------------------------------------------------------


@dataclass
class RunInfo:
    """O que a inferencia precisa saber sobre um treino."""

    run_dir: Path
    lora_dir: Path
    native_window: Dict[str, float]
    size_catalog: Dict[str, List[List[int]]]
    prompts: Dict[str, str]
    config: Dict[str, object] = field(default_factory=dict)

    def apply_geometry(self, nw: NativeWindowConfig) -> List[str]:
        """Sobrescreve a geometria de ``nw`` pela gravada no treino. Devolve as divergencias."""
        diffs = []
        for key, value in self.native_window.items():
            if hasattr(nw, key) and getattr(nw, key) != value:
                diffs.append("{}: config.yaml={} treino={}".format(key, getattr(nw, key), value))
                setattr(nw, key, type(getattr(nw, key))(value))
        return diffs


def resolve_lora_dir(path: Path) -> Path:
    """Aceita a pasta do treino (usa ``melhor``), a de um checkpoint, ou o arquivo de pesos."""
    path = Path(path)
    if path.is_file() and path.name == LORA_WEIGHTS:
        return path.parent
    if (path / LORA_WEIGHTS).is_file():
        return path
    if (path / RUN_FILE).is_file():
        best = json.loads((path / RUN_FILE).read_text(encoding="utf-8")).get("melhor_checkpoint")
        if best and (path / best / LORA_WEIGHTS).is_file():
            return path / best
        raise FileNotFoundError("treino {} ainda sem checkpoint escolhido (melhor_checkpoint)".format(path))
    raise FileNotFoundError("nenhum {} em {}".format(LORA_WEIGHTS, path))


def load_run(path: str | Path) -> RunInfo:
    """Le ``treino.json`` a partir da pasta do treino ou de um checkpoint dentro dela."""
    lora_dir = resolve_lora_dir(Path(path))
    run_dir = lora_dir
    for _ in range(4):
        if (run_dir / RUN_FILE).is_file():
            break
        run_dir = run_dir.parent
    else:
        raise FileNotFoundError("{} nao encontrado acima de {}".format(RUN_FILE, lora_dir))
    info = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    catalog = json.loads((run_dir / "catalogo_tamanhos.json").read_text(encoding="utf-8"))
    return RunInfo(run_dir, lora_dir, info["geometria"], catalog, info["prompts_classe"], info)


__all__ = ["LORA_TARGETS", "LORA_WEIGHTS", "RUN_FILE", "PromptPolicy", "PasteDataset", "collate",
           "add_lora", "save_lora", "apply_lora", "RunInfo", "resolve_lora_dir", "load_run"]
