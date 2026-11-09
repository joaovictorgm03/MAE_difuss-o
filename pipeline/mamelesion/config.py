"""Configuracao central do pipeline (todos os hiperparametros em um so lugar).

Carregada de um YAML; qualquer campo pode ser sobrescrito em codigo ou pela CLI
(``--set diffusion.guidance_scale=6``), o que mantem o modulo desacoplado da GUI
e dos scripts de teste.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathsConfig:
    input_dir: str = "../sem_anomalia"
    output_dir: str = "outputs"
    model_id: str = "Likalto4/mammo_lesion-inpainting"
    #: caminho local para o snapshot (permite rodar totalmente offline)
    model_local_path: Optional[str] = None


@dataclass
class PreprocessConfig:
    canvas_size: int = 1024            # resolucao de trabalho e de saida
    model_resolution: int = 512        # resolucao do modelo (latente 64x64)
    framing: str = "center_crop"       # center_crop (artigo) | pad
    max_value: Optional[float] = None  # saturacao usada no uint16 -> uint8
    breast_threshold: int = 10
    detector_long_side_mm: float = 295.0
    mm_per_px: Optional[float] = None  # sobrescreve a estimativa geometrica


@dataclass
class LesionTypePreset:
    """Preset por tipo de lesao.

    O modelo do artigo foi ajustado com um unico prompt generico
    ("a mammogram with a lesion") para todas as imagens de treino, ou seja, o
    condicionamento textual nao carrega o tipo da lesao. O controle de tipo aqui
    e feito pela geometria da regiao e pelos parametros de difusao, que sao os
    fatores que de fato mudam a aparencia gerada.
    """

    diameter_mm_range: Tuple[float, float] = (8.0, 25.0)
    aspect_range: Tuple[float, float] = (0.75, 1.35)
    guidance_scale: float = 4.0
    num_inference_steps: int = 40
    detail_transfer: float = 0.80
    lesion_gain: float = 1.0
    description: str = ""
    #: prompt proprio do tipo; ``None`` usa ``lesion.prompt``. Os tipos BI-RADS
    #: do modo nativo usam os prompts com que o LoRA foi ajustado.
    prompt: Optional[str] = None


def _default_presets() -> Dict[str, LesionTypePreset]:
    return {
        # Massa: lesao focal densa, bordas definidas, 10-25 mm.
        "mass": LesionTypePreset((10.0, 25.0), (0.8, 1.25), 4.0, 40, 0.80, 1.00,
                                 "massa focal densa de bordas definidas"),
        # Nodulo: menor e mais circunscrito.
        "nodule": LesionTypePreset((5.0, 12.0), (0.85, 1.18), 4.5, 40, 0.75, 1.00,
                                   "nodulo pequeno e circunscrito"),
        # Densidade/assimetria focal: area maior, sinal atenuado para ficar difuso.
        "density": LesionTypePreset((20.0, 40.0), (0.6, 1.6), 3.5, 40, 0.85, 0.60,
                                    "assimetria focal / densidade difusa"),
    }


@dataclass
class LesionConfig:
    prompt: str = "a mammogram with a lesion"
    negative_prompt: str = ""
    lesion_type: str = "mass"
    mask_shape: str = "rect"           # rect (como no treino) | ellipse
    snap_to_latent: bool = True
    min_coverage: float = 0.98         # fracao minima da caixa dentro da mama
    types: Dict[str, LesionTypePreset] = field(default_factory=_default_presets)

    def preset(self, lesion_type: Optional[str] = None) -> LesionTypePreset:
        key = lesion_type or self.lesion_type
        if key not in self.types:
            raise KeyError("tipo de lesao desconhecido: {} (disponiveis: {})".format(
                key, ", ".join(sorted(self.types))))
        return self.types[key]


@dataclass
class DiffusionConfig:
    num_inference_steps: int = 40      # valor usado em siemens_inpainting.py
    guidance_scale: float = 4.0
    scheduler: str = "dpmsolver"       # dpmsolver (repo original) | ddim
    seed: Optional[int] = 1337
    dtype: str = "float16"
    num_images_per_prompt: int = 1
    attention_slicing: bool = True     # economiza VRAM em GPUs de 12 GB


@dataclass
class CompositionConfig:
    """Como o resultado 512 do modelo volta para o canvas de 1024."""

    mode: str = "hybrid"               # hybrid (so a ROI muda) | global (artigo)
    blend: str = "feather"             # feather | poisson | hard
    feather_px: int = 24
    detail_transfer: float = 0.80      # reinjecao de textura de alta frequencia
    detail_sigma: float = 1.5
    border_offset_correction: bool = True
    #: ganho aplicado ao sinal da lesao (1.0 = como o modelo gerou). Valores
    #: menores produzem densidades sutis sem alterar a forma gerada.
    lesion_gain: float = 1.0


@dataclass
class NativeWindowConfig:
    """Geracao da lesao numa janela em resolucao original (ver :mod:`window`).

    Os campos de geometria (``context_factor``, ``mask_margin_frac``,
    ``feather_frac``, ``paste_feather_frac``) definem a distribuicao em que o
    LoRA e ajustado. Quando ``lora_path`` aponta para um treino, os valores
    gravados nele prevalecem sobre estes -- ver :func:`lora.load_run`.
    """

    enabled: bool = False              # false = modo canvas (comportamento original)
    context_factor: float = 2.0        # lado da janela / maior lado da lesao
    min_scale: float = 0.75            # teto de reducao da lesao; acima dele o contexto encolhe
    min_context: float = 1.25          # contexto minimo quando o teto atua
    mask_margin_frac: float = 0.10     # margem da mascara em torno da lesao
    feather_frac: float = 0.08         # transicao na composicao (fracao do lado da caixa)
    #: textura nativa do detector na margem regenerada em volta da lesao (sigma do
    #: passa-baixa em px do modelo; 0 desliga). So composicao: nao exige retreino.
    margin_texture_sigma: float = 3.0
    #: piso da mudanca de baixa frequencia, em niveis de 0-255: a lesao nao escurece o
    #: tecido do hospedeiro alem disso (halo escuro aprendido dos recortes). None desliga.
    density_floor: Optional[float] = -3.0
    paste_feather_frac: float = 0.15   # transicao da colagem nas amostras de treino (scripts/analise_colagem.py)
    lora_path: Optional[str] = None    # pasta de um treino (treinos/<run>) ou de um checkpoint
    lora_scale: float = 1.0            # intensidade do adaptador na inferencia
    #: guidance da inferencia; ``None`` usa a escolhida na selecao do treino
    #: (``melhor_guidance`` em treino.json) ou, sem ela, a do preset do tipo
    guidance: Optional[float] = None
    size_source: str = "catalog"       # catalog (tamanhos reais do treino) | preset (faixa em mm)


@dataclass
class OutputConfig:
    size: int = 1024
    save_grayscale: bool = True
    save_mask: bool = True
    save_metadata: bool = True
    save_comparison: bool = True


@dataclass
class QualityConfig:
    ring_px: int = 40                  # anel de tecido de referencia ao redor da ROI
    glcm_distances: List[int] = field(default_factory=lambda: [1, 3, 5])
    glcm_levels: int = 64
    patch_size: int = 96               # patches para as analises baseadas em DINOv2
    #: recorte menor, usado na sonda de impressao digital: precisa caber inteiro
    #: dentro da area reconstruida e fora da lesao
    probe_patch_size: int = 40
    n_control_patches: int = 200
    dinov2_id: str = "facebook/dinov2-base"
    dinov2_local_path: Optional[str] = None
    cad_scales_mm: List[float] = field(default_factory=lambda: [5.0, 8.0, 12.0, 18.0, 25.0])
    cad_top_k: int = 10
    #: portao de QC na geracao em lote: criterios obrigatorios e numero de
    #: tentativas (com nova semente) antes de desistir de um caso
    qc_enabled: bool = True
    qc_max_attempts: int = 3
    qc_required: List[str] = field(
        default_factory=lambda: ["weber_contrast", "cnr", "outside_changed_fraction",
                                 "roi_delta_ratio"])


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    lesion: LesionConfig = field(default_factory=LesionConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)
    native_window: NativeWindowConfig = field(default_factory=NativeWindowConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    # -- io ----------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "Config":
        path = Path(path) if path else REPO_ROOT / "config.yaml"
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        cfg = cls()
        for section, values in (raw or {}).items():
            if not hasattr(cfg, section) or values is None:
                continue
            _apply(getattr(cfg, section), values)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)

    # -- overrides ---------------------------------------------------------
    def override(self, dotted: str, value: Any) -> "Config":
        """Sobrescreve um campo por caminho pontilhado (``diffusion.seed=7``)."""
        target: Any = self
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[part] if isinstance(target, dict) else getattr(target, part)
        leaf = parts[-1]
        if isinstance(target, dict):
            target[leaf] = value
        else:
            current = getattr(target, leaf)
            setattr(target, leaf, _coerce(value, current))
        return self

    def apply_overrides(self, pairs: Optional[List[str]]) -> "Config":
        for pair in pairs or []:
            key, _, raw = pair.partition("=")
            self.override(key.strip(), yaml.safe_load(raw))
        return self

    # -- derivados ---------------------------------------------------------
    @property
    def scale_canvas_to_model(self) -> float:
        return self.preprocessing.model_resolution / self.preprocessing.canvas_size

    def resolve_input_dir(self) -> Path:
        p = Path(self.paths.input_dir)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    def resolve_output_dir(self) -> Path:
        p = Path(self.paths.output_dir)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _apply(obj: Any, values: Dict[str, Any]) -> None:
    """Aplica um dicionario sobre um dataclass, recursivamente."""
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        if key == "types" and isinstance(value, dict):
            presets = getattr(obj, "types")
            for name, preset_values in value.items():
                base = presets.get(name, LesionTypePreset())
                _apply(base, preset_values)
                presets[name] = base
            continue
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        else:
            setattr(obj, key, _coerce(value, current))


def _coerce(value: Any, current: Any) -> Any:
    """Converte listas em tuplas quando o campo original e uma tupla."""
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


__all__ = ["Config", "PathsConfig", "PreprocessConfig", "LesionConfig",
           "LesionTypePreset", "DiffusionConfig", "CompositionConfig",
           "NativeWindowConfig", "OutputConfig", "QualityConfig", "REPO_ROOT"]
