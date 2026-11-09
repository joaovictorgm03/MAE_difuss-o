"""Wrapper do pipeline de inpainting de lesoes do MAM-E.

Modelo: ``Likalto4/mammo_lesion-inpainting`` - Stable Diffusion v1-5 ajustado com
DreamBooth para inpainting de lesoes mamograficas, publicado pelos autores do
artigo (arXiv:2311.09822v1). A arquitetura confere com o descrito na secao 2.3:
UNet de 9 canais de entrada (latente + mascara + latente mascarado), VAE
congelado comprimindo 512x512 -> 64x64 e text encoder CLIP ajustado.

Os valores padrao de amostragem (40 passos, guidance 4.0, DPMSolverMultistep,
prompt "a mammogram with a lesion") sao os mesmos de
``generation/inpainting/siemens_inpainting.py`` no repositorio dos autores.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from .config import Config
from .masks import LesionBox

_DTYPES = {"float16": "float16", "fp16": "float16", "float32": "float32", "fp32": "float32"}


def resolve_model_path(model_id: str, local_path: Optional[str] = None) -> str:
    """Devolve um caminho carregavel para o modelo, priorizando o cache local.

    ``huggingface_hub.snapshot_download(local_files_only=True)`` recusa snapshots
    a que faltem arquivos irrelevantes (.gitattributes), entao o cache e
    inspecionado diretamente antes de recorrer ao Hub.
    """
    if local_path:
        return str(local_path)

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache = Path(HF_HUB_CACHE)
    except Exception:
        cache = Path.home() / ".cache" / "huggingface" / "hub"

    repo_dir = cache / ("models--" + model_id.replace("/", "--"))
    ref = repo_dir / "refs" / "main"
    if ref.exists():
        commit = ref.read_text().strip()
        snapshot = repo_dir / "snapshots" / commit
        if (snapshot / "model_index.json").exists():
            return str(snapshot)
    snapshots = repo_dir / "snapshots"
    if snapshots.is_dir():
        for snap in sorted(snapshots.iterdir(), reverse=True):
            if (snap / "model_index.json").exists():
                return str(snap)
    return model_id  # ultimo recurso: baixar do Hub


@dataclass
class GenerationResult:
    """Saida de uma geracao, com tudo que os testes de qualidade precisam."""

    composed: np.ndarray            # canvas final em tons de cinza (1024x1024)
    original: np.ndarray            # canvas original (1024x1024)
    generated_model_res: np.ndarray  # saida crua do modelo (512x512, cinza)
    mask: np.ndarray                # mascara usada, no canvas (0/255)
    box: Optional[LesionBox]
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def difference(self) -> np.ndarray:
        return self.composed.astype(np.float32) - self.original.astype(np.float32)


class LesionInpainter:
    """Carrega o modelo sob demanda e gera lesoes sinteticas."""

    def __init__(self, cfg: Config, device: Optional[str] = None,
                 lora_path: Optional[str] = None, lora_scale: float = 1.0):
        self.cfg = cfg
        self._pipe = None
        self._device = device
        #: adaptador LoRA (pasta com pytorch_lora_weights.safetensors). So o modo
        #: nativo o usa; ``None`` deixa o checkpoint original intocado.
        self.lora_path = lora_path
        self.lora_scale = float(lora_scale)

    # -- infraestrutura ----------------------------------------------------
    @property
    def device(self) -> str:
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def load(self):
        """Carrega (uma unica vez) o pipeline de difusao."""
        if self._pipe is not None:
            return self._pipe

        import torch
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler

        dtype_name = _DTYPES.get(self.cfg.diffusion.dtype, "float16")
        if self.device == "cpu":
            dtype_name = "float32"
        torch_dtype = getattr(torch, dtype_name)

        path = resolve_model_path(self.cfg.paths.model_id, self.cfg.paths.model_local_path)
        pipe = DiffusionPipeline.from_pretrained(
            path, safety_checker=None, requires_safety_checker=False, torch_dtype=torch_dtype
        )
        if self.cfg.diffusion.scheduler == "dpmsolver":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif self.cfg.diffusion.scheduler == "ddim":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        if self.cfg.diffusion.attention_slicing:
            pipe.enable_attention_slicing()
        if self.lora_path:
            from .finetune import apply_lora
            apply_lora(pipe, self.lora_path, self.lora_scale)
        self._pipe = pipe
        return pipe

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # -- geracao -----------------------------------------------------------
    def generate(
        self,
        canvas: np.ndarray,
        mask: np.ndarray,
        box: Optional[LesionBox] = None,
        lesion_type: Optional[str] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        detail_transfer: Optional[float] = None,
        lesion_gain: Optional[float] = None,
    ) -> GenerationResult:
        """Gera a lesao dentro da mascara e devolve o canvas composto.

        ``canvas`` e ``mask`` estao na resolucao do canvas (1024). A imagem e a
        mascara sao reduzidas para 512 (resolucao do modelo), a difusao roda em
        512 e o resultado volta para 1024 via :mod:`mamelesion.compose`.
        """
        import torch
        from .preprocessing import downscale, to_rgb
        from .compose import compose

        pipe = self.load()
        pre, dif, comp = self.cfg.preprocessing, self.cfg.diffusion, self.cfg.composition
        preset = self.cfg.lesion.preset(lesion_type)

        steps = int(steps if steps is not None else preset.num_inference_steps)
        guidance = float(guidance if guidance is not None else preset.guidance_scale)
        prompt = prompt if prompt is not None else self.cfg.lesion.prompt
        negative_prompt = negative_prompt if negative_prompt is not None else self.cfg.lesion.negative_prompt
        seed = dif.seed if seed is None else seed

        res = pre.model_resolution
        image_512 = Image.fromarray(to_rgb(downscale(canvas, res)))
        mask_small = downscale(mask, res)
        mask_512 = Image.fromarray(((mask_small > 127).astype(np.uint8) * 255), mode="L")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        autocast = torch.autocast("cuda") if self.device == "cuda" else _NullContext()
        with autocast, torch.inference_mode():
            out = pipe(
                prompt=prompt,
                image=image_512,
                mask_image=mask_512,
                negative_prompt=negative_prompt or None,
                num_images_per_prompt=dif.num_images_per_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=res,
                width=res,
                generator=generator,
            ).images[0]

        generated = np.array(out.convert("L"))

        from dataclasses import replace
        detail_transfer = preset.detail_transfer if detail_transfer is None else detail_transfer
        lesion_gain = preset.lesion_gain if lesion_gain is None else lesion_gain
        comp_cfg = replace(comp, detail_transfer=float(detail_transfer),
                           lesion_gain=float(lesion_gain))

        composed = compose(canvas, generated, mask, comp_cfg, box)

        params = {
            "model_id": self.cfg.paths.model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "lesion_type": lesion_type or self.cfg.lesion.lesion_type,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "scheduler": dif.scheduler,
            "seed": seed,
            "model_resolution": res,
            "canvas_size": pre.canvas_size,
            "composition_mode": comp_cfg.mode,
            "blend": comp_cfg.blend,
            "feather_px": comp_cfg.feather_px,
            "detail_transfer": comp_cfg.detail_transfer,
            "lesion_gain": comp_cfg.lesion_gain,
            "border_offset_correction": comp_cfg.border_offset_correction,
        }
        return GenerationResult(composed, canvas, generated, mask, box, params)


    def generate_window(self, image_u8: np.ndarray, mask_u8: np.ndarray, prompt: str,
                        seed: int, steps: int = 40, guidance: float = 4.0,
                        negative_prompt: Optional[str] = None) -> np.ndarray:
        """Inpainting cru de uma janela ja na resolucao do modelo (modo nativo).

        Diferente de :meth:`generate`, nao reamostra nem compoe: recebe a janela
        uint8 ``model_res x model_res`` e devolve a saida do modelo em cinza, no
        mesmo tamanho. A composicao e a volta a imagem original ficam em
        :mod:`mamelesion.window`.
        """
        return run_inpaint(self.load(), image_u8, mask_u8, prompt, seed, steps, guidance,
                           negative_prompt, self.device)


def run_inpaint(pipe, image_u8: np.ndarray, mask_u8: np.ndarray, prompt: str, seed: int,
                steps: int = 40, guidance: float = 4.0, negative_prompt: Optional[str] = None,
                device: str = "cuda") -> np.ndarray:
    """Uma chamada de inpainting sobre arrays; usada pela inferencia e pela validacao do treino.

    A imagem cinza e replicada em RGB e a mascara binarizada em 127, exatamente o
    que as amostras de treino (:mod:`mamelesion.paste`) entregam ao UNet.
    """
    import torch

    res = int(image_u8.shape[0])
    if image_u8.shape != (res, res) or mask_u8.shape != (res, res):
        raise ValueError("janela e mascara devem ser quadradas e do mesmo tamanho: {} {}".format(
            image_u8.shape, mask_u8.shape))
    rgb = np.repeat(image_u8[..., None], 3, axis=2)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    out = pipe(
        prompt=prompt,
        image=Image.fromarray(rgb),
        mask_image=Image.fromarray(((mask_u8 > 127).astype(np.uint8) * 255), mode="L"),
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        height=res,
        width=res,
        generator=generator,
    ).images[0]
    return np.array(out.convert("L"))


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


__all__ = ["LesionInpainter", "GenerationResult", "resolve_model_path"]
