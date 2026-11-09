"""Uma lesao gerada numa janela nativa: da posicao ao arquivo na profundidade original.

Encadeia as pecas de :mod:`mamelesion.window` na ordem unica em que devem ser
usadas. O pipeline de producao, a avaliacao e a validacao durante o treino
chamam :func:`generate_native` -- mudam apenas a funcao que roda o modelo
(com ou sem LoRA, com o pipeline de producao ou com o UNet em treino).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .config import NativeWindowConfig
from .masks import LesionBox
from .window import (Window, box_mask, extract, lesion_box_model, native_footprint, plan_window,
                     restore_native_margin, floor_low_frequency, to_u8, window_delta, writeback)

#: ``(janela_u8, mascara_u8, prompt, seed, passos, guidance, prompt_negativo) -> saida_u8``
GenerateFn = Callable[[np.ndarray, np.ndarray, str, int, int, float, Optional[str]], np.ndarray]


@dataclass
class NativeLesion:
    """Resultado de uma geracao, com tudo que QC, metricas e gravacao usam."""

    raw_out: np.ndarray = field(repr=False)        # imagem inteira, dtype original
    mask_native: np.ndarray = field(repr=False)    # 0/255, exatamente onde houve alteracao
    window: Window
    box: LesionBox                                 # caixa da mascara na entrada do modelo
    footprint: Tuple[int, int, int, int]           # lesao na imagem original
    original_u8: np.ndarray = field(repr=False)    # janela na entrada do modelo
    generated_u8: np.ndarray = field(repr=False)   # saida crua do modelo
    composed_u8: np.ndarray = field(repr=False)    # janela com a lesao composta
    mask_u8: np.ndarray = field(repr=False)
    prompt: str
    seed: int

    def lesion_crop(self, ref: float) -> np.ndarray:
        """A pegada da lesao na imagem gerada, em float32 [0, 1] -- comparavel aos recortes reais."""
        x0, y0, x1, y1 = self.footprint
        return self.raw_out[y0:y1, x0:x1].astype(np.float32) / float(ref)

    def meta(self) -> Dict[str, object]:
        return {"janela": self.window.to_dict(), "caixa_modelo": self.box.to_dict(),
                "pegada_nativa": list(self.footprint), "prompt": self.prompt, "seed": int(self.seed)}


def generate_native(generate_fn: GenerateFn, raw: np.ndarray, ref: float,
                    center_xy: Tuple[float, float], lesion_wh: Tuple[int, int],
                    nw: NativeWindowConfig, prompt: str, seed: int, steps: int = 40,
                    guidance: float = 4.0, gain: float = 1.0,
                    negative_prompt: Optional[str] = None, model_res: int = 512) -> NativeLesion:
    """Gera uma lesao de ``lesion_wh`` px nativos centrada em ``center_xy``."""
    fx0, fy0, fx1, fy1 = native_footprint(center_xy, lesion_wh, raw.shape)
    wh = (fx1 - fx0, fy1 - fy0)
    center = ((fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0)
    window = plan_window(center, wh, raw.shape, nw.context_factor, model_res,
                         nw.min_scale, nw.min_context)
    original_u8 = to_u8(extract(raw, window, ref))
    box = lesion_box_model(window, center, wh, nw.mask_margin_frac)
    mask_u8 = box_mask(box, model_res)

    generated_u8 = generate_fn(original_u8, mask_u8, prompt, int(seed), int(steps), float(guidance),
                               negative_prompt)
    u0, v0 = window.to_model(fx0, fy0)
    u1, v1 = window.to_model(fx1, fy1)
    composed_source = restore_native_margin(original_u8, generated_u8, (u0, v0, u1, v1),
                                            nw.margin_texture_sigma)
    composed_source = floor_low_frequency(original_u8, composed_source, nw.density_floor)
    delta = window_delta(original_u8, composed_source, mask_u8, box, nw.feather_frac, gain)
    composed_u8 = np.clip(original_u8.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    raw_out, mask_native = writeback(raw, window, delta, mask_u8, ref)
    return NativeLesion(raw_out, mask_native, window, box, (fx0, fy0, fx1, fy1), original_u8,
                        generated_u8, composed_u8, mask_u8, prompt, int(seed))


@dataclass
class _WindowCase:
    """O minimo de :class:`generator.Case` que :func:`quality.evaluate` le."""

    name: str
    breast: np.ndarray
    mm_per_px: float


def evaluate_window(lesion: NativeLesion, mm_per_px_native: float, quality_cfg, lesion_type: str,
                    breast_threshold: int = 10, name: str = "janela") -> Dict[str, object]:
    """Roda a bateria de :mod:`quality` sobre a janela composta, como o modo canvas faz no canvas.

    As metricas saem na escala da entrada do modelo; ``mm_per_px`` e convertido
    para ela, entao diametros em mm continuam corretos. Devolve a avaliacao com
    ``qc_ok`` e ``qc_falhas`` segundo ``quality_cfg.qc_required``.
    """
    from . import quality as qmod
    from .inpainting import GenerationResult
    from .preprocessing import breast_mask

    case = _WindowCase(name, breast_mask(lesion.original_u8, breast_threshold),
                       float(mm_per_px_native) / lesion.window.scale)
    result = GenerationResult(lesion.composed_u8, lesion.original_u8, lesion.generated_u8, lesion.mask_u8,
                              lesion.box, {"lesion_type": lesion_type})
    evaluation = qmod.evaluate(result, case, quality_cfg, lesion_type)
    passed, failed = qmod.passes_qc(evaluation, quality_cfg.qc_required)
    evaluation["qc_ok"], evaluation["qc_falhas"] = passed, failed
    return evaluation


def window_overlay(lesion: NativeLesion) -> np.ndarray:
    """Painel original | sintetica | diferenca (x4) da janela na entrada do modelo, com a caixa."""
    import cv2 as cv

    diff = np.clip(np.abs(lesion.composed_u8.astype(np.float32) - lesion.original_u8) * 4, 0, 255)
    panels = []
    for img in (lesion.original_u8, lesion.composed_u8, diff.astype(np.uint8)):
        rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        b = lesion.box
        cv.rectangle(rgb, (b.x1, b.y1), (b.x2 - 1, b.y2 - 1), (255, 60, 60), 1)
        panels.append(rgb)
    sep = np.full((lesion.original_u8.shape[0], 6, 3), 255, np.uint8)
    return np.hstack([panels[0], sep, panels[1], sep, panels[2]])


__all__ = ["GenerateFn", "NativeLesion", "generate_native", "evaluate_window", "window_overlay"]
