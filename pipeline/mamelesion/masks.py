"""Construcao e validacao de mascaras de lesao.

O artigo (secao 2.2, "Lesion inpainting") gera as mascaras de treino a partir das
coordenadas de bounding box das anotacoes: 255 dentro da caixa e 0 fora. Por isso
a mascara retangular e a forma *dentro da distribuicao de treino* e e o padrao
aqui; elipse e mascara livre (pincel da GUI) ficam disponiveis como opcoes.

Todas as coordenadas desta modulo estao no canvas de 1024x1024. A conversao para
os 512x512 do modelo e uma divisao por 2 (ver preprocessing.build_canvas).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import cv2 as cv
import numpy as np


@dataclass
class LesionBox:
    """Caixa da lesao no canvas, com utilitarios de conversao."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        self.x1, self.x2 = int(min(self.x1, self.x2)), int(max(self.x1, self.x2))
        self.y1, self.y2 = int(min(self.y1, self.y2)), int(max(self.y1, self.y2))

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def scaled(self, factor: float) -> "LesionBox":
        return LesionBox(int(round(self.x1 * factor)), int(round(self.y1 * factor)),
                         int(round(self.x2 * factor)), int(round(self.y2 * factor)))

    def expanded(self, px: int, limit: Optional[int] = None) -> "LesionBox":
        box = LesionBox(self.x1 - px, self.y1 - px, self.x2 + px, self.y2 + px)
        if limit is not None:
            box = LesionBox(max(0, box.x1), max(0, box.y1),
                            min(limit, box.x2), min(limit, box.y2))
        return box

    def slices(self) -> Tuple[slice, slice]:
        return slice(self.y1, self.y2), slice(self.x1, self.x2)

    def to_dict(self) -> dict:
        return asdict(self)


def snap_to_latent(box: LesionBox, canvas: int = 1024, model_res: int = 512,
                   latent_factor: int = 8) -> LesionBox:
    """Alinha a caixa a grade do espaco latente.

    O VAE comprime 512 -> 64 (fator 8), logo 1 pixel latente corresponde a
    ``canvas / model_res * latent_factor`` = 16 pixels do canvas de 1024.
    Alinhar evita que a borda da mascara caia no meio de um pixel latente, o que
    produz vazamento de meio-tom na fronteira.
    """
    step = int(canvas / model_res * latent_factor)
    x1 = (box.x1 // step) * step
    y1 = (box.y1 // step) * step
    x2 = int(np.ceil(box.x2 / step)) * step
    y2 = int(np.ceil(box.y2 / step)) * step
    return LesionBox(max(0, x1), max(0, y1), min(canvas, x2), min(canvas, y2))


def mask_from_box(box: LesionBox, size: int = 1024, shape: str = "rect",
                  feather: int = 0) -> np.ndarray:
    """Mascara binaria uint8 (0/255) a partir da caixa.

    ``shape``: 'rect' (como no treino do artigo) ou 'ellipse'.
    ``feather`` > 0 suaviza a borda; util apenas para composicao, nao para o
    modelo, que espera mascara binaria.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    if shape == "rect":
        cv.rectangle(mask, (box.x1, box.y1), (box.x2, box.y2), 255, thickness=cv.FILLED)
    elif shape == "ellipse":
        cx, cy = box.center
        cv.ellipse(mask, (cx, cy), (max(1, box.w // 2), max(1, box.h // 2)),
                   0, 0, 360, 255, thickness=cv.FILLED)
    else:
        raise ValueError("forma de mascara desconhecida: " + str(shape))
    if feather > 0:
        k = feather * 2 + 1
        mask = cv.GaussianBlur(mask, (k, k), feather / 2.0)
    return mask


def box_from_mask(mask: np.ndarray) -> Optional[LesionBox]:
    """Caixa envolvente dos pixels nao nulos de uma mascara livre."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return LesionBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def binarize_mask(mask: np.ndarray, thresh: int = 0) -> np.ndarray:
    """Satura qualquer valor nao nulo para 255 (mesma logica da GUI original)."""
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    out = np.zeros(mask.shape[:2], dtype=np.uint8)
    out[mask > thresh] = 255
    return out


# ---------------------------------------------------------------------------
# Validacao e amostragem
# ---------------------------------------------------------------------------


def coverage_inside_breast(box: LesionBox, breast: np.ndarray) -> float:
    """Fracao da caixa que cai dentro do tecido mamario (0..1)."""
    sub = breast[box.slices()]
    if sub.size == 0:
        return 0.0
    return float((sub > 0).mean())


def validate_box(box: LesionBox, breast: np.ndarray, min_coverage: float = 0.98,
                 min_side_px: int = 16, canvas: int = 1024) -> Tuple[bool, str]:
    """Valida a caixa antes de gerar. Devolve (ok, motivo)."""
    if box.w < min_side_px or box.h < min_side_px:
        return False, "regiao muito pequena (minimo {} px de lado)".format(min_side_px)
    if box.x1 < 0 or box.y1 < 0 or box.x2 > canvas or box.y2 > canvas:
        return False, "regiao fora do canvas"
    cov = coverage_inside_breast(box, breast)
    if cov < min_coverage:
        return False, ("regiao {:.0f}% dentro da mama (minimo {:.0f}%): "
                       "lesao no fundo/borda nao e radiologicamente plausivel"
                       ).format(cov * 100, min_coverage * 100)
    return True, "ok"


def mm_to_px(mm: float, mm_per_px: float) -> int:
    return max(1, int(round(mm / mm_per_px)))


def sample_box(
    breast: np.ndarray,
    mm_per_px: float,
    diameter_mm_range: Tuple[float, float] = (8.0, 25.0),
    aspect_range: Tuple[float, float] = (0.75, 1.35),
    margin_px: int = 12,
    rng: Optional[np.random.Generator] = None,
    max_tries: int = 500,
    canvas: int = 1024,
) -> Optional[LesionBox]:
    """Amostra aleatoriamente uma caixa valida dentro da mama.

    Reproduz a estrategia de ``siemens_inpainting.py`` (amostrar area e razao de
    aspecto e rejeitar caixas que saiam do tecido), porem parametrizada em
    milimetros em vez de pixels, o que mantem o tamanho das lesoes coerente com a
    faixa clinica de massas (tipicamente 8-25 mm) independentemente da resolucao.
    """
    rng = rng or np.random.default_rng()
    eroded = cv.erode(breast, cv.getStructuringElement(cv.MORPH_ELLIPSE, (margin_px * 2 + 1,) * 2))
    ys, xs = np.nonzero(eroded > 0)
    if len(xs) == 0:
        return None

    for _ in range(max_tries):
        idx = int(rng.integers(len(xs)))
        cx, cy = int(xs[idx]), int(ys[idx])
        diam_mm = float(rng.uniform(*diameter_mm_range))
        aspect = float(rng.uniform(*aspect_range))
        d = mm_to_px(diam_mm, mm_per_px)
        w = max(8, int(round(d * np.sqrt(aspect))))
        h = max(8, int(round(d / np.sqrt(aspect))))
        box = LesionBox(cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
        if box.x1 < 0 or box.y1 < 0 or box.x2 > canvas or box.y2 > canvas:
            continue
        if coverage_inside_breast(box, breast) >= 0.995:
            return box
    return None


__all__ = [
    "LesionBox", "snap_to_latent", "mask_from_box", "box_from_mask", "binarize_mask",
    "coverage_inside_breast", "validate_box", "sample_box", "mm_to_px",
]
