"""Pre-processamento de mamografias segundo MAM-E (arXiv:2311.09822v1, secao 2.2).

Passos do artigo reproduzidos aqui:
  1. DICOM -> PNG (funcao :func:`dicom_to_gray`), aplicando LUT/VOI e corrigindo
     interpretacao fotometrica MONOCHROME1.
  2. Escalonamento das intensidades uint16 -> uint8 no intervalo [0, 255].
  3. Formato RGB replicando o canal cinza nos tres canais (necessario para usar
     pesos pre-treinados de Stable Diffusion).
  4. Enquadramento quadrado "aspect-preserved centered": Resize(lado curto) +
     CenterCrop, exatamente como dataset_analysis/save_pngs/save_RGB_pngs.py do
     repositorio original do MAM-E.

Diferenca deliberada: o canvas de trabalho e construido em 1024x1024 (requisito
de entrega). O canvas de 512x512 usado pelo modelo e obtido por reducao 2x exata
do canvas de 1024, de modo que a geometria entre as duas escalas e trivial e o
enquadramento visto pelo modelo e identico ao do artigo.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------


def dicom_to_gray(path: str | Path) -> Tuple[np.ndarray, Optional[float]]:
    """Le um DICOM de mamografia e devolve (array cinza, espacamento em mm/px).

    Aplica VOI LUT quando disponivel e inverte MONOCHROME1 para que tecido denso
    fique claro (convencao usada pelos datasets do artigo).
    """
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:  # LUT ausente ou malformada: segue com os dados crus
        pass
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr

    spacing = None
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        if hasattr(ds, tag):
            try:
                spacing = float(getattr(ds, tag)[0])
                break
            except Exception:
                pass
    return np.asarray(arr), spacing


def read_gray(path: str | Path) -> Tuple[np.ndarray, Optional[float]]:
    """Le PNG/JPEG/DICOM e devolve (array cinza 2D, espacamento mm/px ou None)."""
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom", ""}:
        try:
            return dicom_to_gray(path)
        except Exception:
            pass
    im = Image.open(path)
    if im.mode not in {"L", "I;16", "I", "F"}:
        im = im.convert("L")
    return np.asarray(im), None


# ---------------------------------------------------------------------------
# Escalonamento de intensidade (artigo: uint16 -> uint8 em [0, 255])
# ---------------------------------------------------------------------------


def to_uint8(arr: np.ndarray, max_value: Optional[float] = None) -> np.ndarray:
    """Escala intensidades para uint8 [0, 255].

    ``max_value`` reproduz o valor de saturacao fixo usado no repositorio
    original (3500 para Siemens/VinDr, 4095 para Hologic). Quando None, usa o
    maximo da propria imagem (robusto para PNG/JPEG ja em 8 bits).
    """
    arr = np.asarray(arr)
    if arr.dtype == np.uint8 and max_value is None:
        return arr
    ref = float(max_value) if max_value else (float(arr.max()) or 1.0)
    scaled = (arr.astype(np.float64) / ref) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def to_rgb(gray: np.ndarray) -> np.ndarray:
    """Replica o canal cinza nos tres canais RGB (artigo, secao 2.2)."""
    if gray.ndim == 3:
        return gray
    return np.stack((gray,) * 3, axis=-1)


# ---------------------------------------------------------------------------
# Enquadramento quadrado + geometria
# ---------------------------------------------------------------------------


@dataclass
class Framing:
    """Transformacao original -> canvas quadrado, guardada para reverter."""

    orig_w: int
    orig_h: int
    scale: float          # fator aplicado sobre a imagem original
    off_x: int            # deslocamento do crop, ja em coordenadas redimensionadas
    off_y: int
    pad_x: int            # padding aplicado (modo pad)
    pad_y: int
    size: int             # lado do canvas (1024)
    mode: str             # center_crop (artigo) ou pad

    def to_dict(self) -> dict:
        return asdict(self)

    def canvas_to_orig(self, x: float, y: float) -> Tuple[float, float]:
        return ((x - self.pad_x + self.off_x) / self.scale,
                (y - self.pad_y + self.off_y) / self.scale)

    def orig_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return (x * self.scale - self.off_x + self.pad_x,
                y * self.scale - self.off_y + self.pad_y)


def build_canvas(
    gray: np.ndarray,
    size: int = 1024,
    mode: str = "center_crop",
    max_value: Optional[float] = None,
) -> Tuple[np.ndarray, Framing]:
    """Constroi o canvas quadrado uint8 e a geometria correspondente.

    ``center_crop``: Resize(lado curto -> size) + CenterCrop(size). Identico ao
    pre-processamento do artigo, apenas em 1024 no lugar de 512.
    ``pad``: Resize(lado longo -> size) + padding preto centralizado; preserva a
    mama inteira, util para aumento de dataset, ao custo de alterar levemente o
    enquadramento visto pelo modelo.
    """
    gray = to_uint8(gray, max_value)
    h, w = gray.shape[:2]

    if mode == "center_crop":
        scale = size / min(w, h)
    elif mode == "pad":
        scale = size / max(w, h)
    else:
        raise ValueError("modo de enquadramento desconhecido: " + str(mode))

    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR
    resized = cv.resize(gray, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((size, size), dtype=np.uint8)
    off_x = off_y = pad_x = pad_y = 0
    if mode == "center_crop":
        off_x = max(0, (new_w - size) // 2)
        off_y = max(0, (new_h - size) // 2)
        crop = resized[off_y:off_y + size, off_x:off_x + size]
        canvas[:crop.shape[0], :crop.shape[1]] = crop
    else:
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, Framing(w, h, scale, off_x, off_y, pad_x, pad_y, size, mode)


def downscale(canvas: np.ndarray, size: int = 512) -> np.ndarray:
    """Reduz o canvas para a resolucao do modelo (512 -> latente 64x64)."""
    if canvas.shape[0] == size:
        return canvas
    return cv.resize(canvas, (size, size), interpolation=cv.INTER_AREA)


# ---------------------------------------------------------------------------
# Segmentacao da mama (regiao valida para posicionar lesoes)
# ---------------------------------------------------------------------------


def breast_mask(gray: np.ndarray, thresh: int = 10) -> np.ndarray:
    """Mascara binaria (0/255) da maior componente conexa de tecido.

    Mesma ideia de ``get_normal_BBox`` em generation/inpainting/siemens_inpainting.py
    do repositorio original: binarizacao + maior componente conexa, aqui com
    limiar explicito e fechamento morfologico para eliminar buracos de ruido.
    """
    if gray.ndim == 3:
        gray = gray[..., 0]
    _, binary = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=4)
    if n <= 1:
        return np.zeros_like(gray)
    largest = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    mask = np.zeros_like(gray)
    mask[labels == largest] = 255
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(mask, contours, -1, 255, thickness=cv.FILLED)
    return mask


def breast_side(mask: np.ndarray) -> str:
    """Devolve L ou R conforme o lado do canvas ocupado pela mama."""
    half = mask.shape[1] // 2
    return "L" if mask[:, :half].sum() >= mask[:, half:].sum() else "R"


def estimate_mm_per_px(framing: Framing, detector_long_side_mm: float = 295.0,
                       dicom_spacing: Optional[float] = None) -> float:
    """Estima mm por pixel *no canvas*.

    Se o espacamento do DICOM estiver disponivel, ele e convertido para a escala
    do canvas. Caso contrario assume-se que o lado longo da imagem original cobre
    ``detector_long_side_mm`` (detector padrao 24x30 cm), estimativa suficiente
    para dimensionar lesoes em mm. Sempre sobrescrivel pela configuracao.
    """
    if dicom_spacing:
        return float(dicom_spacing) / framing.scale
    long_side_px = max(framing.orig_w, framing.orig_h)
    mm_per_orig_px = detector_long_side_mm / long_side_px
    return mm_per_orig_px / framing.scale


__all__ = [
    "Framing", "dicom_to_gray", "read_gray", "to_uint8", "to_rgb",
    "build_canvas", "downscale", "breast_mask", "breast_side", "estimate_mm_per_px",
]
