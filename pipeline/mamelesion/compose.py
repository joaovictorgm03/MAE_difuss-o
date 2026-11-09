"""Composicao do resultado do modelo (512x512) de volta no canvas de 1024x1024.

Dois modos:

``global``  - reproduz literalmente o artigo/GUI original: a imagem inteira sai
              do VAE e e apenas reamostrada para 1024. Todo o tecido passa pelo
              autoencoder, o que reduz um pouco o detalhe fora da lesao.

``hybrid``  - (padrao) somente a regiao mascarada e substituida; o restante do
              mamograma permanece *identico* a imagem real de entrada em 1024.
              E o comportamento desejado para aumento de dataset: a unica
              diferenca entre o par (real, sintetico) e a lesao.

O corte de 2x entre a escala do modelo e a de saida e compensado por transferencia
opcional de alta frequencia da imagem original, que devolve a granulacao do
detector dentro da ROI (as lesoes do tipo massa/densidade sao estruturas de baixa
e media frequencia, entao isso nao apaga a lesao gerada).
"""

from __future__ import annotations

from typing import Optional

import cv2 as cv
import numpy as np

from .config import CompositionConfig
from .masks import LesionBox


def upscale(img: np.ndarray, size: int) -> np.ndarray:
    """Reamostra para ``size`` com Lanczos (melhor preservacao de borda)."""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv.resize(img, (size, size), interpolation=cv.INTER_LANCZOS4)


def feather_alpha(mask: np.ndarray, feather_px: int,
                  box: Optional[LesionBox] = None, max_frac: float = 0.25) -> np.ndarray:
    """Alpha suave em [0,1]: 1 no interior, decaindo ate 0 na borda da mascara.

    A transicao acontece *dentro* da mascara, de forma que nenhum pixel fora da
    regiao marcada pelo usuario e alterado. A largura efetiva e limitada a
    ``max_frac`` do menor lado da regiao: sem esse teto, uma banda fixa de 24 px
    consumiria quase toda uma ROI pequena e atenuaria o contraste da propria
    lesao.
    """
    binary = (mask > 127).astype(np.uint8)
    if feather_px <= 0:
        return binary.astype(np.float32)
    if box is not None:
        feather_px = int(max(2, min(feather_px, max_frac * min(box.w, box.h))))
    dist = cv.distanceTransform(binary, cv.DIST_L2, 5)
    alpha = np.clip(dist / float(feather_px), 0.0, 1.0)
    return alpha.astype(np.float32)


def border_offset(diff: np.ndarray, mask: np.ndarray, band_px: int = 8) -> float:
    """Desnivel medio (DC) entre gerado e original na faixa interna da borda.

    Essa faixa corresponde a tecido que o modelo apenas reconstruiu (nao e a
    lesao), logo qualquer diferenca sistematica ali e erro de reconstrucao do
    VAE, nao sinal clinico. Remove-la elimina o degrau visivel no contorno.
    """
    binary = (mask > 127).astype(np.uint8)
    eroded = cv.erode(binary, cv.getStructuringElement(cv.MORPH_ELLIPSE, (band_px * 2 + 1,) * 2))
    band = (binary > 0) & (eroded == 0)
    if band.sum() < 10:
        return 0.0
    return float(np.median(diff[band]))


def detail_residual(original: np.ndarray, sigma: float) -> np.ndarray:
    """Componente de alta frequencia da imagem original (textura do detector)."""
    k = int(max(3, round(sigma * 6) | 1))
    low = cv.GaussianBlur(original.astype(np.float32), (k, k), sigma)
    return original.astype(np.float32) - low


def compose(
    original: np.ndarray,
    generated_model_res: np.ndarray,
    mask: np.ndarray,
    cfg: CompositionConfig,
    box: Optional[LesionBox] = None,
) -> np.ndarray:
    """Combina original (canvas) + saida do modelo (512) em um canvas final.

    Parametros
    ----------
    original : canvas original em tons de cinza, uint8, size x size
    generated_model_res : saida do modelo em tons de cinza, uint8, 512x512
    mask : mascara binaria no canvas (0/255)
    box : caixa da lesao, usada apenas para restringir o custo do Poisson
    """
    size = original.shape[0]
    generated = upscale(generated_model_res, size).astype(np.float32)
    orig = original.astype(np.float32)

    if cfg.mode == "global":
        return np.clip(generated, 0, 255).astype(np.uint8)
    if cfg.mode != "hybrid":
        raise ValueError("modo de composicao desconhecido: " + str(cfg.mode))

    if cfg.blend == "poisson":
        return _poisson(original, generated, mask, box)

    diff = generated - orig
    if cfg.border_offset_correction:
        diff = diff - border_offset(diff, mask)

    if cfg.blend == "hard":
        alpha = (mask > 127).astype(np.float32)
    else:  # feather
        alpha = feather_alpha(mask, cfg.feather_px, box)

    out = orig + alpha * float(cfg.lesion_gain) * diff

    if cfg.detail_transfer > 0:
        residual = detail_residual(orig, cfg.detail_sigma)
        out = out + alpha * cfg.detail_transfer * residual

    return np.clip(out, 0, 255).astype(np.uint8)


def _poisson(original: np.ndarray, generated: np.ndarray, mask: np.ndarray,
             box: Optional[LesionBox]) -> np.ndarray:
    """Clonagem no dominio do gradiente (cv.seamlessClone).

    Preserva os gradientes da lesao gerada e forca a continuidade exata na borda,
    eliminando o degrau por construcao. Mais caro e ocasionalmente instavel em
    mascaras coladas na borda da imagem, por isso nao e o padrao.
    """
    src = np.clip(generated, 0, 255).astype(np.uint8)
    src_rgb = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    dst_rgb = cv.cvtColor(original, cv.COLOR_GRAY2BGR)
    binary = ((mask > 127).astype(np.uint8)) * 255
    # seamlessClone exige que a mascara nao encoste na borda da imagem
    binary[0, :] = binary[-1, :] = binary[:, 0] = binary[:, -1] = 0
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return original
    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    try:
        blended = cv.seamlessClone(src_rgb, dst_rgb, binary, center, cv.NORMAL_CLONE)
    except cv.error:
        return original
    return cv.cvtColor(blended, cv.COLOR_BGR2GRAY)


__all__ = ["compose", "upscale", "feather_alpha", "border_offset", "detail_residual"]
