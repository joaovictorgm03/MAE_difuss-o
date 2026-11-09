"""Enquadramento pela mama e devolucao da lesao a resolucao original.

Duas necessidades que o pipeline original nao cobria:

**Enquadramento.** ``build_canvas(mode='center_crop')`` reproduz o artigo, mas
mamografias sao muito mais altas do que largas (neste conjunto, ate 916x2542):
o corte central descarta metade da mama. ``mode='pad'`` preserva tudo, ao custo
de encolher o tecido para uma faixa estreita cercada de preto, o que reduz a
resolucao efetiva da lesao. :func:`build_breast_canvas` corta pela caixa
envolvente da mama antes de redimensionar: preserva a mama inteira **e** usa o
canvas todo para tecido.

**Resolucao original.** O modelo trabalha em 512 e o canvas em 1024, mas a
entrega precisa sair no tamanho original de cada arquivo (ate 3508x4644, 8 ou
16 bits). A volta e feita somando o *delta* da lesao a imagem original em sua
profundidade nativa:

    saida = original + upsample(delta_da_lesao)

O delta e zero fora da regiao marcada, entao todo pixel fora da lesao permanece
**bit a bit identico** ao arquivo de entrada, inclusive em 16 bits. E como a
imagem original entra inteira na soma, a textura nativa do detector dentro da
lesao e preservada por construcao -- nao ha perda de granulacao pela passagem
do modelo em 512, que e o que a transferencia de detalhe do canvas tenta
compensar de forma aproximada.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2 as cv
import numpy as np
from PIL import Image

from .compose import border_offset, feather_alpha
from .config import CompositionConfig
from .masks import LesionBox
from .preprocessing import Framing, breast_mask, to_uint8


# ---------------------------------------------------------------------------
# Leitura preservando a profundidade original
# ---------------------------------------------------------------------------


@dataclass
class SourceImage:
    """Imagem de entrada com o necessario para reconstrui-la sem perdas."""

    path: Path
    raw: np.ndarray               # array nativo (uint8 ou uint16), 2D
    gray_u8: np.ndarray           # versao uint8 usada pelo canvas
    scale_to_u8: float            # raw = u8 * scale_to_u8 (aproximadamente)
    bit_depth: int
    pixel_spacing_mm: Optional[float]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.raw.shape[:2]

    def meta(self) -> Dict[str, Any]:
        return {
            "width": int(self.raw.shape[1]),
            "height": int(self.raw.shape[0]),
            "bit_depth": self.bit_depth,
            "dtype": str(self.raw.dtype),
            "value_range": [int(self.raw.min()), int(self.raw.max())],
            "pixel_spacing_mm": self.pixel_spacing_mm,
        }


def read_source(path: str | Path, max_value: Optional[float] = None) -> SourceImage:
    """Le PNG/TIFF/JPEG/DICOM preservando o array nativo.

    ``scale_to_u8`` guarda o fator do escalonamento para uint8, para que o delta
    calculado no canvas possa voltar a faixa dinamica original. Levanta
    ``ValueError`` com mensagem descritiva quando o arquivo nao e uma imagem
    2D utilizavel.
    """
    path = Path(path)
    spacing: Optional[float] = None

    if path.suffix.lower() in {".dcm", ".dicom"}:
        from .preprocessing import dicom_to_gray
        raw, spacing = dicom_to_gray(path)
        raw = np.asarray(raw)
    else:
        try:
            im = Image.open(path)
            im.load()
        except Exception as exc:                     # arquivo corrompido/truncado
            raise ValueError("nao foi possivel decodificar a imagem: {}".format(exc)) from exc
        if im.mode in {"RGB", "RGBA", "P"}:
            im = im.convert("L")
        raw = np.asarray(im)

    if raw.ndim == 3:
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError("esperado array 2D, recebido shape {}".format(raw.shape))
    if raw.size == 0 or min(raw.shape) < 64:
        raise ValueError("imagem pequena demais: {}x{}".format(raw.shape[1], raw.shape[0]))
    if raw.max() == raw.min():
        raise ValueError("imagem constante (sem conteudo)")

    if raw.dtype == np.uint8:
        bit_depth, native = 8, raw
    else:
        bit_depth = 16
        native = raw.astype(np.uint16) if raw.dtype != np.uint16 else raw

    gray_u8 = to_uint8(native, max_value)
    ref = float(max_value) if max_value else (float(native.max()) or 1.0)
    scale = 1.0 if bit_depth == 8 else ref / 255.0
    return SourceImage(path, native, gray_u8, scale, bit_depth, spacing)


# ---------------------------------------------------------------------------
# Enquadramento pela caixa envolvente da mama
# ---------------------------------------------------------------------------


def build_breast_canvas(gray_u8: np.ndarray, size: int = 1024, margin_frac: float = 0.02,
                        breast_threshold: int = 10) -> Tuple[np.ndarray, Framing, np.ndarray]:
    """Canvas quadrado centrado na mama, preservando-a por inteiro.

    A sequencia e redimensionar -> recortar -> preencher, exatamente a ordem que
    :class:`~mamelesion.preprocessing.Framing` assume, de modo que
    ``canvas_to_orig`` continua valido: basta expressar o deslocamento do
    recorte em coordenadas ja redimensionadas.

    Devolve ``(canvas, framing, mascara_da_mama_no_canvas)``.
    """
    h, w = gray_u8.shape[:2]

    # mascara em baixa resolucao: 8x mais rapida e suficiente para a caixa
    small = cv.resize(gray_u8, (max(64, w // 8), max(64, h // 8)), interpolation=cv.INTER_AREA)
    small_mask = breast_mask(small, breast_threshold)
    ys, xs = np.nonzero(small_mask > 0)
    if len(xs) == 0:
        raise ValueError("nenhum tecido mamario detectado (imagem toda escura?)")

    fx, fy = w / small.shape[1], h / small.shape[0]
    x0, x1 = int(xs.min() * fx), int(np.ceil((xs.max() + 1) * fx))
    y0, y1 = int(ys.min() * fy), int(np.ceil((ys.max() + 1) * fy))

    mx = int(round(margin_frac * (x1 - x0)))
    my = int(round(margin_frac * (y1 - y0)))
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(w, x1 + mx), min(h, y1 + my)

    box_w, box_h = x1 - x0, y1 - y0
    if box_w < 32 or box_h < 32:
        raise ValueError("regiao de mama degenerada: {}x{}".format(box_w, box_h))

    scale = size / float(max(box_w, box_h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR
    resized = cv.resize(gray_u8, (new_w, new_h), interpolation=interp)

    off_x, off_y = int(round(x0 * scale)), int(round(y0 * scale))
    crop_w, crop_h = int(round(box_w * scale)), int(round(box_h * scale))
    crop_w, crop_h = min(crop_w, new_w - off_x), min(crop_h, new_h - off_y)
    crop = resized[off_y:off_y + crop_h, off_x:off_x + crop_w]

    canvas = np.zeros((size, size), dtype=np.uint8)
    pad_x, pad_y = (size - crop.shape[1]) // 2, (size - crop.shape[0]) // 2
    canvas[pad_y:pad_y + crop.shape[0], pad_x:pad_x + crop.shape[1]] = crop

    framing = Framing(w, h, scale, off_x, off_y, pad_x, pad_y, size, "breast_crop")
    return canvas, framing, breast_mask(canvas, breast_threshold)


# ---------------------------------------------------------------------------
# Delta da lesao
# ---------------------------------------------------------------------------


def lesion_delta(result, cfg: CompositionConfig) -> np.ndarray:
    """Delta da lesao no canvas, em unidades uint8 e **sem** transferencia de detalhe.

    A transferencia de detalhe de :func:`mamelesion.compose.compose` existe para
    devolver a granulacao perdida quando a saida do modelo em 512 e ampliada. Na
    volta a resolucao original essa granulacao ja esta presente -- e a da propria
    imagem de entrada, que entra inteira na soma. Reaplicar o residual aqui
    dobraria o ruido dentro da ROI, que e exatamente o artefato que a metrica
    ``noise_ratio_roi_ring`` acusa.
    """
    from .compose import upscale

    size = result.original.shape[0]
    generated = upscale(result.generated_model_res, size).astype(np.float32)
    diff = generated - result.original.astype(np.float32)
    if cfg.border_offset_correction:
        diff = diff - border_offset(diff, result.mask)

    if cfg.blend == "hard":
        alpha = (result.mask > 127).astype(np.float32)
    else:
        alpha = feather_alpha(result.mask, cfg.feather_px, result.box)
    return alpha * float(cfg.lesion_gain) * diff


def writeback_native(source: SourceImage, framing: Framing, delta_canvas: np.ndarray,
                     box: LesionBox, mask_canvas: Optional[np.ndarray] = None,
                     context_px: int = 8) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """Aplica o delta do canvas sobre a imagem original, na resolucao original.

    Somente a janela correspondente a ROI e tocada; o restante do array e o
    original intocado. O delta e reamostrado, reescalonado para a faixa dinamica
    nativa e somado, com saturacao nos limites do dtype.

    A interpolacao e **bilinear, nao Lanczos**: os lobulos negativos do Lanczos
    produzem ringing na fronteira, o que espalha alteracoes de alguns niveis de
    cinza por milhares de pixels *fora* da regiao marcada e quebra a garantia de
    identidade bit a bit. O delta ja e de baixa frequencia (vem de uma ampliacao
    do resultado do modelo em 512), entao nao ha detalhe a preservar que
    justifique o risco. Quando ``mask_canvas`` e fornecida, o delta ainda e
    zerado fora dela, o que torna a garantia estrutural em vez de estatistica.

    Devolve ``(imagem_nativa, geometria_na_original, mascara_na_original)``. A
    mascara devolvida e exatamente a que delimitou a alteracao, e nao uma
    reamostragem independente -- gravar as duas separadamente abriria margem
    para elas divergirem por arredondamento.
    """
    out = source.raw.copy()
    h, w = out.shape[:2]

    # ROI do canvas -> original, com folga para a transicao suave
    grown = box.expanded(context_px, limit=framing.size)
    ox0, oy0 = framing.canvas_to_orig(grown.x1, grown.y1)
    ox1, oy1 = framing.canvas_to_orig(grown.x2, grown.y2)
    ox0, oy0 = int(np.floor(ox0)), int(np.floor(oy0))
    ox1, oy1 = int(np.ceil(ox1)), int(np.ceil(oy1))
    ox0, oy0 = max(0, ox0), max(0, oy0)
    ox1, oy1 = min(w, ox1), min(h, oy1)
    if ox1 - ox0 < 2 or oy1 - oy0 < 2:
        raise ValueError("ROI degenerada apos mapeamento para a resolucao original")

    sub = delta_canvas[grown.y1:grown.y2, grown.x1:grown.x2]
    if sub.size == 0:
        raise ValueError("ROI vazia no canvas")

    interp = cv.INTER_LINEAR if (ox1 - ox0) >= sub.shape[1] else cv.INTER_AREA
    delta_native = cv.resize(sub.astype(np.float32), (ox1 - ox0, oy1 - oy0), interpolation=interp)

    applied = np.zeros(out.shape[:2], dtype=np.uint8)
    if mask_canvas is not None:
        sub_mask = mask_canvas[grown.y1:grown.y2, grown.x1:grown.x2]
        native_mask = cv.resize(sub_mask, (ox1 - ox0, oy1 - oy0), interpolation=cv.INTER_NEAREST)
        delta_native[native_mask <= 127] = 0.0
        applied[oy0:oy1, ox0:ox1] = np.where(native_mask > 127, 255, 0).astype(np.uint8)
    else:
        applied[oy0:oy1, ox0:ox1] = 255

    delta_native = delta_native * source.scale_to_u8

    info = np.iinfo(out.dtype)
    patch = out[oy0:oy1, ox0:ox1].astype(np.float32) + delta_native
    out[oy0:oy1, ox0:ox1] = np.clip(patch, info.min, info.max).astype(out.dtype)

    geometry = {
        "x1": ox0, "y1": oy0, "x2": ox1, "y2": oy1,
        "width": ox1 - ox0, "height": oy1 - oy0,
        "center": [(ox0 + ox1) // 2, (oy0 + oy1) // 2],
        "scale_canvas_to_orig": float(1.0 / framing.scale),
    }
    return out, geometry, applied


def native_lesion_mask(framing: Framing, mask_canvas: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Reamostra a mascara do canvas para a resolucao original (uint8 0/255)."""
    h, w = shape
    full = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.nonzero(mask_canvas > 127)
    if len(xs) == 0:
        return full
    cx0, cy0, cx1, cy1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    ox0, oy0 = framing.canvas_to_orig(cx0, cy0)
    ox1, oy1 = framing.canvas_to_orig(cx1, cy1)
    ox0, oy0 = max(0, int(np.floor(ox0))), max(0, int(np.floor(oy0)))
    ox1, oy1 = min(w, int(np.ceil(ox1))), min(h, int(np.ceil(oy1)))
    if ox1 - ox0 < 1 or oy1 - oy0 < 1:
        return full
    sub = mask_canvas[cy0:cy1, cx0:cx1]
    full[oy0:oy1, ox0:ox1] = cv.resize(sub, (ox1 - ox0, oy1 - oy0), interpolation=cv.INTER_NEAREST)
    return full


def canvas_mask_from_native(mask_native: np.ndarray, framing: Framing) -> np.ndarray:
    """Projeta uma mascara nativa (0/255) no canvas, com a mesma sequencia de :func:`build_breast_canvas`.

    Inverso de :func:`native_lesion_mask`, usado pelo modo nativo, em que a mascara
    nasce na resolucao original. Vizinho mais proximo: a mascara continua binaria.
    """
    h, w = mask_native.shape[:2]
    size = framing.size
    new_w, new_h = max(1, int(round(w * framing.scale))), max(1, int(round(h * framing.scale)))
    resized = cv.resize(mask_native, (new_w, new_h), interpolation=cv.INTER_NEAREST)
    src = resized[framing.off_y:, framing.off_x:]
    ch = min(src.shape[0], size - framing.pad_y)
    cw = min(src.shape[1], size - framing.pad_x)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[framing.pad_y:framing.pad_y + ch, framing.pad_x:framing.pad_x + cw] = src[:ch, :cw]
    return canvas


def native_texture_metrics(native: np.ndarray, original: np.ndarray,
                           geometry: Dict[str, Any], mask: np.ndarray,
                           background_level: float = 0.0,
                           ring_frac: float = 0.6) -> Dict[str, float]:
    """Ruido, nitidez e contraste medidos **na saida entregue**, nao no canvas.

    As metricas de :mod:`mamelesion.quality` avaliam a composicao em 1024, que
    e um intermediario: ela recebe transferencia de detalhe para compensar a
    ampliacao 2x da saida do modelo. A imagem em resolucao original nao passa
    por isso -- a granulacao dela vem do proprio detector. As duas texturas,
    portanto, nao sao a mesma, e afirmar a do canvas como se fosse a do arquivo
    entregue seria impreciso.

    A lesao e segmentada aqui pela **mesma** rotina usada no canvas
    (:func:`mamelesion.quality.segment_lesion`, Otsu sobre a diferenca positiva)
    em vez de se tomar a caixa inteira. Sem isso o contraste nativo mediria a
    ROI toda -- lesao mais o tecido nao alterado ao redor dela dentro da caixa --
    e sairia sistematicamente menor que o do canvas, convidando a leitura falsa
    de que a lesao enfraquece na resolucao original.

    ``ring`` e o tecido real dentro da janela de contexto, fora da ROI, como em
    :func:`mamelesion.quality.ring_mask` -- inclusive na exclusao do fundo. Sem
    ``background_level``, uma lesao proxima da borda faz o anel atravessar a
    linha da pele: bastam ~2% de pixels de fundo para dominar a variancia do
    Laplaciano do anel e derrubar as razoes para perto de zero, o que se le como
    "lesao lisa demais" quando na verdade e o anel que esta contaminado.
    """
    from .quality import _highpass, segment_lesion

    x1, y1, x2, y2 = geometry["x1"], geometry["y1"], geometry["x2"], geometry["y2"]
    pad_y = max(16, int(ring_frac * (y2 - y1)))
    pad_x = max(16, int(ring_frac * (x2 - x1)))
    cy0, cy1 = max(0, y1 - pad_y), min(native.shape[0], y2 + pad_y)
    cx0, cx1 = max(0, x1 - pad_x), min(native.shape[1], x2 + pad_x)

    context = native[cy0:cy1, cx0:cx1].astype(np.float32)
    context_orig = original[cy0:cy1, cx0:cx1].astype(np.float32)
    sub_mask = mask[cy0:cy1, cx0:cx1]
    roi = sub_mask > 127
    ring = (~roi) & (context > background_level)
    if roi.sum() < 64 or ring.sum() < 64:
        return {}

    lesion = segment_lesion(context_orig, context, sub_mask)
    if not lesion.any():
        lesion = roi

    hp = _highpass(context)
    lap = cv.Laplacian(context, cv.CV_32F)
    eps = 1e-6
    lesion_mean, ring_mean = float(context[lesion].mean()), float(context[ring].mean())
    ring_std = float(context[ring].std())
    return {
        "native_noise_ratio_roi_ring": float(hp[roi].std() / (hp[ring].std() + eps)),
        "native_sharpness_ratio": float(lap[roi].var() / (lap[ring].var() + eps)),
        "native_weber_contrast": float((lesion_mean - ring_mean) / (ring_mean + eps)),
        "native_cnr": float((lesion_mean - ring_mean) / (ring_std + eps)),
        "native_lesion_area_px": float(lesion.sum()),
    }


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Grava PNG preservando 8 ou 16 bits.

    ``cv.imwrite`` grava uint16 corretamente em PNG; o Pillow e usado como
    reserva porque algumas builds de OpenCV sem suporte a PNG de 16 bits falham
    silenciosamente devolvendo False.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), image):
        mode = "I;16" if image.dtype == np.uint16 else "L"
        Image.fromarray(image, mode=mode).save(str(path))


__all__ = [
    "SourceImage", "read_source", "build_breast_canvas", "lesion_delta",
    "writeback_native", "native_lesion_mask", "native_texture_metrics", "save_image",
]
