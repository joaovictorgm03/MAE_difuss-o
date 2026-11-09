"""Amostras de treino do LoRA: recorte real BI-RADS colado em tecido saudavel.

Por que colagem
---------------
O ajuste de inpainting precisa de ``(imagem, mascara)`` em que a imagem contem a
lesao e o entorno e tecido real. So temos recortes justos da lesao, sem a
mamografia de origem. Colar o recorte numa janela de mamografia saudavel **da
mesma fonte** (mesmo detector, mesma escala de 16 bits) produz exatamente o par
que a inferencia vai usar: contexto saudavel ao redor, lesao sob a mascara.

O que a colagem nao ensina: como a lesao deforma o tecido *ao redor* dela
(retracao, distorcao alem da caixa), porque o entorno e de outra paciente. Essa
limitacao esta registrada em ``docs/FINETUNING.md``.

Contrato com a inferencia
-------------------------
Toda a geometria vem de :mod:`mamelesion.window` -- janela, caixa da mascara,
alinhamento a grade, escala de intensidade e posicionamento anatomico. Este
modulo so acrescenta o que e proprio do treino: aumentos, casamento de nivel e a
colagem com borda suave.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2 as cv
import numpy as np

from .anatomy import AnatomyConfig, PlausibilityMap, plausibility_map
from .birads_data import LesionCrop, parse_ids
from .config import Config, NativeWindowConfig
from .masks import LesionBox
from .native import build_breast_canvas, read_source
from .preprocessing import Framing, estimate_mm_per_px
from .window import (Window, anatomical_center, box_mask, lesion_box_model, native_footprint,
                     plan_window, reference_max, to_u8)


@dataclass
class PasteConfig:
    """Parametros das amostras. Os de geometria espelham ``native_window``."""

    model_res: int = 512
    context_factor: float = 2.0
    min_scale: float = 0.75
    min_context: float = 1.25
    mask_margin_frac: float = 0.10
    paste_feather_frac: float = 0.15
    #: fator de escala sorteado para o recorte; pequeno para nao mudar a
    #: distribuicao de tamanhos que o catalogo da inferencia reproduz
    scale_jitter: Tuple[float, float] = (0.90, 1.10)
    #: espelhamentos preservam a textura da lesao; rotacoes de 90 graus nao,
    #: porque o detector desta fonte tem listras horizontais que viram verticais
    flip_horizontal: bool = True
    flip_vertical: bool = True
    #: casamento de nivel do fundo do recorte com o tecido que ele cobre:
    #: "plano" ajusta nivel e gradiente; "mediana" so o nivel; "nenhum" desliga
    level_match: str = "plano"
    #: forma da transicao da colagem: "superelipse" (sem cantos retos) ou "retangulo"
    alpha_shape: str = "superelipse"
    #: sem ultimo recurso no treino: amostra que so caberia sobre peitoral ou pele
    #: levanta ValueError e o dataset sorteia outro par recorte/hospedeiro. Custo
    #: medido: lesoes > 600 px sao aceitas em ~50% das tentativas.
    allow_fallback: bool = False

    @classmethod
    def from_native_window(cls, nw: NativeWindowConfig, model_res: int = 512, **kw) -> "PasteConfig":
        return cls(model_res=model_res, context_factor=nw.context_factor,
                   min_scale=nw.min_scale, min_context=nw.min_context,
                   mask_margin_frac=nw.mask_margin_frac,
                   paste_feather_frac=nw.paste_feather_frac, **kw)


@dataclass
class Host:
    """Mamografia saudavel pronta para receber colagens."""

    name: str
    patient: str
    raw: np.ndarray = field(repr=False)
    ref: float
    framing: Framing
    pmap: PlausibilityMap = field(repr=False)
    mm_per_px_canvas: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.raw.shape[:2]


def prepare_host(path: Path, cfg: Config, anatomy_cfg: Optional[AnatomyConfig] = None,
                 canvas_size: int = 1024) -> Host:
    """Le a mamografia e constroi o mapa anatomico com as mesmas regras do pipeline."""
    from .pipeline import parse_view        # import tardio: pipeline importa este modulo

    path = Path(path)
    source = read_source(path, cfg.preprocessing.max_value)
    canvas, framing, breast = build_breast_canvas(
        source.gray_u8, canvas_size, breast_threshold=cfg.preprocessing.breast_threshold)
    mm = cfg.preprocessing.mm_per_px or estimate_mm_per_px(
        framing, cfg.preprocessing.detector_long_side_mm, source.pixel_spacing_mm)
    pmap = plausibility_map(canvas, breast, float(mm), parse_view(path.stem).view, anatomy_cfg)
    try:
        patient, _ = parse_ids(path.name)
    except ValueError:                      # outra convencao (ex.: 1217_IM-0002_CCE_L): 1o campo = paciente
        patient = path.stem.split("_")[0]
    return Host(path.stem, patient, source.raw, reference_max(source.raw), framing, pmap, float(mm))


@dataclass
class TrainingSample:
    """Um par de treino na resolucao do modelo."""

    image_u8: np.ndarray            # janela com a lesao colada, uint8 model_res^2
    mask_u8: np.ndarray             # 0/255
    box: LesionBox                  # caixa da mascara na entrada do modelo
    window: Window
    footprint: Tuple[int, int, int, int]
    crop_name: str
    lesion_class: str
    host_name: str
    level_offset: float
    placement: str
    #: janela composta na resolucao nativa, antes de ir ao modelo (so com keep_native)
    native_composite: Optional[np.ndarray] = field(default=None, repr=False)

    def meta(self) -> Dict[str, object]:
        return {"crop": self.crop_name, "classe": self.lesion_class, "host": self.host_name,
                "janela": self.window.to_dict(), "caixa_modelo": self.box.to_dict(),
                "pegada_nativa": list(self.footprint), "offset_nivel": round(self.level_offset, 5),
                "posicionamento": self.placement}


def feather_rect(h: int, w: int, feather_px: int) -> np.ndarray:
    """Alpha retangular: 1 no interior, rampa linear ate ~0 nas bordas do recorte."""
    feather_px = max(1, min(int(feather_px), (min(h, w) - 1) // 2 or 1))
    dy = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    dx = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    dist = np.minimum(dy[:, None], dx[None, :])
    return np.clip((dist + 0.5) / float(feather_px), 0.0, 1.0)


#: raio superelipse a partir do qual o recorte e considerado fundo (anel externo)
BACKGROUND_RADIUS = 0.85


def superellipse_radius(h: int, w: int, power: float = 4.0) -> np.ndarray:
    """Raio normalizado da superelipse: 0 no centro, 1 no meio das arestas, >1 nos cantos."""
    y = (np.arange(h, dtype=np.float32) + 0.5) / h * 2.0 - 1.0
    x = (np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0
    return (np.abs(x)[None, :] ** power + np.abs(y)[:, None] ** power) ** (1.0 / power)


def superellipse_alpha(h: int, w: int, feather_frac: float, power: float = 4.0) -> np.ndarray:
    """Alpha em forma de superelipse com rampa suave (smoothstep), 1 no centro e 0 nos cantos.

    Um alpha retangular deixa as quatro arestas do recorte visiveis quando o fundo
    dele nao casa exatamente com o hospedeiro -- e o modelo aprenderia a gerar um
    bloco retangular. A superelipse (``power`` 4) cobre quase todo o recorte mas
    nao tem arestas retas; a rampa ocupa ``feather_frac`` do semi-eixo.
    """
    r = superellipse_radius(h, w, power)
    t = np.clip((1.0 - r) / max(1e-3, 2.0 * feather_frac), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _plane(values: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Coeficientes (a, b, c) de ``a + b*x + c*y`` por minimos quadrados."""
    A = np.stack([np.ones_like(xs), xs, ys], axis=1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, values.astype(np.float64), rcond=None)
    return coef


def level_correction(region: np.ndarray, lesion: np.ndarray, band: np.ndarray, mode: str) -> np.ndarray:
    """Campo a somar ao recorte para que seu fundo (``band``) case com o tecido coberto.

    ``plano`` ajusta nivel e gradiente linear -- corrige o recorte cujo fundo
    escurece de um lado, caso comum nos recortes justos; ``mediana`` so desloca o
    nivel. O contraste lesao/entorno do recorte e preservado nos dois.
    """
    h, w = lesion.shape
    if mode == "nenhum" or not band.any():
        return np.zeros((h, w), np.float32)
    if mode == "mediana":
        return np.full((h, w), float(np.median(region[band]) - np.median(lesion[band])), np.float32)
    if mode != "plano":
        raise ValueError("level_match desconhecido: " + str(mode))
    ys, xs = np.nonzero(band)
    if len(ys) > 4000:                                   # subamostra: o ajuste e linear
        keep = np.linspace(0, len(ys) - 1, 4000).astype(int)
        ys, xs = ys[keep], xs[keep]
    diff = region[ys, xs] - lesion[ys, xs]
    a, b, c = _plane(diff, ys / max(h - 1, 1), xs / max(w - 1, 1))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (a + b * xx / max(w - 1, 1) + c * yy / max(h - 1, 1)).astype(np.float32)


def augment_crop(image: np.ndarray, rng: np.random.Generator, cfg: PasteConfig) -> np.ndarray:
    """Espelhamentos e escala leve. Nunca altera intensidade."""
    out = image
    if cfg.flip_horizontal and rng.random() < 0.5:
        out = out[:, ::-1]
    if cfg.flip_vertical and rng.random() < 0.5:
        out = out[::-1, :]
    lo, hi = cfg.scale_jitter
    if hi > lo:
        s = float(rng.uniform(lo, hi))
        h, w = out.shape
        nw, nh = max(8, int(round(w * s))), max(8, int(round(h * s)))
        interp = cv.INTER_AREA if s < 1 else cv.INTER_LINEAR
        out = cv.resize(np.ascontiguousarray(out), (nw, nh), interpolation=interp)
    return np.ascontiguousarray(out, dtype=np.float32)


def make_sample(crop: LesionCrop, host: Host, rng: np.random.Generator, cfg: PasteConfig,
                anatomy_cfg: Optional[AnatomyConfig] = None, keep_native: bool = False) -> TrainingSample:
    """Monta um par de treino: recorte aumentado, colado numa posicao plausivel do hospedeiro."""
    lesion = augment_crop(crop.image, rng, cfg)
    lh, lw = lesion.shape
    if max(lw, lh) > min(host.shape):
        raise ValueError("recorte {} maior que o hospedeiro {}".format(crop.name, host.name))

    center, info = anatomical_center(host.pmap, host.framing, host.mm_per_px_canvas,
                                     (lw, lh), rng, anatomy_cfg, allow_fallback=cfg.allow_fallback)
    fx0, fy0, fx1, fy1 = native_footprint(center, (lw, lh), host.shape)
    fcenter = ((fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0)
    window = plan_window(fcenter, (lw, lh), host.shape, cfg.context_factor, cfg.model_res,
                         cfg.min_scale, cfg.min_context)

    native = host.raw[window.y0:window.y1, window.x0:window.x1].astype(np.float32) / host.ref
    px0, py0 = fx0 - window.x0, fy0 - window.y0
    region = native[py0:py0 + lh, px0:px0 + lw]

    if cfg.alpha_shape == "superelipse":
        alpha = superellipse_alpha(lh, lw, cfg.paste_feather_frac)
    elif cfg.alpha_shape == "retangulo":
        alpha = feather_rect(lh, lw, max(2, int(round(cfg.paste_feather_frac * min(lh, lw)))))
    else:
        raise ValueError("alpha_shape desconhecido: " + str(cfg.alpha_shape))
    # fundo do recorte = anel externo, definido pela geometria e nao pela rampa: uma
    # rampa larga incluiria a periferia da lesao e o ajuste passaria a atenua-la
    band = superellipse_radius(lh, lw) > BACKGROUND_RADIUS
    correction = level_correction(region, lesion, band, cfg.level_match)
    offset = float(np.median(correction)) if correction.size else 0.0
    lesion = np.clip(lesion + correction, 0.0, 1.0)
    native[py0:py0 + lh, px0:px0 + lw] = region * (1.0 - alpha) + lesion * alpha
    composite = native.copy() if keep_native else None

    if window.side != cfg.model_res:
        native = cv.resize(native, (cfg.model_res, cfg.model_res), interpolation=cv.INTER_AREA)
    box = lesion_box_model(window, fcenter, (lw, lh), cfg.mask_margin_frac)
    return TrainingSample(to_u8(native), box_mask(box, cfg.model_res), box, window,
                          (fx0, fy0, fx1, fy1), crop.name, crop.lesion_class, host.name,
                          offset, str(info.get("placement", "anatomia")), composite)


__all__ = ["PasteConfig", "Host", "prepare_host", "TrainingSample", "feather_rect",
           "superellipse_radius", "superellipse_alpha", "level_correction", "augment_crop", "make_sample"]
