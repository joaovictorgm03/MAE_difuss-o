"""Janela quadrada na resolucao original, onde a lesao e gerada no modo nativo.

Por que existe
--------------
No modo canvas o modelo ve a mamografia inteira reduzida (~6x nas imagens de 16
bits): uma lesao BI-RADS 4/5 tipica vira ~48 px na entrada do modelo e o detalhe
fino que a define -- espiculas, distorcao, microcalcificacoes -- nao sobrevive.
Medido nos 67 recortes reais: correlacao do detalhe fino entre original e
reconstrucao de 0,06 na escala do canvas contra 0,81 numa janela em resolucao
nativa. Aqui o modelo recebe so a vizinhanca da lesao, em (quase) resolucao
original.

Contrato
--------
Este modulo e a **unica** implementacao da geometria nativo <-> janela <-> modelo
e da volta do delta a imagem original. O treino do LoRA
(:mod:`mamelesion.paste`), a avaliacao (:mod:`mamelesion.lesion_eval`) e o
pipeline importam daqui. Uma divergencia entre eles -- outro fator de contexto,
outra grade de alinhamento, outra escala de intensidade -- faria o modelo ser
usado fora da distribuicao em que foi ajustado, sem erro nenhum aparecer.

Unidades de intensidade: ``raw / ref`` em [0, 1], com ``ref`` = valor de
saturacao da imagem (65535 nas mamografias de 16 bits desta fonte, 255 nas de 8
bits). A entrada do modelo e a quantizacao disso para uint8, como no treino do
checkpoint original.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import cv2 as cv
import numpy as np

from .compose import border_offset, feather_alpha
from .masks import LesionBox

#: um pixel do latente corresponde a 8 pixels na entrada de 512 do modelo
LATENT_GRID = 8


@dataclass(frozen=True)
class Window:
    """Quadrado ``side x side`` da imagem original, reamostrado para ``model_res``."""

    x0: int
    y0: int
    side: int
    model_res: int = 512

    @property
    def scale(self) -> float:
        """Fator nativo -> modelo (1,0 quando a janela ja tem o lado do modelo)."""
        return self.model_res / float(self.side)

    @property
    def x1(self) -> int:
        return self.x0 + self.side

    @property
    def y1(self) -> int:
        return self.y0 + self.side

    def to_model(self, x: float, y: float) -> Tuple[float, float]:
        return ((x - self.x0) * self.scale, (y - self.y0) * self.scale)

    def to_native(self, u: float, v: float) -> Tuple[float, float]:
        return (self.x0 + u / self.scale, self.y0 + v / self.scale)

    def to_dict(self) -> Dict[str, float]:
        out = asdict(self)
        out["scale_native_to_model"] = round(self.scale, 6)
        return out


def plan_window(center_xy: Tuple[float, float], lesion_wh: Tuple[int, int],
                image_shape: Tuple[int, int], context_factor: float = 2.0,
                model_res: int = 512, min_scale: float = 0.75,
                min_context: float = 1.25) -> Window:
    """Janela centrada na lesao, com ``context_factor`` vezes o maior lado dela.

    O lado nunca e menor que ``model_res``: lesoes pequenas ficam em resolucao
    nativa exata (sem reamostragem); so as grandes sao reduzidas.

    **Teto de reducao.** Contexto maior custa detalhe. Medido nos 59 recortes de
    treino apos o VAE (``scripts/analise_escala_vae.py``), correlacao do detalhe
    fino mediana / p10:

    ============================  =======  =====
    politica                      mediana  p10
    ============================  =======  =====
    contexto 1,5                  0,746    0,598
    contexto 2,0                  0,697    0,477
    contexto 2,0, teto 0,75       0,726    0,635
    ============================  =======  =====

    Por isso, quando o contexto pedido exigiria reduzir a lesao abaixo de
    ``min_scale``, e o **contexto** que encolhe (ate ``min_context``), nao a
    lesao. A lesao tipica (323 px) nao e afetada e mantem a mascara em ~36% da
    janela; so as maiores trocam contexto por detalhe. ``min_scale <= 0``
    desliga o teto.

    A janela e deslocada -- nunca encolhida -- para caber na imagem, entao a
    lesao pode ficar fora do centro perto das bordas, mas continua inteira nela.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    lw, lh = int(lesion_wh[0]), int(lesion_wh[1])
    if lw < 1 or lh < 1:
        raise ValueError("lesao degenerada: {}x{}".format(lw, lh))

    longest = max(lw, lh)
    side = max(model_res, int(math.ceil(longest * context_factor)))
    if min_scale > 0:
        cap = max(int(math.ceil(model_res / min_scale)), int(math.ceil(longest * min_context)))
        side = min(side, cap)
    side = max(side, longest)
    side = min(side, h, w)
    if max(lw, lh) > side:
        raise ValueError("lesao de {}x{} px nao cabe numa imagem {}x{}".format(lw, lh, w, h))

    cx, cy = center_xy
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = min(max(0, x0), w - side)
    y0 = min(max(0, y0), h - side)
    return Window(x0, y0, side, model_res)


def lesion_box_model(window: Window, center_xy: Tuple[float, float],
                     lesion_wh: Tuple[int, int], margin_frac: float = 0.10,
                     grid: int = LATENT_GRID) -> LesionBox:
    """Caixa da mascara na entrada do modelo: pegada da lesao + margem, na grade.

    A margem da ao modelo espaco para a transicao lesao -> tecido. O alinhamento
    a grade do latente e **para fora** (a caixa so cresce), para que a borda da
    mascara nao caia no meio de um pixel latente -- o que vaza meio-tom na
    fronteira. Coordenadas semiabertas: ``[x1, x2) x [y1, y2)``.
    """
    lw, lh = lesion_wh
    cx, cy = window.to_model(*center_xy)
    half_w = lw * window.scale / 2.0
    half_h = lh * window.scale / 2.0
    margin = margin_frac * max(lw, lh) * window.scale

    x1 = int(math.floor((cx - half_w - margin) / grid)) * grid
    y1 = int(math.floor((cy - half_h - margin) / grid)) * grid
    x2 = int(math.ceil((cx + half_w + margin) / grid)) * grid
    y2 = int(math.ceil((cy + half_h + margin) / grid)) * grid
    res = window.model_res
    return LesionBox(max(0, x1), max(0, y1), min(res, x2), min(res, y2))


def box_mask(box: LesionBox, size: int) -> np.ndarray:
    """Mascara uint8 0/255 exatamente sobre ``[x1, x2) x [y1, y2)``.

    Nao usa ``cv.rectangle``, que desenha ``x2`` e ``y2`` inclusive e deixaria a
    mascara um pixel maior que a caixa gravada nos metadados.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[box.y1:box.y2, box.x1:box.x2] = 255
    return mask


def reference_max(raw: np.ndarray) -> float:
    """Valor de saturacao usado para levar ``raw`` a [0, 1].

    Imagens de 16 bits usam o maximo observado, como :func:`native.read_source`
    (nas mamografias desta fonte ele e 65535 em 15 de 15). Imagens de 8 bits
    usam 255.
    """
    if raw.dtype == np.uint8:
        return 255.0
    return float(raw.max()) or 1.0


def extract(raw: np.ndarray, window: Window, ref: float) -> np.ndarray:
    """Recorta a janela e a leva a ``model_res``, em float32 [0, 1]."""
    sub = raw[window.y0:window.y1, window.x0:window.x1].astype(np.float32) / float(ref)
    if sub.shape != (window.side, window.side):
        raise ValueError("janela fora da imagem: {} em {}".format(window, raw.shape))
    if window.side != window.model_res:
        sub = cv.resize(sub, (window.model_res, window.model_res), interpolation=cv.INTER_AREA)
    return np.clip(sub, 0.0, 1.0)


def to_u8(img01: np.ndarray) -> np.ndarray:
    """[0, 1] -> uint8, arredondando (a entrada do modelo e sempre 8 bits)."""
    return np.clip(np.rint(img01 * 255.0), 0, 255).astype(np.uint8)


def window_delta(original_u8: np.ndarray, generated_u8: np.ndarray, mask: np.ndarray,
                 box: LesionBox, feather_frac: float = 0.08, gain: float = 1.0,
                 offset_correction: bool = True) -> np.ndarray:
    """Delta da lesao na janela, em niveis uint8 (float32), ja com o alpha aplicado.

    Nao ha transferencia de detalhe: na escala nativa a granulacao do detector ja
    esta na imagem original, que entra inteira na soma de
    :func:`writeback`. A transicao suave acontece dentro da mascara
    (:func:`compose.feather_alpha`), com largura proporcional ao menor lado da
    caixa -- estreita de proposito (~ a margem da mascara): uma faixa larga
    atenuaria a periferia da lesao, que e onde estao as espiculas.
    """
    diff = generated_u8.astype(np.float32) - original_u8.astype(np.float32)
    if offset_correction:
        diff = diff - border_offset(diff, mask, band_px=max(4, int(0.05 * min(box.w, box.h))))
    feather_px = max(2, int(round(feather_frac * min(box.w, box.h))))
    alpha = feather_alpha(mask, feather_px, box, max_frac=0.5)
    return alpha * float(gain) * diff


def restore_native_margin(original_u8: np.ndarray, generated_u8: np.ndarray,
                          footprint_model: Tuple[float, float, float, float], sigma: float = 3.0,
                          r_in: float = 0.9, r_out: float = 1.15) -> np.ndarray:
    """Saida do modelo com a textura fina da **margem** trocada pela do detector original.

    A mascara vai alem da lesao, e o tecido dessa margem e redesenhado pelo
    modelo e passa pelo VAE, que o suaviza. Medido no lote de producao: ruido
    0,82 e textura fina 0,80 do tecido original em volta -- um "remendo" liso ao
    redor da lesao, sem degrau de nivel nem aresta.

    Aqui a mudanca do modelo ``d = gerado - original`` e dividida em baixa
    frequencia (``G_sigma * d``, escala em px da entrada do modelo) e o resto.
    Fora do corpo da lesao so a baixa frequencia e mantida, entao a textura fina
    volta a ser a da imagem original; dentro da pegada (superelipse de raio
    ``r_in``, rampa suave ate ``r_out``) a mudanca inteira e mantida, porque
    passar a lesao inteira por passa-baixa piorava a textura dela (W1 0,42 ->
    0,71). Validacao: ruido na margem 0,83 -> 0,96, textura 0,80 -> 0,91, lesao
    com KID, textura e contraste dentro do ruido.

    ``footprint_model`` = ``(u0, v0, u1, v1)`` da lesao na entrada do modelo.
    Devolve float32, pronto para :func:`window_delta`. ``sigma <= 0`` devolve a
    saida do modelo sem alteracao.
    """
    gen = generated_u8.astype(np.float32)
    if sigma <= 0:
        return gen
    orig = original_u8.astype(np.float32)
    d = gen - orig
    low = cv.GaussianBlur(d, (0, 0), float(sigma))
    h, w = d.shape
    u0, v0, u1, v1 = footprint_model
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx + 0.5 - (u0 + u1) / 2.0) / max((u1 - u0) / 2.0, 1.0)
    y = (yy + 0.5 - (v0 + v1) / 2.0) / max((v1 - v0) / 2.0, 1.0)
    r = (np.abs(x) ** 4 + np.abs(y) ** 4) ** 0.25
    t = np.clip((r_out - r) / (r_out - r_in), 0.0, 1.0)
    keep = t * t * (3.0 - 2.0 * t)
    return orig + low + keep * (d - low)


def floor_low_frequency(original_u8: np.ndarray, source: np.ndarray, floor: Optional[float] = -3.0,
                        sigma: float = 3.0) -> np.ndarray:
    """Impede que a lesao retire densidade do tecido: a mudanca de baixa frequencia fica >= ``floor``.

    Os recortes BI-RADS 5 reais sao justos e trazem a gordura escura da propria
    paciente em volta da massa; o modelo aprendeu "massa clara + halo escuro" e
    passou a *escurecer* o tecido do hospedeiro ao redor da lesao. Medido no lote
    de producao: periferia da lesao escurecida >= 8 niveis em 22% (8 bits) e 7%
    (16 bits) dos pixels, ate -29 niveis. Fisicamente uma lesao soma atenuacao;
    as areas gordurosas que ja existem continuam escuras porque vem da imagem
    original, a que o delta e somado.

    So a baixa frequencia (``G_sigma`` da mudanca, px do modelo, em niveis uint8)
    e limitada; a textura fina mantem o sinal. Validacao: escurecimento 22% -> 0%
    com KID, textura e contraste da lesao dentro do ruido. ``floor=None`` desliga.
    """
    src = source.astype(np.float32)
    if floor is None:
        return src
    orig = original_u8.astype(np.float32)
    d = src - orig
    low = cv.GaussianBlur(d, (0, 0), float(sigma))
    return orig + np.maximum(low, float(floor)) + (d - low)


def writeback(raw: np.ndarray, window: Window, delta_u8: np.ndarray, mask: np.ndarray,
              ref: float) -> Tuple[np.ndarray, np.ndarray]:
    """Soma o delta da janela a imagem original, na profundidade original.

    Garantia estrutural: fora da mascara o delta e **zero exato** antes da soma,
    entao esses pixels saem bit a bit identicos. A mascara e reamostrada por
    vizinho mais proximo e o delta por bilinear -- nunca Lanczos, cujos lobulos
    negativos espalham alteracoes para fora da regiao.

    Devolve ``(imagem_nativa, mascara_nativa_0_255)``; a mascara e exatamente a
    que delimitou a alteracao.
    """
    side = window.side
    delta = delta_u8.astype(np.float32)
    msk = mask
    if side != window.model_res:
        delta = cv.resize(delta, (side, side), interpolation=cv.INTER_LINEAR)
        msk = cv.resize(mask, (side, side), interpolation=cv.INTER_NEAREST)
    inside = msk > 127
    delta = np.where(inside, delta, 0.0) * (float(ref) / 255.0)

    out = raw.copy()
    info = np.iinfo(raw.dtype)
    region = out[window.y0:window.y1, window.x0:window.x1]
    patch = region.astype(np.float64) + delta
    # arredondar so dentro: fora a soma e com zero e o valor ja e inteiro
    patch = np.where(inside, np.rint(patch), patch)
    out[window.y0:window.y1, window.x0:window.x1] = np.clip(patch, info.min, info.max).astype(raw.dtype)

    applied = np.zeros(raw.shape[:2], dtype=np.uint8)
    applied[window.y0:window.y1, window.x0:window.x1] = np.where(inside, 255, 0).astype(np.uint8)
    return out, applied


def native_footprint(center_xy: Tuple[float, float], lesion_wh: Tuple[int, int],
                     image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Retangulo ``(x0, y0, x1, y1)`` nativo de tamanho ``lesion_wh`` centrado no ponto."""
    lw, lh = int(lesion_wh[0]), int(lesion_wh[1])
    h, w = image_shape[:2]
    x0 = int(round(center_xy[0] - lw / 2.0))
    y0 = int(round(center_xy[1] - lh / 2.0))
    x0 = min(max(0, x0), w - lw)
    y0 = min(max(0, y0), h - lh)
    return x0, y0, x0 + lw, y0 + lh


def anatomical_center(pmap, framing, mm_per_px_canvas: float, lesion_wh: Tuple[int, int],
                      rng: np.random.Generator, anatomy_cfg=None, allow_fallback: bool = True,
                      relaxed_tries: int = 600) -> Tuple[Tuple[float, float], Dict[str, object]]:
    """Centro nativo para uma lesao de ``lesion_wh`` px nativos, pelo mapa anatomico.

    Treino e inferencia chamam esta funcao, entao as lesoes coladas no treino
    caem nos mesmos tipos de tecido em que o modelo vai gera-las. Tres niveis,
    registrados em ``placement``:

    ``anatomia``
        a regra do modo canvas (:func:`anatomy.sample_anatomical_box`): centro
        proporcional ao score e **todo** pixel da caixa plausivel, inclusive a
        exigencia de tecido denso.
    ``anatomia_relaxada``
        centro ainda proporcional ao score (cai em tecido denso), mas a caixa so
        precisa respeitar as exclusoes duras: inteira dentro da mama, sem tocar o
        peitoral e a pelo menos ``min_skin_distance_mm`` da pele. Necessario para
        lesoes reais grandes: medido no treino, 35% das amostras nao passavam no
        nivel estrito porque a periferia de uma lesao de 300 px sempre pega
        gordura -- o que e anatomicamente normal.
    ``fallback``
        so o centro, entre pixels validos. Ultimo recurso; com
        ``allow_fallback=False`` levanta ``ValueError``.
    """
    from .anatomy import AnatomyConfig, sample_anatomical_box

    cfg = anatomy_cfg or AnatomyConfig()
    size = pmap.score.shape[0]
    s = framing.scale
    wc, hc = max(1.0, lesion_wh[0] * s), max(1.0, lesion_wh[1] * s)
    diameter_mm = math.sqrt(wc * hc) * mm_per_px_canvas
    aspect = wc / hc
    sampled = sample_anatomical_box(pmap, mm_per_px_canvas, (diameter_mm, diameter_mm),
                                    (aspect, aspect), cfg, rng, canvas=size)
    if sampled is not None:
        box, diag = sampled
        cx, cy = (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0
        info: Dict[str, object] = {"placement": "anatomia", "canvas_box": box.to_dict()}
        info.update({k: round(float(v), 4) for k, v in diag.items()})
        return framing.canvas_to_orig(cx, cy), info

    weights = pmap.score.astype(np.float64).ravel()
    total = weights.sum()
    if total > 0:
        half_w, half_h = int(math.ceil(wc / 2.0)), int(math.ceil(hc / 2.0))
        picks = rng.choice(weights.size, size=min(relaxed_tries, weights.size), replace=True, p=weights / total)
        for idx in picks:
            cy, cx = int(idx // size), int(idx % size)
            x1, y1, x2, y2 = cx - half_w, cy - half_h, cx + half_w, cy + half_h
            if x1 < 0 or y1 < 0 or x2 > size or y2 > size:
                continue
            if (pmap.breast[y1:y2, x1:x2] > 0).mean() < 0.999:
                continue
            if (pmap.pectoral[y1:y2, x1:x2] > 0).any():
                continue
            if float(pmap.skin_distance_mm[y1:y2, x1:x2].min()) < cfg.min_skin_distance_mm:
                continue
            box = LesionBox(x1, y1, x2, y2)
            return framing.canvas_to_orig(float(cx), float(cy)), {
                "placement": "anatomia_relaxada", "canvas_box": box.to_dict(),
                "anat_score_mean": round(float(pmap.score[y1:y2, x1:x2].mean()), 4),
                "skin_distance_mm_min": round(float(pmap.skin_distance_mm[y1:y2, x1:x2].min()), 4)}

    if not allow_fallback:
        raise ValueError("nenhuma posicao anatomicamente plausivel para {}x{} px".format(*lesion_wh))
    ys, xs = np.nonzero(pmap.score > 0)
    if len(xs) == 0:
        raise ValueError("mapa de plausibilidade vazio")
    i = int(rng.integers(len(xs)))
    return framing.canvas_to_orig(float(xs[i]), float(ys[i])), {"placement": "fallback"}


__all__ = ["Window", "LATENT_GRID", "plan_window", "lesion_box_model", "box_mask",
           "reference_max", "extract", "to_u8", "window_delta", "restore_native_margin", "floor_low_frequency", "writeback",
           "native_footprint", "anatomical_center"]
