"""Plausibilidade anatomica para posicionamento de lesoes sinteticas.

O modulo :mod:`mamelesion.masks` sorteia caixas *uniformemente* dentro da
mascara da mama. Isso e suficiente para reproduzir o artigo, mas produz lesoes
em lugares onde elas nao ocorrem: dentro do musculo peitoral, encostadas na
linha da pele, na ponta do mamilo ou no meio de gordura pura.

Aqui a regiao valida e construida como um *mapa de plausibilidade* em [0, 1],
produto de quatro fatores independentes:

``f_pele``      distancia da borda da mama (linha de pele), em milimetros.
                Lesoes a menos de ``min_skin_distance_mm`` da pele sao
                rejeitadas; entre esse valor e ``skin_ramp_mm`` a preferencia
                cresce linearmente.
``f_peitoral``  zero dentro do musculo peitoral e na sua margem de seguranca.
                O peitoral so e procurado em incidencias obliquas (MLO/OE),
                onde ele aparece; em CC nao ha peitoral visivel na maioria dos
                exames.
``f_densidade`` posto percentual da intensidade local dentro do tecido mamario.
                Tecido denso (glandular/fibroglandular) e claro na convencao
                usada pelo pre-processamento; gordura e escura. Lesoes do tipo
                massa/nodulo nascem em tecido fibroglandular.
``f_glandular`` fracao de tecido denso numa vizinhanca do ponto. Separa
                "denso" de "denso isolado": um pixel claro cercado de gordura
                e provavelmente ruido ou vaso, nao parenquima.

A amostragem e proporcional ao mapa, de modo que as lesoes se distribuem pelo
parenquima em vez de se concentrarem num unico ponto de maximo.

Coordenadas: todas no canvas quadrado (ver :mod:`mamelesion.preprocessing`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2 as cv
import numpy as np

from .masks import LesionBox, coverage_inside_breast, mm_to_px

EPS = 1e-6


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------


@dataclass
class AnatomyConfig:
    """Parametros das regras de plausibilidade anatomica.

    Os valores padrao foram escolhidos a partir da pratica radiologica descrita
    no BI-RADS (localizacao de achados) e conferidos visualmente nos mapas
    gerados para este conjunto: a faixa de 8 mm de pele exclui a derme e o
    tecido subcutaneo imediato, onde uma massa sintetica ficaria obviamente
    colada na borda.
    """

    min_skin_distance_mm: float = 8.0
    skin_ramp_mm: float = 20.0
    pectoral_margin_mm: float = 6.0
    detect_pectoral: bool = True
    pectoral_fallback: bool = True

    density_percentile: float = 45.0      # abaixo disso e considerado gordura
    density_ramp: float = 25.0            # largura da rampa em pontos percentuais
    glandular_window_mm: float = 12.0     # vizinhanca para medir "densidade isolada"
    min_glandular_fraction: float = 0.35  # fracao minima de tecido denso na vizinhanca
    #: preferencia por densidade intermediaria (None = desligada, comportamento
    #: original). Acima deste percentil o peso cai linearmente ate
    #: ``density_upper_floor`` no percentil 100: a lesao continua podendo cair no
    #: tecido mais denso, so fica menos provavel. Existe porque, sem ela, as
    #: lesoes se concentram no tecido mais denso (percentil mediano 88 no
    #: lote_birads1_v1) e ficam mascaradas.
    density_upper_percentile: Optional[float] = None
    density_upper_floor: float = 0.4

    min_box_score: float = 0.15           # score minimo aceito no pior pixel da caixa
    min_mean_score: float = 0.35          # score medio minimo dentro da caixa
    smooth_sigma_mm: float = 1.5          # suavizacao antes de medir densidade

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------------
# Geometria da mama
# ---------------------------------------------------------------------------


def chest_wall_side(breast: np.ndarray) -> str:
    """Lado do canvas onde esta a parede toracica ('left' ou 'right').

    A parede toracica e o lado em que a mama e *cortada* pelo enquadramento:
    ali a extensao vertical do tecido e maxima. Comparar a altura da mascara
    nas duas extremidades da caixa envolvente e mais robusto do que comparar
    massa por metade do canvas, que se confunde quando a mama e pequena e esta
    deslocada.
    """
    ys, xs = np.nonzero(breast > 0)
    if len(xs) == 0:
        return "left"
    x0, x1 = int(xs.min()), int(xs.max())
    band = max(3, (x1 - x0) // 20)
    left = int((breast[:, x0:x0 + band] > 0).sum())
    right = int((breast[:, max(x0, x1 - band):x1 + 1] > 0).sum())
    return "left" if left >= right else "right"


def nipple_point(breast: np.ndarray, side: str) -> Optional[Tuple[int, int]]:
    """Estima o mamilo como o ponto do contorno mais distante da parede toracica.

    E a definicao geometrica usada em CAD de mamografia. Falha em mamas muito
    achatadas, onde varios pontos empatam; nesses casos o empate e resolvido
    pela mediana das linhas, o que devolve o centro da regiao mais anterior.
    """
    ys, xs = np.nonzero(breast > 0)
    if len(xs) == 0:
        return None
    if side == "left":
        target = int(xs.max())
        rows = ys[xs >= target - 2]
    else:
        target = int(xs.min())
        rows = ys[xs <= target + 2]
    if len(rows) == 0:
        return None
    return int(target), int(np.median(rows))


def skin_distance_map(breast: np.ndarray, side: str) -> np.ndarray:
    """Distancia (em pixels) de cada ponto ate a linha de pele.

    A borda da mama junto a parede toracica **nao** e pele: e o limite do campo
    de visao do detector. Contar essa borda como pele excluiria toda a regiao
    retroglandular, que e tecido perfeitamente valido. Por isso a mascara e
    espelhada para fora daquele lado antes da transformada de distancia, o que
    empurra a fronteira artificial para longe.
    """
    h, w = breast.shape[:2]
    pad = max(8, w // 8)
    extended = np.zeros((h, w + pad), dtype=np.uint8)
    if side == "left":
        # parede toracica a esquerda: espelha a faixa inicial para a esquerda
        extended[:, pad:] = breast
        extended[:, :pad] = breast[:, :1]
        dist = cv.distanceTransform((extended > 0).astype(np.uint8), cv.DIST_L2, 5)
        return dist[:, pad:]
    extended[:, :w] = breast
    extended[:, w:] = breast[:, -1:]
    dist = cv.distanceTransform((extended > 0).astype(np.uint8), cv.DIST_L2, 5)
    return dist[:, :w]


# ---------------------------------------------------------------------------
# Musculo peitoral
# ---------------------------------------------------------------------------


def _triangle(corner_x: int, y0: int, a: int, b: int, side: str,
              shape: Tuple[int, int]) -> np.ndarray:
    """Triangulo peitoral definido pela reta (a, y0)--(corner_x, b)."""
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array([[corner_x, y0], [a, y0], [corner_x, b]], dtype=np.int32)
    cv.fillPoly(mask, [pts], 255)
    return mask


def _pectoral_by_edge(canvas: np.ndarray, breast: np.ndarray, side: str,
                      bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Procura a borda **reta** do peitoral por busca exaustiva de retas.

    O limite do peitoral em MLO e, com boa aproximacao, um segmento de reta que
    vai da borda superior da mama ate a lateral da parede toracica. Em vez de
    limiarizar por intensidade -- que confunde o musculo com parenquima denso e
    foi o que fez a deteccao por componentes conexas falhar neste conjunto --
    percorre-se uma grade de retas candidatas e escolhe-se a que acumula mais
    gradiente ao longo do proprio segmento.

    A reta vencedora so e aceita se destacar-se claramente das demais (o musculo
    tem uma borda unica e forte) e se o lado do musculo for mais claro que o
    lado do parenquima, que e a assinatura fisica esperada.
    """
    x0, y0, x1, y1 = bbox
    bh, bw = y1 - y0 + 1, x1 - x0 + 1
    if bh < 64 or bw < 64:
        return None

    blur = cv.GaussianBlur(canvas, (9, 9), 3)
    gx = cv.Sobel(blur, cv.CV_32F, 1, 0, ksize=5)
    gy = cv.Sobel(blur, cv.CV_32F, 0, 1, ksize=5)
    grad = cv.magnitude(gx, gy)

    # A borda do proprio detector e da pele produz o gradiente mais forte da
    # imagem. Sem excluir uma faixa junto ao contorno, a reta vencedora e sempre
    # um segmento curto colado nessa borda, e nao o limite do musculo.
    interior = cv.erode(breast, cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25)))
    grad = np.where(interior > 0, grad, 0.0)

    corner_x = x0 if side == "left" else x1
    a_values = np.linspace(x0 + 0.08 * bw, x0 + 0.80 * bw, 28) if side == "left" else \
        np.linspace(x1 - 0.80 * bw, x1 - 0.08 * bw, 28)
    b_values = np.linspace(y0 + 0.10 * bh, y0 + 0.90 * bh, 28)

    breast_area = float((breast > 0).sum())
    min_len = 0.30 * bh
    n = 240
    best_score, best = -np.inf, None
    scores = []
    for a in a_values:
        for b in b_values:
            length = float(np.hypot(a - corner_x, b - y0))
            if length < min_len:
                continue
            # area analitica do triangulo: filtro barato aplicado *durante* a
            # busca, para que um candidato invalido nao elimine os validos
            tri_area = 0.5 * abs(a - corner_x) * (b - y0)
            if not (0.03 * breast_area < tri_area < 0.40 * breast_area):
                continue
            xs = np.linspace(a, corner_x, n).astype(np.int32)
            ys = np.linspace(y0, b, n).astype(np.int32)
            np.clip(xs, 0, canvas.shape[1] - 1, out=xs)
            np.clip(ys, 0, canvas.shape[0] - 1, out=ys)
            if float((breast[ys, xs] > 0).mean()) < 0.85:
                continue
            score = float(grad[ys, xs].mean())
            scores.append(score)
            if score > best_score:
                best_score, best = score, (int(a), int(b))

    if best is None or len(scores) < 30:
        return None

    # a borda do peitoral e um maximo destacado, nao apenas o maior de um plato
    median, spread = float(np.median(scores)), float(np.std(scores))
    if spread <= 0 or (best_score - median) < 1.5 * spread:
        return None

    a, b = best
    mask = cv.bitwise_and(_triangle(corner_x, y0, a, b, side, canvas.shape[:2]), breast)
    area = float((mask > 0).sum())
    if not (0.03 * breast_area < area < 0.40 * breast_area):
        return None

    outside = cv.bitwise_and(cv.bitwise_not(mask), breast)
    band = cv.bitwise_and(cv.dilate(mask, np.ones((41, 41), np.uint8)), outside)
    if (mask > 0).sum() < 100 or (band > 0).sum() < 100:
        return None
    if float(canvas[mask > 0].mean()) - float(canvas[band > 0].mean()) < 3.0:
        return None                                   # musculo deve ser mais claro
    return mask


def detect_pectoral(canvas: np.ndarray, breast: np.ndarray, side: str,
                    fallback: bool = True) -> Tuple[np.ndarray, str]:
    """Mascara do musculo peitoral em incidencia obliqua.

    Tres estrategias, em ordem de confiabilidade:

    1. ``borda_reta`` -- busca da reta de maior gradiente entre a borda superior
       e a lateral da parede toracica (:func:`_pectoral_by_edge`).
    2. ``componente`` -- maior componente conexa clara ancorada no canto
       superior, util quando o musculo e muito mais denso que o parenquima e a
       borda e curva.
    3. ``geometrico`` -- triangulo conservador no canto, so quando o chamador
       sabe que a incidencia e obliqua e as duas anteriores falharam.

    Devolve ``(mascara, metodo)``; o metodo vai para os metadados de cada
    imagem, porque a confianca na exclusao depende de qual deles decidiu.
    """
    empty = np.zeros(breast.shape[:2], dtype=np.uint8)
    ys, xs = np.nonzero(breast > 0)
    if len(xs) == 0:
        return empty, "ausente"

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bh, bw = y1 - y0 + 1, x1 - x0 + 1
    area = float((breast > 0).sum())

    edge_mask = _pectoral_by_edge(canvas, breast, side, (x0, y0, x1, y1))
    if edge_mask is not None:
        return edge_mask, "borda_reta"

    values = canvas[breast > 0]
    if values.size == 0:
        return empty, "ausente"
    thresh = float(np.percentile(values, 82))

    bright = ((canvas >= thresh) & (breast > 0)).astype(np.uint8)
    bright = cv.morphologyEx(bright, cv.MORPH_OPEN,
                             cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9)))
    bright = cv.morphologyEx(bright, cv.MORPH_CLOSE,
                             cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)))

    n, labels, stats, centroids = cv.connectedComponentsWithStats(bright, connectivity=8)
    corner_x = x0 if side == "left" else x1
    best, best_area = 0, 0.0
    for i in range(1, n):
        comp_area = float(stats[i, cv.CC_STAT_AREA])
        if comp_area < 0.02 * area or comp_area > 0.40 * area:
            continue
        cy = float(centroids[i][1])
        if cy > y0 + 0.55 * bh:            # centroide deve ficar na parte superior
            continue
        top = stats[i, cv.CC_STAT_TOP]
        if top > y0 + 0.15 * bh:           # deve encostar no topo da mama
            continue
        left = stats[i, cv.CC_STAT_LEFT]
        right = left + stats[i, cv.CC_STAT_WIDTH]
        touches = (left <= corner_x + 0.10 * bw) if side == "left" else (right >= corner_x - 0.10 * bw)
        if not touches:
            continue
        if comp_area > best_area:
            best, best_area = i, comp_area

    if best:
        mask = np.zeros_like(empty)
        mask[labels == best] = 255
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(mask, contours, -1, 255, thickness=cv.FILLED)
        return mask, "componente"

    if not fallback:
        return empty, "ausente"

    # Fallback geometrico: triangulo conservador no canto superior da parede
    # toracica. Cobre o peitoral tipico sem invadir o parenquima central.
    tri_h = int(0.35 * bh)
    tri_w = int(0.30 * bw)
    mask = np.zeros_like(empty)
    if side == "left":
        pts = np.array([[x0, y0], [x0 + tri_w, y0], [x0, y0 + tri_h]], dtype=np.int32)
    else:
        pts = np.array([[x1, y0], [x1 - tri_w, y0], [x1, y0 + tri_h]], dtype=np.int32)
    cv.fillPoly(mask, [pts], 255)
    mask = cv.bitwise_and(mask, breast)
    return mask, "geometrico"


# ---------------------------------------------------------------------------
# Densidade do parenquima
# ---------------------------------------------------------------------------


def density_rank_map(canvas: np.ndarray, breast: np.ndarray, sigma_px: float) -> np.ndarray:
    """Posto percentual (0..100) da intensidade local dentro do tecido mamario.

    Trabalhar em posto e nao em intensidade absoluta torna o criterio invariante
    ao ganho do detector e ao bit depth da origem, o que importa aqui porque o
    conjunto mistura PNGs de 8 e 16 bits de aparelhos diferentes.
    """
    k = int(max(3, round(sigma_px * 6)) | 1)
    smooth = cv.GaussianBlur(canvas.astype(np.float32), (k, k), max(0.5, sigma_px))

    inside = breast > 0
    values = smooth[inside]
    if values.size == 0:
        return np.zeros_like(smooth)

    # posto por interpolacao sobre 256 quantis: barato e suficientemente fino
    quantiles = np.percentile(values, np.linspace(0, 100, 256))
    quantiles = np.maximum.accumulate(quantiles)  # garante monotonicidade
    rank = np.interp(smooth, quantiles, np.linspace(0, 100, 256))
    rank[~inside] = 0.0
    return rank.astype(np.float32)


def glandular_fraction_map(dense: np.ndarray, window_px: int) -> np.ndarray:
    """Fracao de tecido denso numa janela quadrada ao redor de cada pixel."""
    k = int(max(3, window_px) | 1)
    return cv.blur(dense.astype(np.float32), (k, k))


# ---------------------------------------------------------------------------
# Mapa de plausibilidade
# ---------------------------------------------------------------------------


@dataclass
class PlausibilityMap:
    """Mapa de plausibilidade e os componentes que o originaram."""

    score: np.ndarray                     # float32 em [0, 1]
    skin_distance_mm: np.ndarray
    density_rank: np.ndarray
    pectoral: np.ndarray
    breast: np.ndarray
    side: str
    nipple: Optional[Tuple[int, int]]
    pectoral_method: str
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return self.score > 0

    def valid_fraction(self) -> float:
        total = float((self.breast > 0).sum())
        if total <= 0:
            return 0.0
        return float((self.score > 0).sum() / total)


def _ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Rampa linear: 0 abaixo de ``lo``, 1 acima de ``hi``."""
    if hi <= lo:
        return (x >= hi).astype(np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def plausibility_map(canvas: np.ndarray, breast: np.ndarray, mm_per_px: float,
                     view: str = "unknown", cfg: Optional[AnatomyConfig] = None) -> PlausibilityMap:
    """Constroi o mapa de plausibilidade anatomica do canvas.

    Parametros
    ----------
    canvas : mamografia enquadrada, uint8, quadrada
    breast : mascara binaria do tecido (0/255), mesmo tamanho
    mm_per_px : escala do canvas, usada para converter todos os limiares
    view : 'CC', 'MLO' ou 'unknown'; decide se o peitoral e procurado
    """
    cfg = cfg or AnatomyConfig()
    side = chest_wall_side(breast)

    dist_px = skin_distance_map(breast, side)
    dist_mm = dist_px * mm_per_px
    f_skin = _ramp(dist_mm, cfg.min_skin_distance_mm, cfg.skin_ramp_mm)

    pect_method = "nao_aplicavel"
    pectoral = np.zeros_like(breast)
    if cfg.detect_pectoral and view.upper() in {"MLO", "OBLIQUA", "LATERAL", "UNKNOWN"}:
        pectoral, pect_method = detect_pectoral(
            canvas, breast, side, fallback=cfg.pectoral_fallback and view.upper() == "MLO")
    f_pect = np.ones_like(f_skin)
    if pectoral.any():
        margin = mm_to_px(cfg.pectoral_margin_mm, mm_per_px)
        grown = cv.dilate(pectoral, cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (margin * 2 + 1,) * 2))
        f_pect[grown > 0] = 0.0

    rank = density_rank_map(canvas, breast, cfg.smooth_sigma_mm / max(mm_per_px, EPS))
    f_dens = _ramp(rank, cfg.density_percentile, cfg.density_percentile + cfg.density_ramp)
    if cfg.density_upper_percentile is not None:
        f_dens = f_dens * (1.0 - (1.0 - cfg.density_upper_floor)
                           * _ramp(rank, cfg.density_upper_percentile, 100.0))

    dense = (rank >= cfg.density_percentile) & (breast > 0)
    window = mm_to_px(cfg.glandular_window_mm, mm_per_px)
    frac = glandular_fraction_map(dense, window)
    f_gland = _ramp(frac, cfg.min_glandular_fraction, min(1.0, cfg.min_glandular_fraction + 0.25))

    score = f_skin * f_pect * f_dens * f_gland
    score[breast == 0] = 0.0

    stats = {
        "chest_wall_side": side,
        "pectoral_method": pect_method,
        "pectoral_area_frac": float((pectoral > 0).sum() / max(1, (breast > 0).sum())),
        "valid_area_frac": float((score > 0).sum() / max(1, (breast > 0).sum())),
        "median_density_rank_valid": float(np.median(rank[score > 0])) if (score > 0).any() else 0.0,
    }
    return PlausibilityMap(score.astype(np.float32), dist_mm.astype(np.float32),
                           rank, pectoral, breast, side, nipple_point(breast, side),
                           pect_method, stats)


# ---------------------------------------------------------------------------
# Amostragem
# ---------------------------------------------------------------------------


def sample_anatomical_box(
    pmap: PlausibilityMap,
    mm_per_px: float,
    diameter_mm_range: Tuple[float, float],
    aspect_range: Tuple[float, float] = (0.75, 1.35),
    cfg: Optional[AnatomyConfig] = None,
    rng: Optional[np.random.Generator] = None,
    canvas: int = 1024,
    max_tries: int = 800,
) -> Optional[Tuple[LesionBox, Dict[str, float]]]:
    """Sorteia uma caixa cujo interior inteiro e anatomicamente plausivel.

    O centro e sorteado com probabilidade proporcional ao score, e a caixa so e
    aceita se **todos** os seus pixels tiverem score acima de
    ``cfg.min_box_score`` e a media ficar acima de ``cfg.min_mean_score``. A
    exigencia sobre o minimo evita caixas que atravessam a fronteira do musculo
    peitoral ou saem pela linha da pele.

    Devolve ``(caixa, diagnostico)`` ou ``None`` se nao houver regiao valida.
    """
    cfg = cfg or AnatomyConfig()
    rng = rng or np.random.default_rng()

    weights = pmap.score.astype(np.float64)
    total = weights.sum()
    if total <= 0:
        return None
    flat = weights.ravel() / total
    width = pmap.score.shape[1]

    # amostrar indices de uma vez e so entao testar: evita 800 chamadas ao RNG
    picks = rng.choice(flat.size, size=min(max_tries, flat.size), replace=True, p=flat)

    # Exclusoes duras explicitas. Com min_box_score > 0 elas ja estavam implicitas
    # (score zero no peitoral + margem e a menos de min_skin_distance_mm da pele);
    # com min_box_score = 0 -- caixa que pode conter gordura, para a lesao nao
    # ficar presa no tecido mais denso -- sao elas que protegem.
    proibido = pmap.skin_distance_mm < cfg.min_skin_distance_mm
    if pmap.pectoral.any():
        margem = mm_to_px(cfg.pectoral_margin_mm, mm_per_px)
        proibido = proibido | (cv.dilate(pmap.pectoral, cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (margem * 2 + 1,) * 2)) > 0)

    for idx in picks:
        cy, cx = int(idx // width), int(idx % width)
        diam_mm = float(rng.uniform(*diameter_mm_range))
        aspect = float(rng.uniform(*aspect_range))
        d = mm_to_px(diam_mm, mm_per_px)
        w = max(8, int(round(d * np.sqrt(aspect))))
        h = max(8, int(round(d / np.sqrt(aspect))))
        box = LesionBox(cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
        if box.x1 < 0 or box.y1 < 0 or box.x2 > canvas or box.y2 > canvas:
            continue
        sub = pmap.score[box.slices()]
        if sub.size == 0:
            continue
        if float(sub.min()) < cfg.min_box_score:
            continue
        if float(sub.mean()) < cfg.min_mean_score:
            continue
        if coverage_inside_breast(box, pmap.breast) < 0.999:
            continue
        if proibido[box.slices()].any():
            continue
        return box, box_diagnostics(box, pmap, mm_per_px)
    return None


def box_diagnostics(box: LesionBox, pmap: PlausibilityMap, mm_per_px: float) -> Dict[str, float]:
    """Medidas anatomicas da caixa escolhida, para o log e os metadados."""
    sub = pmap.score[box.slices()]
    dist = pmap.skin_distance_mm[box.slices()]
    rank = pmap.density_rank[box.slices()]
    cx, cy = box.center
    out = {
        "anat_score_min": float(sub.min()) if sub.size else 0.0,
        "anat_score_mean": float(sub.mean()) if sub.size else 0.0,
        "skin_distance_mm_min": float(dist.min()) if dist.size else 0.0,
        "skin_distance_mm_center": float(pmap.skin_distance_mm[cy, cx]),
        "density_rank_mean": float(rank.mean()) if rank.size else 0.0,
        "pectoral_overlap_frac": float((pmap.pectoral[box.slices()] > 0).mean()) if sub.size else 0.0,
    }
    if pmap.nipple is not None:
        nx, ny = pmap.nipple
        out["nipple_distance_mm"] = float(np.hypot(cx - nx, cy - ny) * mm_per_px)
    return out


# ---------------------------------------------------------------------------
# Descricao textual da localizacao
# ---------------------------------------------------------------------------


def describe_region(box: LesionBox, pmap: PlausibilityMap, view: str = "unknown") -> str:
    """Descricao geometrica da localizacao, no referencial da imagem.

    Deliberadamente **nao** usa a nomenclatura de quadrantes do BI-RADS (QSL,
    QII, ...): sem a lateralidade da paciente e a orientacao de aquisicao
    registradas no arquivo, mapear linhas da imagem para superior/inferior
    anatomico seria adivinhacao. O que esta abaixo e verificavel na propria
    imagem: profundidade em relacao a parede toracica e posicao vertical.
    """
    ys, xs = np.nonzero(pmap.breast > 0)
    if len(xs) == 0:
        return "indeterminada"
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    cx, cy = box.center

    span_x = max(1, x1 - x0)
    depth = (cx - x0) / span_x if pmap.side == "left" else (x1 - cx) / span_x
    thirds = ("terco posterior", "terco medio", "terco anterior")
    depth_label = thirds[min(2, int(depth * 3))]

    span_y = max(1, y1 - y0)
    vertical = (cy - y0) / span_y
    rows = ("faixa superior", "faixa media", "faixa inferior")
    row_label = rows[min(2, int(vertical * 3))]

    view_label = {"CC": "craniocaudal", "MLO": "obliqua mediolateral",
                  "LATERAL": "lateral", "UNKNOWN": "incidencia nao identificada",
                  }.get(view.upper(), view.lower())
    return "{}, {} ({}, parede toracica a {})".format(
        depth_label, row_label, view_label,
        "esquerda" if pmap.side == "left" else "direita")


__all__ = [
    "AnatomyConfig", "PlausibilityMap", "plausibility_map", "sample_anatomical_box",
    "box_diagnostics", "describe_region", "chest_wall_side", "nipple_point",
    "skin_distance_map", "detect_pectoral", "density_rank_map", "glandular_fraction_map",
]
