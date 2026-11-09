#!/usr/bin/env python
"""Auditoria independente de um lote de lesoes sinteticas.

Le do disco o que foi *entregue* (``original/`` + ``masks/original/``) e a
mamografia de origem, e recalcula todas as metricas do zero. Nao usa nenhum
numero gravado pelo pipeline: os campos de ``metadata/*.json`` servem apenas
para localizar a lesao (pegada, janela) e para comparar depois.

Quatro grupos de criterios, cada um com severidade:

* **integridade** -- o arquivo entregue e o original fora da lesao, a mascara
  descreve a alteracao, a caixa confere;
* **artefato** -- degrau na borda da mascara, costura, halo escuro, saturacao,
  periodicidade de grade do VAE (calibrada contra recortes de tecido real da
  *mesma* mama), ruido e nitidez em relacao ao anel vizinho;
* **anatomia** -- lesao dentro do tecido, distancia da pele, sobreposicao com o
  peitoral, truncamento na borda da imagem, score do mapa de plausibilidade;
* **clinica** -- contraste de Weber, CNR, saldo de densidade na ROI, diametro.

Depois dos criterios por imagem, uma passagem de **duplicacao** compara as
variantes da mesma mama (IoU das pegadas e correlacao dos recortes).

    .venv/bin/python scripts/auditar_dataset.py ../mamas_com_lesao/lote_variacoes_v1

Saidas em ``<lote>/auditoria/``: ``auditoria.csv``, ``auditoria.json`` e
``RELATORIO_QUALIDADE.md``. Codigo de saida 0 se toda lesao entregue foi
avaliada; 1 se alguma reprovou; 2 se o lote nao pode ser lido.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2 as cv
import numpy as np

from mamelesion.anatomy import AnatomyConfig, plausibility_map
from mamelesion.native import build_breast_canvas, read_source
from mamelesion.pipeline import ViewInfo, load_view_table, resolve_view
from mamelesion.preprocessing import breast_mask

EPS = 1e-8

#: criterio -> (grupo, severidade, lo, hi, descricao). Faixas iguais as do
#: projeto (quality.ARTIFACT_RULES, PLAUSIBILITY_RULES, AnatomyConfig,
#: native_window.density_floor) onde ja existiam.
CRITERIOS: Dict[str, Tuple[str, str, float, float, str]] = {
    "fora_da_mascara_px":      ("integridade", "bloqueante", 0, 0, "pixels alterados fora da mascara"),
    "dimensoes_ok":            ("integridade", "bloqueante", 1, 1, "largura/altura/dtype iguais aos do original"),
    "mascara_area_px":         ("integridade", "bloqueante", 1, math.inf, "mascara nao vazia"),
    "mascara_componentes":     ("integridade", "bloqueante", 1, 1, "mascara e uma unica regiao conexa"),
    "caixa_confere":           ("integridade", "bloqueante", 1, 1, "caixa do metadado == caixa da mascara"),

    "degrau_borda_u8":         ("artefato", "bloqueante", -12.0, 12.0, "degrau de nivel atravessando a borda da mascara"),
    "costura_excesso":         ("artefato", "bloqueante", -1.0, 0.60, "gradiente extra na costura vs. tecido vizinho"),
    "halo_u8":                 ("artefato", "bloqueante", -3.5, 12.0, "saldo de nivel na margem redesenhada (piso -3)"),
    "saturacao_frac":          ("artefato", "bloqueante", 0.0, 0.02, "fracao da lesao saturada no topo da escala"),
    "grade_z":                 ("artefato", "bloqueante", -math.inf, 6.0, "periodicidade de grade vs. tecido real da mesma mama"),
    "textura_roi":             ("artefato", "bloqueante", 0.38, 1.60, "media geometrica de ruido e nitidez vs. anel (veu/bloco liso)"),
    # Mancha branca chapada: o nucleo da lesao fica mais claro que QUALQUER
    # tecido original em volta. Massa densa real compete em brilho com o
    # parenquima vizinho; a mancha passa por cima dele e aparece estourada no
    # janelamento do proprio recorte. Limiar 0,25 posto olhando as aprovadas
    # do piloto do birads_1 ordenadas por esta medida: >= 0,28 eram manchas
    # chapadas/quadradas, <= 0,24 massas com textura (ver RELATORIO_QUALIDADE.md).
    # Rebaixado a AVISO no mesmo dia: aplicada aos 92 recortes reais de
    # birads_4/5, a medida passaria de 0,25 em 47% deles -- lesao densa real E
    # mais clara que o tecido em volta. A "mancha chapada" que motivou o criterio
    # era em boa parte janelamento: exibida com os percentis do tecido original,
    # qualquer lesao mais clara que ele estoura em branco. Autojanelada como um
    # recorte real, a lesao mostra textura. O contraste agora e limitado por
    # crel_afastado (fidelidade), na mesma regua da calibracao.
    "nucleo_acima_tecido":     ("artefato", "aviso", 0.0, 0.25, "fracao do nucleo da lesao mais clara que o p99,5 do tecido original em volta (mancha branca chapada)"),
    "ruido_roi_anel":          ("artefato", "aviso", 0.55, 1.60, "ruido da lesao / ruido do anel vizinho"),
    "nitidez_roi_anel":        ("artefato", "aviso", 0.25, 2.50, "nitidez da lesao / nitidez do anel"),
    "preenchimento_pegada":    ("artefato", "aviso", 0.03, 0.92, "fracao da pegada ocupada pela lesao (1,0 = a caixa inteira virou lesao)"),
    # Lesao BI-RADS 4/5 e irregular por definicao (margem espiculada): a
    # circularidade fica entre 0,03 e 0,45 neste lote. O aviso pega so o
    # segmento estilhacado, nao a lesao espiculada.
    "circularidade":           ("artefato", "aviso", 0.05, 1.05, "circularidade da lesao segmentada"),
    # Formato medido no ACRESCIMO de densidade (saida - original), nao na imagem:
    # na imagem o tecido claro do entorno contamina (contato_borda, acima, nao
    # separou nada). Elipse preenche no maximo 0,785 do retangulo envolvente, entao
    # extensao alta sozinha tambem pega oval lisa (plausivel); o que separa o
    # bloco retangular -- heranca das mascaras retangulares do treino do MAM-E --
    # sao os cantos cheios. Limiares postos olhando as aprovadas v3 ordenadas:
    # extensao > 0,70 e cantos >= 0,33 eram blocos quadrados/retangulares (2 falsos
    # positivos em ~8); ovais lisas ficam com cantos 0,11-0,15.
    "extensao_acrescimo":      ("artefato", "aviso", 0.0, 1.0, "area da lesao / retangulo que a envolve (no acrescimo de densidade)"),
    "cantos_acrescimo":        ("artefato", "aviso", 0.0, 1.0, "preenchimento dos 4 cantos (20%) do retangulo envolvente"),
    "bloco_retangular":        ("artefato", "bloqueante", 0, 0, "lesao compacta (extensao > 0,70) com cantos cheios (>= 0,33): bloco retangular"),

    "cobertura_tecido":        ("anatomia", "bloqueante", 0.98, 1.0, "fracao da lesao sobre tecido mamario"),
    "distancia_pele_mm":       ("anatomia", "bloqueante", 8.0, math.inf, "distancia minima da lesao ate a pele"),
    "peitoral_frac":           ("anatomia", "bloqueante", 0.0, 0.0, "fracao da lesao sobre o musculo peitoral"),
    "toca_borda_img":          ("anatomia", "bloqueante", 0, 0, "lesao truncada pela borda da imagem"),
    "score_anatomico":         ("anatomia", "aviso", 0.35, 1.0, "score medio do mapa de plausibilidade na lesao"),

    "weber_nativo":            ("clinica", "bloqueante", 0.05, 1.50, "contraste de Weber da lesao vs. anel"),
    "cnr_nativo":              ("clinica", "bloqueante", 0.30, 12.0, "razao contraste-ruido"),
    "roi_delta_ratio":         ("clinica", "bloqueante", 0.01, 1.00, "saldo medio de densidade em toda a pegada"),
    "diametro_mm":             ("clinica", "bloqueante", 5.0, 60.0, "diametro equivalente da lesao"),
    "tamanho_no_catalogo":     ("clinica", "aviso", 1, 1, "tamanho dentro da faixa dos recortes reais"),

    "iou_max_variantes":       ("duplicacao", "bloqueante", 0.0, 0.50, "sobreposicao com outra variante da mesma mama"),
    "similaridade_max":        ("duplicacao", "bloqueante", -1.0, 0.95, "correlacao do recorte com outra variante"),
}


# ---------------------------------------------------------------------------
# Medidas
# ---------------------------------------------------------------------------


def _hp(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Componente de alta frequencia (textura/ruido)."""
    return img - cv.GaussianBlur(img, (0, 0), sigma)


def _periodic_score(v: np.ndarray, period: float) -> float:
    """Energia espectral no periodo dado, relativa a mediana do espectro."""
    n = len(v)
    k = n / max(period, 1e-6)
    if n < 32 or k < 2 or k > n / 2 - 2:
        return float("nan")
    f = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(n)))
    banda = float(f[int(k) - 1:int(k) + 2].max())
    fundo = float(np.median(f[1:]) + EPS)
    return banda / fundo


def grade_score(patch: np.ndarray, period: float) -> float:
    """Periodicidade de grade (blocos do VAE) no recorte, em linhas e colunas."""
    if patch.size < 1024:
        return float("nan")
    p = np.abs(_hp(patch.astype(np.float32)))
    linhas = _periodic_score(p.mean(axis=1), period)
    colunas = _periodic_score(p.mean(axis=0), period)
    vals = [x for x in (linhas, colunas) if np.isfinite(x)]
    return float(max(vals)) if vals else float("nan")


def control_patches(gray: np.ndarray, tecido: np.ndarray, shape: Tuple[int, int],
                    proibido: Tuple[int, int, int, int], rng: np.random.Generator,
                    n: int = 24) -> List[np.ndarray]:
    """Recortes de tecido real da mesma mama, do tamanho da lesao."""
    h, w = shape
    H, W = gray.shape
    if h >= H or w >= W:
        return []
    px0, py0, px1, py1 = proibido
    integral = cv.integral(tecido.astype(np.uint8) // 255)
    saida: List[np.ndarray] = []
    for _ in range(n * 40):
        if len(saida) >= n:
            break
        x = int(rng.integers(0, W - w))
        y = int(rng.integers(0, H - h))
        if not (x + w < px0 or x > px1 or y + h < py0 or y > py1):
            continue                       # sobrepoe a lesao
        area = (integral[y + h, x + w] - integral[y, x + w]
                - integral[y + h, x] + integral[y, x])
        if area < 0.995 * h * w:           # exige tecido cheio, como a lesao
            continue
        saida.append(gray[y:y + h, x:x + w])
    return saida


def robust_z(valor: float, amostras: List[float]) -> float:
    finitos = [v for v in amostras if np.isfinite(v)]
    if not np.isfinite(valor) or len(finitos) < 5:
        return float("nan")
    med = float(np.median(finitos))
    mad = float(np.median(np.abs(np.asarray(finitos) - med)))
    return (valor - med) / (1.4826 * mad + 1e-6)


def ring_of(mask: np.ndarray, tecido: np.ndarray, largura: int,
            fora: np.ndarray) -> np.ndarray:
    """Anel de tecido real em volta da mascara, sem tocar a regiao alterada."""
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * largura + 1, 2 * largura + 1))
    dil = cv.dilate(mask, k)
    anel = (dil > 0) & (mask == 0) & (tecido > 0) & fora
    return anel


# ---------------------------------------------------------------------------
# Hospedeiro (uma mamografia de entrada, compartilhada pelas suas variantes)
# ---------------------------------------------------------------------------


class Hospedeiro:
    """Tudo que se recalcula uma vez por mamografia de origem."""

    def __init__(self, path: Path, limiar: int, canvas: int, detector_mm: float,
                 incidencias: Optional[Dict[str, ViewInfo]] = None) -> None:
        self.path = path
        self.src = read_source(path)
        self.raw = self.src.raw
        self.ref = float(self.src.scale_to_u8 * 255.0)
        self.k_u8 = 255.0 / max(self.ref, EPS)
        self.H, self.W = self.raw.shape[:2]
        self.mm_px = detector_mm / float(max(self.H, self.W))
        self.tecido = breast_mask(self.src.gray_u8, limiar)
        dist = cv.distanceTransform((self.tecido > 0).astype(np.uint8), cv.DIST_L2, 5)
        self.dist_mm = dist * self.mm_px
        # mapa de plausibilidade recalculado com a mesma rotina do pipeline
        cvs, framing, mama = build_breast_canvas(self.src.gray_u8, canvas, breast_threshold=limiar)
        self.framing = framing
        view = resolve_view(path.stem, incidencias)
        self.view = view
        mm_canvas = self.mm_px / framing.scale
        self.pmap = plausibility_map(cvs, mama, mm_canvas, view.view, AnatomyConfig())
        # crc32, nao hash(): hash() de str muda a cada processo (PYTHONHASHSEED) e
        # tornava o sorteio dos controles do grade_z irreprodutivel entre execucoes
        self.rng = np.random.default_rng(zlib.crc32(path.stem.encode("utf-8")))

    def canvas_box(self, fp: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x0, y0 = self.framing.orig_to_canvas(fp[0], fp[1])
        x1, y1 = self.framing.orig_to_canvas(fp[2], fp[3])
        lado = self.pmap.score.shape[0]
        return (max(0, int(x0)), max(0, int(y0)),
                min(lado, int(math.ceil(x1))), min(lado, int(math.ceil(y1))))


# ---------------------------------------------------------------------------
# Auditoria de uma lesao
# ---------------------------------------------------------------------------


def nucleo_acima_tecido(src: np.ndarray, out: np.ndarray, mascara: np.ndarray,
                        pegada: Tuple[int, int, int, int], fator: float = 1.8, q: float = 99.5) -> float:
    """Fracao do nucleo da lesao mais clara que o percentil ``q`` do tecido original em volta.

    Janela quadrada de ``fator`` x o lado maior da pegada (a mesma dos paineis de
    conferencia); referencia = tecido original da janela fora da mascara; nucleo =
    pixels da pegada, dentro da mascara, onde a lesao somou ao menos 25% do seu
    acrescimo p95. 0 = a lesao nao passa do tecido mais claro em volta; 1 = o
    nucleo inteiro fica acima dele (mancha branca chapada).
    """
    x1, y1, x2, y2 = (int(v) for v in pegada)
    lado = int(max(x2 - x1, y2 - y1) * fator)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    H, W = src.shape[:2]
    x0, y0 = max(0, cx - lado // 2), max(0, cy - lado // 2)
    xe, ye = min(W, x0 + lado), min(H, y0 + lado)
    a = src[y0:ye, x0:xe].astype(np.float32)
    b = out[y0:ye, x0:xe].astype(np.float32)
    m = mascara[y0:ye, x0:xe] > 0
    p5 = float(np.percentile(a, 5))
    vizinho = (a > p5 + 0.02 * (float(a.max()) - p5)) & ~m
    ref = float(np.percentile(a[vizinho], q)) if vizinho.sum() > 100 else float(np.percentile(a, q))
    nucleo = np.zeros(m.shape, bool)
    nucleo[max(0, y1 - y0):y2 - y0, max(0, x1 - x0):x2 - x0] = True
    nucleo &= m
    if not nucleo.any():
        return 0.0
    d = b - a
    les = nucleo & (d > 0.25 * float(np.percentile(d[nucleo], 95)))
    if les.sum() < 50:                      # quase nada somado: nao e mancha
        return 0.0
    return float((b[les] > ref).mean())


def forma_do_acrescimo(src: np.ndarray, out: np.ndarray,
                       pegada: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """(extensao, cantos) da lesao isolada pelo que o modelo acrescentou.

    Janela = pegada + 25% do lado; acrescimo suavizado (sigma 2% do lado),
    limiar em 40% do seu p99, maior componente. ``extensao`` = area / retangulo
    envolvente; ``cantos`` = preenchimento medio dos 4 cantos de 20% x 20% desse
    retangulo.
    """
    x1, y1, x2, y2 = (int(v) for v in pegada)
    m = max(x2 - x1, y2 - y1) // 4
    X0, Y0 = max(0, x1 - m), max(0, y1 - m)
    d = out[Y0:y2 + m, X0:x2 + m].astype(np.float32) - src[Y0:y2 + m, X0:x2 + m].astype(np.float32)
    d = cv.GaussianBlur(d, (0, 0), max(2.0, 0.02 * (x2 - x1)))
    s = (d > 0.4 * float(np.percentile(d, 99))).astype(np.uint8)
    n, lab, st, _ = cv.connectedComponentsWithStats(s, 8)
    if n < 2:
        return float("nan"), float("nan")
    s = lab == 1 + int(np.argmax(st[1:, cv.CC_STAT_AREA]))
    ys, xs = np.nonzero(s)
    b = s[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = b.shape
    ch, cw = max(1, h // 5), max(1, w // 5)
    cantos = float(np.mean([b[:ch, :cw].mean(), b[:ch, -cw:].mean(), b[-ch:, :cw].mean(), b[-ch:, -cw:].mean()]))
    return float(b.mean()), cantos


def contato_borda(seg: np.ndarray, caixa: Tuple[int, int, int, int], faixa: float = 0.06) -> float:
    """Fracao media dos 4 lados da caixa ao longo dos quais a lesao encosta na borda.

    Lesao oval ou irregular toca a caixa em poucos pontos (~0,3-0,5 para uma
    elipse inscrita); um bloco que preenche o retangulo da mascara encosta nos
    quatro lados quase inteiros (~1). ``caixa`` = (y0, x0, y1, x1) no recorte;
    ``faixa`` = espessura da faixa junto a cada lado, em fracao do lado.
    """
    y0, x0, y1, x1 = caixa
    sub = seg[max(0, y0):y1, max(0, x0):x1]
    h, w = sub.shape
    if h < 8 or w < 8:
        return 0.0
    by, bx = max(2, int(round(faixa * h))), max(2, int(round(faixa * w)))
    lados = [sub[:by, :].any(axis=0), sub[-by:, :].any(axis=0),
             sub[:, :bx].any(axis=1), sub[:, -bx:].any(axis=1)]
    return float(np.mean([l.mean() for l in lados]))


def auditar_lesao(host: Hospedeiro, lesao: Dict[str, Any], catalogo: Dict[str, List[List[int]]],
                  ring_mm: float = 12.0) -> Dict[str, Any]:
    from mamelesion import quality as qmod

    med: Dict[str, Any] = {}
    out_path = Path(lesao["files"]["original"])
    mask_path = Path(lesao["files"]["mask_original"])
    out = cv.imread(str(out_path), cv.IMREAD_UNCHANGED)
    mask = cv.imread(str(mask_path), cv.IMREAD_UNCHANGED)
    if out is None or mask is None:
        raise FileNotFoundError("arquivo entregue ausente: {}".format(out_path))
    src = host.raw

    med["dimensoes_ok"] = int(out.shape == src.shape and out.dtype == src.dtype
                              and mask.shape == src.shape[:2])
    if not med["dimensoes_ok"]:
        return {"medidas": med, "erro": "dimensoes/dtype divergentes"}

    bin_mask = (mask > 127).astype(np.uint8)
    dif = out.astype(np.float32) - src.astype(np.float32)
    alterado = np.abs(dif) > 0
    med["fora_da_mascara_px"] = int(np.count_nonzero(alterado & (bin_mask == 0)))
    med["mascara_area_px"] = int(bin_mask.sum())
    n_comp, _ = cv.connectedComponents(bin_mask)
    med["mascara_componentes"] = int(n_comp - 1)
    if med["mascara_area_px"] == 0:
        return {"medidas": med, "erro": "mascara vazia"}

    ys, xs = np.nonzero(bin_mask)
    mbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    g = lesao["box_original"]
    med["caixa_confere"] = int(mbox == (g["x1"], g["y1"], g["x2"], g["y2"]))

    fx0, fy0, fx1, fy1 = [int(v) for v in g["pegada_lesao"]]
    fw, fh = fx1 - fx0, fy1 - fy0
    med["nucleo_acima_tecido"] = round(nucleo_acima_tecido(src, out, bin_mask, (fx0, fy0, fx1, fy1)), 4)
    ext, cantos = forma_do_acrescimo(src, out, (fx0, fy0, fx1, fy1))
    med["extensao_acrescimo"], med["cantos_acrescimo"] = round(ext, 4), round(cantos, 4)
    med["bloco_retangular"] = int(np.isfinite(ext) and ext > 0.70 and cantos >= 0.33)
    med["largura_px"], med["altura_px"] = fw, fh
    med["diametro_mm"] = round(float(math.sqrt(fw * fh) * host.mm_px), 2)
    med["toca_borda_img"] = int(fx0 <= 1 or fy0 <= 1 or fx1 >= host.W - 1 or fy1 >= host.H - 1)

    # --- recorte de trabalho: pegada + anel ---------------------------------
    anel_px = max(8, int(round(ring_mm / host.mm_px)))
    wx0, wy0 = max(0, fx0 - 2 * anel_px), max(0, fy0 - 2 * anel_px)
    wx1, wy1 = min(host.W, fx1 + 2 * anel_px), min(host.H, fy1 + 2 * anel_px)
    a = src[wy0:wy1, wx0:wx1].astype(np.float32) * host.k_u8       # original, escala u8
    b = out[wy0:wy1, wx0:wx1].astype(np.float32) * host.k_u8       # entregue
    m = bin_mask[wy0:wy1, wx0:wx1] * 255
    tec = host.tecido[wy0:wy1, wx0:wx1]
    intacto = ~alterado[wy0:wy1, wx0:wx1]
    fp = np.zeros(a.shape, bool)
    fp[fy0 - wy0:fy1 - wy0, fx0 - wx0:fx1 - wx0] = True

    anel = ring_of(m, tec, anel_px, intacto)
    if anel.sum() < 200:                                   # mama pequena: afrouxa o anel
        anel = ring_of(m, tec, 2 * anel_px, intacto)
    if anel.sum() < 50:
        return {"medidas": med, "erro": "sem anel de tecido de referencia"}

    seg = qmod.segment_lesion(a, b, m).astype(bool)
    segmentada = bool(seg.any())
    if not segmentada:                       # nada ficou mais denso: sem lesao a medir
        seg = fp
    anel_mean, anel_std = float(a[anel].mean()), float(a[anel].std())
    les_mean = float(b[seg].mean())
    med["nivel_anel_u8"] = round(anel_mean, 2)
    med["weber_nativo"] = round((les_mean - anel_mean) / (anel_mean + EPS), 4)
    med["cnr_nativo"] = round((les_mean - anel_mean) / (anel_std + EPS), 4)
    med["roi_delta_ratio"] = round(float((b[fp] - a[fp]).mean()) / (float(a[fp].mean()) + EPS), 4)

    hp_b, hp_a = _hp(b), _hp(a)
    med["ruido_roi_anel"] = round(float(hp_b[fp].std()) / (float(hp_a[anel].std()) + EPS), 4)
    gb = cv.magnitude(cv.Sobel(b, cv.CV_32F, 1, 0, 3), cv.Sobel(b, cv.CV_32F, 0, 1, 3))
    ga = cv.magnitude(cv.Sobel(a, cv.CV_32F, 1, 0, 3), cv.Sobel(a, cv.CV_32F, 0, 1, 3))
    med["nitidez_roi_anel"] = round(float(gb[fp].mean()) / (float(ga[anel].mean()) + EPS), 4)
    # Media geometrica das duas: o artefato de "veu" (ou bloco liso) derruba as
    # duas juntas, enquanto uma massa densa real baixa so o ruido -- o limiar de
    # 0,38 foi posto olhando os dois extremos (ver RELATORIO_QUALIDADE.md).
    med["textura_roi"] = round(float(np.sqrt(max(med["ruido_roi_anel"], 0.0)
                                             * max(med["nitidez_roi_anel"], 0.0))), 4)

    if segmentada:
        med["preenchimento_pegada"] = round(float(seg[fp].mean()), 4)
        cnts, _ = cv.findContours(seg.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv.contourArea)
            per = float(cv.arcLength(c, True))
            med["circularidade"] = round(4.0 * math.pi * float(cv.contourArea(c)) / (per * per + EPS), 4)
        med["contato_borda"] = round(contato_borda(seg, (fy0 - wy0, fx0 - wx0, fy1 - wy0, fx1 - wx0)), 4)

    # --- costura e degrau na borda da mascara -------------------------------
    k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    dentro = ((m > 0) & (cv.erode(m, k3) == 0)) & tec.astype(bool)
    fora_b = ((cv.dilate(m, k3) > 0) & (m == 0)) & tec.astype(bool) & intacto
    if dentro.sum() > 20 and fora_b.sum() > 20:
        med["degrau_borda_u8"] = round(float(b[dentro].mean() - b[fora_b].mean()), 3)
        borda = dentro | fora_b
        med["costura_excesso"] = round(
            float(gb[borda].mean() - ga[borda].mean()) / (float(ga[anel].mean()) + EPS), 4)
    else:
        med["degrau_borda_u8"] = 0.0
        med["costura_excesso"] = 0.0

    # --- halo: margem redesenhada em volta da lesao --------------------------
    margem = (m > 0) & (~fp)
    med["halo_u8"] = round(float((b[margem] - a[margem]).mean()), 3) if margem.sum() > 50 else 0.0

    # --- saturacao introduzida ----------------------------------------------
    topo = 0.999 * host.ref
    sat_out = float((out[fy0:fy1, fx0:fx1] >= topo).mean())
    sat_src = float((src[fy0:fy1, fx0:fx1] >= topo).mean())
    med["saturacao_frac"] = round(max(0.0, sat_out - sat_src), 5)

    # --- grade do VAE, calibrada em tecido real da mesma mama ---------------
    lado = float(lesao["generation"]["janela"]["side"])
    periodo = 8.0 * lado / float(lesao["generation"]["janela"]["model_res"])
    score = grade_score(b[fy0 - wy0:fy1 - wy0, fx0 - wx0:fx1 - wx0], periodo)
    controles = control_patches(src.astype(np.float32) * host.k_u8, host.tecido, (fh, fw),
                                (fx0, fy0, fx1, fy1), host.rng, n=24)
    med["grade_score"] = round(float(score), 3) if np.isfinite(score) else None
    med["grade_z"] = round(robust_z(score, [grade_score(c, periodo) for c in controles]), 3)
    med["n_controles"] = len(controles)

    # --- anatomia ------------------------------------------------------------
    med["cobertura_tecido"] = round(float((host.tecido[fy0:fy1, fx0:fx1] > 0).mean()), 4)
    med["distancia_pele_mm"] = round(float(host.dist_mm[fy0:fy1, fx0:fx1].min()), 2)
    cx0, cy0, cx1, cy1 = host.canvas_box((fx0, fy0, fx1, fy1))
    if cx1 > cx0 and cy1 > cy0:
        med["peitoral_frac"] = round(float((host.pmap.pectoral[cy0:cy1, cx0:cx1] > 0).mean()), 4)
        med["score_anatomico"] = round(float(host.pmap.score[cy0:cy1, cx0:cx1].mean()), 4)
    else:
        med["peitoral_frac"], med["score_anatomico"] = 1.0, 0.0

    # --- tamanho no catalogo de recortes reais -------------------------------
    tam = catalogo.get(lesao["lesion_type"])
    if tam:
        arr = np.asarray(tam, float)
        med["tamanho_no_catalogo"] = int(
            arr[:, 0].min() <= fw <= arr[:, 0].max() and arr[:, 1].min() <= fh <= arr[:, 1].max())
    else:
        med["tamanho_no_catalogo"] = 1
    return {"medidas": med}


# ---------------------------------------------------------------------------
# Veredito
# ---------------------------------------------------------------------------


def julgar(med: Dict[str, Any]) -> Dict[str, Any]:
    """Aplica CRITERIOS as medidas e devolve falhas, avisos e nao avaliados."""
    checks, falhas, avisos, nao_avaliados = [], [], [], []
    for nome, (grupo, sev, lo, hi, desc) in CRITERIOS.items():
        if nome not in med:
            continue
        v = med[nome]
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            nao_avaliados.append(nome)
            checks.append({"criterio": nome, "grupo": grupo, "severidade": sev,
                           "valor": None, "faixa": [lo, hi], "ok": None, "descricao": desc})
            continue
        ok = bool(lo <= float(v) <= hi)
        checks.append({"criterio": nome, "grupo": grupo, "severidade": sev,
                       "valor": float(v), "faixa": [lo, hi], "ok": ok, "descricao": desc})
        if not ok:
            (falhas if sev == "bloqueante" else avisos).append(nome)
    return {"checks": checks, "falhas": falhas, "avisos": avisos,
            "nao_avaliados": nao_avaliados,
            "veredito": "aprovada" if not falhas else "reprovada"}


def _crop_norm(path: Path, fp: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if img is None:
        return None
    x0, y0, x1, y1 = fp
    c = img[y0:y1, x0:x1].astype(np.float32)
    if c.size == 0:
        return None
    c = cv.resize(c, (64, 64), interpolation=cv.INTER_AREA)
    c -= c.mean()
    n = float(np.linalg.norm(c))
    return c / n if n > EPS else None


def deduplicar(registros: List[Dict[str, Any]]) -> None:
    """Marca as variantes redundantes de uma mesma mama (supressao nao-maxima).

    Duas variantes so competem quando ocupam quase o mesmo lugar. Percorre as
    variantes da maior para a menor confianca e mantem cada uma que nao conflite
    com nenhuma **ja mantida** -- comparar todos os pares descartaria A por causa
    de B e C por causa de B, mesmo com A e C em posicoes diferentes.

    Recebe so as lesoes que passaram nos demais criterios: uma variante com
    artefato nao deve ocupar o lugar de uma boa.
    """
    lim_iou = CRITERIOS["iou_max_variantes"][3]
    lim_sim = CRITERIOS["similaridade_max"][3]
    por_host: Dict[str, List[Dict[str, Any]]] = {}
    for r in registros:
        por_host.setdefault(r["origem"], []).append(r)

    for _, grupo in por_host.items():
        crops: Dict[str, Optional[np.ndarray]] = {}
        mantidos: List[Dict[str, Any]] = []
        for r in sorted(grupo, key=lambda t: -t.get("confianca", 0.0)):
            pior_iou, pior_sim = 0.0, 0.0
            conflito = False
            for k in mantidos:
                a, b = r["pegada"], k["pegada"]
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = ix * iy
                uniao = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
                iou = inter / uniao if uniao else 0.0
                sim = 0.0
                if iou > 0.30:
                    for t in (r, k):
                        if t["saida"] not in crops:
                            crops[t["saida"]] = _crop_norm(Path(t["arquivo_saida"]), t["pegada"])
                    ca, cb = crops[r["saida"]], crops[k["saida"]]
                    if ca is not None and cb is not None:
                        sim = float((ca * cb).sum())
                pior_iou, pior_sim = max(pior_iou, iou), max(pior_sim, sim)
                if iou > lim_iou or sim > lim_sim:
                    conflito = True
            r["medidas"]["iou_max_variantes"] = round(pior_iou, 4)
            r["medidas"]["similaridade_max"] = round(pior_sim, 4)
            if not conflito:
                mantidos.append(r)


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------


def relatorio(regs: List[Dict[str, Any]], lote: Path, cfg: Dict[str, Any]) -> str:
    ap = [r for r in regs if r["veredito"] == "aprovada"]
    rep = [r for r in regs if r["veredito"] == "reprovada"]
    L: List[str] = []
    L.append("# Relatorio de qualidade -- {}".format(lote.name))
    L.append("")
    L.append("Auditoria independente: as metricas abaixo foram recalculadas a partir dos")
    L.append("arquivos entregues (`original/`, `masks/original/`) e das mamografias de")
    L.append("origem, sem usar nenhum numero gravado pelo pipeline.")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append("| Lesoes auditadas | {} |".format(len(regs)))
    L.append("| Aprovadas | {} ({:.1%}) |".format(len(ap), len(ap) / max(1, len(regs))))
    L.append("| Reprovadas | {} ({:.1%}) |".format(len(rep), len(rep) / max(1, len(regs))))
    L.append("| Mamas de origem | {} |".format(len({r["origem"] for r in regs})))
    L.append("| Criterios por lesao | {} |".format(len(CRITERIOS)))
    L.append("")

    L.append("## Falhas por criterio")
    L.append("")
    L.append("| Criterio | Grupo | Severidade | Faixa aceita | Reprovacoes | Avisos |")
    L.append("|---|---|---|---|---:|---:|")
    for nome, (grupo, sev, lo, hi, desc) in CRITERIOS.items():
        nf = sum(1 for r in regs if nome in r["falhas"])
        na = sum(1 for r in regs if nome in r["avisos"])
        faixa = "{} a {}".format("-inf" if lo == -math.inf else round(lo, 3),
                                 "inf" if hi == math.inf else round(hi, 3))
        L.append("| `{}` | {} | {} | {} | {} | {} |".format(nome, grupo, sev, faixa, nf, na))
    L.append("")
    L.append("Descricao de cada criterio em `scripts/auditar_dataset.py` (dicionario `CRITERIOS`).")
    L.append("")

    nums = ["weber_nativo", "cnr_nativo", "roi_delta_ratio", "diametro_mm", "distancia_pele_mm",
            "textura_roi", "nucleo_acima_tecido", "ruido_roi_anel", "nitidez_roi_anel", "preenchimento_pegada", "circularidade", "degrau_borda_u8", "halo_u8", "grade_z"]
    L.append("## Distribuicao das medidas (lesoes aprovadas)")
    L.append("")
    L.append("| Medida | min | p25 | mediana | p75 | max |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for nome in nums:
        vals = [r["medidas"][nome] for r in ap
                if isinstance(r["medidas"].get(nome), (int, float))
                and np.isfinite(r["medidas"][nome])]
        if not vals:
            continue
        q = np.percentile(vals, [0, 25, 50, 75, 100])
        L.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(nome, *q))
    L.append("")

    if rep:
        L.append("## Lesoes reprovadas")
        L.append("")
        L.append("| Arquivo | Tipo | Criterios reprovados | Valores |")
        L.append("|---|---|---|---|")
        for r in sorted(rep, key=lambda x: x["saida"]):
            vals = "; ".join("{}={}".format(f, r["medidas"].get(f)) for f in r["falhas"])
            L.append("| `{}` | {} | {} | {} |".format(
                r["saida"], r["tipo"], ", ".join(r["falhas"]), vals))
        L.append("")

    avisos = [r for r in regs if r["avisos"]]
    if avisos:
        L.append("## Avisos (lesao mantida, anomalia registrada)")
        L.append("")
        L.append("| Arquivo | Avisos | Valores |")
        L.append("|---|---|---|")
        for r in sorted(avisos, key=lambda x: x["saida"])[:60]:
            vals = "; ".join("{}={}".format(a, r["medidas"].get(a)) for a in r["avisos"])
            L.append("| `{}` | {} | {} |".format(r["saida"], ", ".join(r["avisos"]), vals))
        if len(avisos) > 60:
            L.append("")
            L.append("(+{} outras em `auditoria.csv`)".format(len(avisos) - 60))
        L.append("")

    nao = sorted({n for r in regs for n in r["nao_avaliados"]})
    if nao:
        L.append("## Criterios nao avaliados em alguma lesao")
        L.append("")
        for n in nao:
            q = sum(1 for r in regs if n in r["nao_avaliados"])
            L.append("- `{}`: {} lesao(oes) sem medida valida".format(n, q))
        L.append("")

    L.append("## Reproduzir")
    L.append("")
    L.append("```bash")
    L.append("cd /home/joao/MAE/pipeline")
    L.append(".venv/bin/python scripts/auditar_dataset.py {}".format(
        "../mamas_com_lesao/" + lote.name))
    L.append("```")
    L.append("")
    L.append("Parametros da auditoria: {}".format(json.dumps(cfg, ensure_ascii=False)))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------


def carregar_catalogo(lesoes: List[Dict[str, Any]]) -> Dict[str, List[List[int]]]:
    """Catalogo de tamanhos reais do treino que gerou o lote, se houver."""
    for l in lesoes:
        lora = (l.get("generation") or {}).get("lora")
        if not lora:
            continue
        p = Path(lora)
        for cand in (p.parent.parent / "catalogo_tamanhos.json", p.parent / "catalogo_tamanhos.json"):
            if cand.is_file():
                return json.loads(cand.read_text(encoding="utf-8"))
    return {}


def resolver(caminho: str, lote: Path, sub: str) -> Path:
    """Caminho gravado no metadado, com queda para o lote atual se ele mudou de lugar."""
    p = Path(caminho)
    return p if p.is_file() else (lote / sub / p.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lote", help="pasta do lote (contem metadata/ e original/)")
    ap.add_argument("--saida", default=None, help="pasta do relatorio (padrao: <lote>/auditoria)")
    ap.add_argument("--ring-mm", type=float, default=12.0, help="largura do anel de referencia")
    ap.add_argument("--limit", type=int, default=None, help="audita so as N primeiras mamas")
    ap.add_argument("--incidencias", default=None, metavar="CSV",
                    help="tabela arquivo,view,lateralidade; a MESMA passada ao gerador, senao o "
                         "peitoral e procurado (ou nao) com base no nome do arquivo")
    args = ap.parse_args(argv)

    from mamelesion.config import Config
    cfg = Config.load(None)
    lote = Path(args.lote).expanduser().resolve()
    meta_dir = lote / "metadata"
    if not meta_dir.is_dir():
        print("erro: {} nao tem metadata/".format(lote), file=sys.stderr)
        return 2
    saida = Path(args.saida).expanduser() if args.saida else lote / "auditoria"
    saida.mkdir(parents=True, exist_ok=True)

    incidencias = load_view_table(Path(args.incidencias)) if args.incidencias else None
    metas = sorted(meta_dir.glob("*.json"))
    if args.limit:
        metas = metas[:args.limit]
    todas_lesoes = [l for m in metas for l in json.loads(m.read_text(encoding="utf-8")).get("lesoes", [])]
    catalogo = carregar_catalogo(todas_lesoes)
    param = {"ring_mm": args.ring_mm, "breast_threshold": cfg.preprocessing.breast_threshold,
             "detector_long_side_mm": cfg.preprocessing.detector_long_side_mm,
             "canvas": 1024, "catalogo": bool(catalogo),
             "incidencias": str(Path(args.incidencias).resolve()) if args.incidencias else None}

    regs: List[Dict[str, Any]] = []
    for i, meta_path in enumerate(metas, 1):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lesoes = meta.get("lesoes", [])
        if not lesoes:
            continue
        origem = Path(meta["arquivo"])
        if not origem.is_file():
            print("aviso: origem ausente para {}".format(meta_path.name), file=sys.stderr)
            continue
        host = Hospedeiro(origem, cfg.preprocessing.breast_threshold, 1024,
                          cfg.preprocessing.detector_long_side_mm, incidencias)
        for l in lesoes:
            l = dict(l)
            l["files"] = {"original": str(resolver(l["files"]["original"], lote, "original")),
                          "mask_original": str(resolver(l["files"]["mask_original"], lote, "masks/original"))}
            try:
                res = auditar_lesao(host, l, catalogo, args.ring_mm)
            except Exception as exc:
                res = {"medidas": {}, "erro": "{}: {}".format(type(exc).__name__, exc)}
            g = l["box_original"]
            # numero do pipeline, so para comparacao no relatorio: ele mede no
            # canvas, onde o enquadramento pode ter cortado o ar acima da mama
            # e superestimar a distancia da pele (ver RELATORIO_QUALIDADE.md)
            res["medidas"]["distancia_pele_mm_pipeline"] = (l.get("anatomy") or {}).get("skin_distance_mm_min")
            regs.append({
                "origem": origem.stem, "arquivo_origem": str(origem),
                "saida": Path(l["files"]["original"]).name,
                "arquivo_saida": l["files"]["original"], "mascara": l["files"]["mask_original"],
                "tipo": l["lesion_type"], "indice": l["index"], "seed": l["seed"],
                "confianca": float(l.get("confidence", 0.0)),
                "pegada": [int(v) for v in g["pegada_lesao"]],
                "qc_pipeline": bool(l.get("qc_passed", False)),
                "medidas": res["medidas"], "erro": res.get("erro"),
            })
        print("[{}/{}] {}: {} lesao(oes) auditada(s)".format(i, len(metas), origem.stem, len(lesoes)),
              flush=True)

    if not regs:
        print("erro: nenhuma lesao para auditar", file=sys.stderr)
        return 2

    # Primeiro os criterios por imagem; a deduplicacao so disputa entre as que
    # ja passaram, para que uma variante com artefato nao elimine uma boa.
    for r in regs:
        if r.get("erro"):
            r.update({"checks": [], "falhas": ["auditoria_falhou"], "avisos": [],
                      "nao_avaliados": [], "veredito": "reprovada"})
        else:
            r.update(julgar(r["medidas"]))
    deduplicar([r for r in regs if r["veredito"] == "aprovada"])
    for r in regs:
        if not r.get("erro"):
            r.update(julgar(r["medidas"]))

    ordem = ["origem", "saida", "tipo", "indice", "veredito", "confianca", "qc_pipeline",
             "largura_px", "altura_px", "diametro_mm", "weber_nativo", "cnr_nativo",
             "roi_delta_ratio", "distancia_pele_mm", "cobertura_tecido", "peitoral_frac",
             "distancia_pele_mm_pipeline", "score_anatomico", "toca_borda_img", "ruido_roi_anel", "nitidez_roi_anel",
             "textura_roi", "nucleo_acima_tecido", "preenchimento_pegada", "circularidade", "contato_borda",
             "extensao_acrescimo", "cantos_acrescimo", "bloco_retangular",
             "degrau_borda_u8", "costura_excesso", "halo_u8", "saturacao_frac", "grade_score",
             "grade_z", "n_controles", "fora_da_mascara_px", "mascara_area_px",
             "mascara_componentes", "caixa_confere", "dimensoes_ok", "tamanho_no_catalogo",
             "iou_max_variantes", "similaridade_max", "nivel_anel_u8", "seed"]
    linhas = []
    for r in regs:
        linha = {"origem": r["origem"], "saida": r["saida"], "tipo": r["tipo"],
                 "indice": r["indice"], "veredito": r["veredito"], "confianca": r["confianca"],
                 "qc_pipeline": r["qc_pipeline"], "seed": r["seed"]}
        linha.update(r["medidas"])
        linha["falhas"] = ";".join(r["falhas"])
        linha["avisos"] = ";".join(r["avisos"])
        linha["erro"] = r.get("erro") or ""
        linhas.append(linha)
    campos = [c for c in ordem if any(c in l for l in linhas)]
    campos += [c for c in sorted({k for l in linhas for k in l}) if c not in campos and c not in ("falhas", "avisos", "erro")]
    campos += ["falhas", "avisos", "erro"]
    with open(saida / "auditoria.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for l in linhas:
            w.writerow(l)

    (saida / "auditoria.json").write_text(json.dumps(
        {"lote": str(lote), "parametros": param, "criterios":
         {k: {"grupo": v[0], "severidade": v[1],
              "faixa": [None if v[2] == -math.inf else v[2], None if v[3] == math.inf else v[3]],
              "descricao": v[4]} for k, v in CRITERIOS.items()},
         "lesoes": regs}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (saida / "RELATORIO_QUALIDADE.md").write_text(relatorio(regs, lote, param), encoding="utf-8")

    ap_n = sum(1 for r in regs if r["veredito"] == "aprovada")
    print()
    print("auditoria: {} lesoes, {} aprovadas, {} reprovadas".format(len(regs), ap_n, len(regs) - ap_n))
    print("  {}".format(saida / "RELATORIO_QUALIDADE.md"))
    print("  {}".format(saida / "auditoria.csv"))
    return 0 if ap_n == len(regs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
