#!/usr/bin/env python
"""Calibra o contraste das lesoes de um lote pela distribuicao das lesoes reais.

A lesao entregue e ``origem + delta``. Reescalar esse delta muda o contraste
sem gastar GPU e **sem tocar em nenhum pixel fora da mascara**: onde o delta e
zero, qualquer reescala continua zero. Dois modos:

``densidade`` (padrao)
    ``delta' = delta + (g - 1) * blur(delta, sigma) * mascara``

    Escala so a baixa frequencia -- a densidade da massa -- e deixa a textura
    fina em amplitude cheia. A correcao e multiplicada pela mascara binaria
    porque o blur espalha energia para fora dela; como o pipeline ja entrega o
    delta suavizado ate zero na borda da mascara (``feather_frac``), ali o blur
    tambem e ~0 e o corte nao cria degrau.

``ganho``
    ``delta' = g * delta``, identico a rodar o pipeline com ``lesion_gain = g``
    (a composicao nativa e linear no ganho: ``window.window_delta`` faz
    ``delta = alpha * gain * diff``).

**O modo ``ganho`` custa textura.** Escalar tudo por um escalar encolhe o ruido
e a nitidez da lesao junto com a densidade. Medido em 130 lesoes, contra a faixa
das reais (AUC 0,5 = indistinguivel):

===============  ============  ==========  ==============
descritor        sem calibrar  modo ganho  modo densidade
===============  ============  ==========  ==============
crel                     0,68        0,51            0,52
conspic                  0,70        0,47            0,48
sigma_ratio              0,48        0,37            0,46
grad_borda               0,45        0,32            0,39
radialidade              0,13        0,14            0,14
===============  ============  ==========  ==============

Os dois corrigem o contraste igual; so o modo ``densidade`` nao estraga a
granulacao e a borda no caminho.

O alvo vem dos recortes reais anotados, medidos com um descritor GEOMETRICO:
nucleo = elipse central de 55% do lado; tecido = coroa entre ``ANEL_LO`` e
``ANEL_HI`` vezes o mesmo lado. Duas escolhas importam, e as duas foram
decididas por medicao.

**Por que geometrico, e nao um segmentador.** Um segmentador adaptativo
encontra uma regiao clara em qualquer tecido e mede contraste ~0,27 mesmo numa
mama SEM lesao nenhuma (medido, varrendo o ganho ate zero). Com a elipse fixa o
mesmo nulo cai para 0,03.

**Por que o anel e afastado (0,75-0,95), e nao o imediato.** No recorte real
anotado o anel imediatamente em volta do nucleo AINDA E LESAO -- margem,
espiculas, reacao desmoplastica -- e isso deprime o contraste medido. Medindo o
mesmo nucleo com aneis cada vez mais distantes, nos 82 recortes reais:

=============  =====  =========================  ===================
anel (x lado)  reais  pipeline sem calibrar      o que da para dizer
=============  =====  =========================  ===================
0,55-0,75      0,201  0,277                      anel contaminado por lesao
**0,75-0,95**  0,551  0,435                      comparavel: real e 1,27x
0,95-1,15      0,895  0,509                      so canto do recorte real
1,15-1,35      1,023  0,600                      idem
=============  =====  =========================  ===================

Calibrar pelo anel imediato faz a lesao sintetica ficar com METADE do contraste
da real (0,275 contra 0,551) e some visualmente. Pelo anel de 0,75-0,95 o alvo
fica certo. Acima de 0,95 o anel sai do recorte real pelos lados e so restam os
cantos, que num recorte justo sao gordura escura: numero inflado, nao usar.

Isso bate com o que ``docs/FINETUNING.md`` ja media por outro caminho (razao de
contraste sintetico/real 0,76-0,84): a saida do pipeline e um pouco MENOS
contrastada que a lesao real, nao mais.

Casamento por quantil, nao pela mediana: a lesao de posto r entre as sinteticas
recebe como alvo o quantil (r+0,5)/n da distribuicao real. A mediana e a
dispersao passam a bater, e a ordem relativa e preservada -- uma lesao que saiu
intrinsecamente mais densa continua mais densa que as outras.

    .venv/bin/python scripts/calibrar_contraste.py ../mamas_com_lesao/lote_variacoes_v2 \\
        --reais /home/joao/medgan2/dados/birrads_5

Reescreve ``<lote>/original/*.png`` e grava ``<lote>/calibracao_contraste.csv``
com o ganho aplicado e o contraste antes e depois de cada lesao. Codigo de saida
0 se toda lesao foi calibrada; 1 se alguma ficou fora da faixa de ganho.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DESCRITORES = Path("/home/joao/medgan2/MedGan/avaliacao-2026-09-12/scripts")
for p in (str(REPO_ROOT), str(DESCRITORES)):
    if p not in sys.path:
        sys.path.insert(0, p)

import descritores  # noqa: E402

#: fracao do lado da janela ocupada pela lesao, igual a elipse do descritor
FRAC_NUCLEO = 0.55
#: coroa de tecido de referencia, em multiplos do lado da janela de 55%.
#: Afastada de proposito: ver o cabecalho.
ANEL_LO, ANEL_HI = 0.75, 0.95
#: janela recortada da mamografia = FATOR_JANELA x o lado de 55%, para caber o anel
FATOR_JANELA = 1.6
#: ganho permitido; fora disso a lesao e marcada e o ganho fica no limite
GANHO_MIN, GANHO_MAX = 0.15, 2.50
#: ganhos medidos para ajustar crel(g) em cada lesao
SONDAS = (0.0, 0.50, 1.00)
#: sigma (px nativos) que separa densidade (baixa frequencia) de textura
SIGMA_TEXTURA = 3.0


def compor(a: np.ndarray, d: np.ndarray, m: np.ndarray, g: float,
           modo: str, maxv: float) -> np.ndarray:
    """Recompoe a lesao com o delta reescalado por ``g``, no modo pedido."""
    if modo == "ganho":
        nd = g * d
    else:
        nd = d + (g - 1.0) * cv.GaussianBlur(d, (0, 0), SIGMA_TEXTURA) * m
    return np.clip(np.round(a + nd), 0, maxv)


# ---------------------------------------------------------------------------
# Medida
# ---------------------------------------------------------------------------


def janela_lesao(img: np.ndarray, caixa: Tuple[int, int, int, int],
                 fator: float = 1.0) -> Tuple[int, int, int, int]:
    """Recorte em que a lesao ocupa ``FRAC_NUCLEO`` do lado, como nos recortes reais.

    Com ``fator`` > 1 o recorte sai proporcionalmente maior (mesmo centro), para
    caber o anel afastado; quem mede converte os raios por ``lado55 / lado_real``.
    """
    x0, y0, x1, y1 = caixa
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    lado55 = max(x1 - x0, y1 - y0) / FRAC_NUCLEO
    h, w = img.shape[:2]
    # a janela tem de sair QUADRADA: perto da borda, desloca em vez de truncar,
    # e so encolhe quando nem deslocando cabe. Truncar deixaria o anel eliptico
    # e, na pratica, descartava ~20% das lesoes por janela nao quadrada.
    lado = int(round(fator * lado55))
    lado = max(1, min(lado, h, w))
    wy0 = int(np.clip(cy - lado // 2, 0, h - lado))
    wx0 = int(np.clip(cx - lado // 2, 0, w - lado))
    return wx0, wy0, wx0 + lado, wy0 + lado


def crel_geometrico(x: np.ndarray, frac_core: float = FRAC_NUCLEO,
                    r0: float = ANEL_LO, r1: float = ANEL_HI) -> float:
    """(mediana do nucleo - mediana da coroa) / mediana da coroa.

    Raios em unidades da meia-largura da imagem recebida. Quem chama com uma
    janela maior que a de 55% converte os tres raios pelo mesmo fator, senao o
    nucleo passa a incluir tecido e o contraste sai diluido.
    """
    h, w = x.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt(((yy - cy) / (h / 2.0)) ** 2 + ((xx - cx) / (w / 2.0)) ** 2)
    core = rr <= frac_core
    anel = (rr > r0) & (rr <= r1) & (x > 0)
    if core.sum() < 64 or anel.sum() < 64:
        return float("nan")
    mc, mr = float(np.median(x[core])), float(np.median(x[anel]))
    return (mc - mr) / (abs(mr) + 1e-6)


def crel_de(a: np.ndarray, d: np.ndarray, m: np.ndarray, g: float,
            modo: str, maxv: float, escala: float = 1.0) -> float:
    """Contraste da lesao quando o delta e escalado por ``g``.

    ``escala`` = lado_da_janela_55 / lado_da_janela_recortada; converte os raios
    para a janela maior que foi de fato recortada.
    """
    x = compor(a, d, m, g, modo, maxv).astype(np.float32)
    return crel_geometrico(x, FRAC_NUCLEO * escala, ANEL_LO * escala, ANEL_HI * escala)


#: inclinacao minima de crel(g) para o ganho ser resolvivel
INCL_MIN = 0.02


def ganho_para(a: np.ndarray, d: np.ndarray, m: np.ndarray, alvo: float,
               modo: str, maxv: float, escala: float = 1.0) -> Tuple[float, float, float, str]:
    """Ganho que leva o crel da lesao ao alvo. Devolve (ganho, antes, depois, estado).

    crel(g) e quase afim, entao ajusta uma reta pelas sondas e resolve; depois
    faz um passo de Newton com a medida real, que corrige a curvatura residual.

    Em ~3% das lesoes a reta sai plana ou com inclinacao negativa: o nucleo
    eliptico do descritor nao cobre o que de fato ficou mais denso (lesao
    deslocada dentro da pegada, ou estrutura clara do hospedeiro fora da
    elipse). Ali o ganho nao controla o contraste medido, e resolver a equacao
    joga o ganho para o teto e PIORA a lesao -- medido no lote anterior: 15
    lesoes com alvo 0,130 terminando em 0,316. Nesse caso o ganho fica em 1,0,
    a lesao sai marcada e quem julga e o criterio de fidelidade.
    """
    ys = [crel_de(a, d, m, g, modo, maxv, escala) for g in SONDAS]
    antes = ys[-1]
    xs = np.array(SONDAS, float)
    ok = np.isfinite(ys)
    if ok.sum() < 2:
        return 1.0, antes, antes, "sem_medida"
    incl, base = np.polyfit(xs[ok], np.asarray(ys, float)[ok], 1)
    if incl < INCL_MIN:
        return 1.0, antes, antes, "contraste_nao_controlavel"
    g = float(np.clip((alvo - base) / incl, GANHO_MIN, GANHO_MAX))
    obtido = crel_de(a, d, m, g, modo, maxv, escala)
    if np.isfinite(obtido) and abs(obtido - alvo) > 0.02 * max(abs(alvo), 0.05):
        g2 = float(np.clip(g + (alvo - obtido) / incl, GANHO_MIN, GANHO_MAX))
        obtido2 = crel_de(a, d, m, g2, modo, maxv, escala)
        if np.isfinite(obtido2) and abs(obtido2 - alvo) < abs(obtido - alvo):
            g, obtido = g2, obtido2
    no_limite = g <= GANHO_MIN + 1e-6 or g >= GANHO_MAX - 1e-6
    return g, antes, obtido, ("ganho_no_limite" if no_limite else "ok")


def perfil_real(raiz: str) -> np.ndarray:
    """crel de cada recorte real anotado, com o mesmo descritor e o mesmo anel.

    O recorte real ja E a janela de 55%: o quadrado central dele entra direto,
    sem fator de escala.
    """
    vals = []
    for f in sorted(glob.glob(os.path.join(raiz, "**", "*.png"), recursive=True)):
        im = cv.imread(f, cv.IMREAD_UNCHANGED)
        if im is None or im.ndim != 2 or min(im.shape[:2]) < 96:
            continue
        s = min(im.shape[:2])
        v = crel_geometrico(im[:s, :s].astype(np.float32))
        if np.isfinite(v):
            vals.append(float(v))
    return np.asarray(sorted(vals), float)


# ---------------------------------------------------------------------------
# Lote
# ---------------------------------------------------------------------------


def base_sem_calibracao(lote: Path, simular: bool = False) -> Path:
    """Pasta com a saida CRUA do pipeline, criada na primeira calibracao.

    Calibrar reescreve ``original/``. Sem uma copia do que o pipeline gerou,
    mudar o ganho depois exige regerar o lote inteiro na GPU -- e calibrar duas
    vezes compoe os ganhos em silencio. Na primeira execucao esta funcao copia
    ``original/`` para ``original_sem_calibracao/``; nas seguintes, e dali que
    a calibracao le. Assim o ganho vira um parametro que se pode reajustar em
    minutos de CPU, e o resultado nunca depende de quantas vezes rodou.
    """
    destino = lote / "original_sem_calibracao"
    if not destino.is_dir():
        if simular:                     # medir nao justifica duplicar o lote em disco
            return lote / "original"
        origem = lote / "original"
        destino.mkdir()
        for f in sorted(origem.glob("*.png")):
            shutil.copy2(f, destino / f.name)
        print("copia crua do pipeline guardada em {} ({} arquivos)".format(
            destino.name, len(list(destino.glob("*.png")))))
    return destino


def ler_lesoes(lote: Path) -> List[Dict[str, object]]:
    """Uma entrada por lesao entregue, com a pegada lida dos metadados."""
    out: List[Dict[str, object]] = []
    for meta_path in sorted((lote / "metadata").glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for les in meta.get("lesoes", meta.get("lesions", [])) or []:
            arquivos = les.get("files", les.get("arquivos", {}))
            saida = arquivos.get("original")
            mascara = arquivos.get("mask_original")
            caixa = les.get("box_original") or {}
            pegada = caixa.get("pegada_lesao")
            if not (saida and mascara and pegada):
                continue
            out.append({
                "origem": Path(meta.get("arquivo", meta.get("file", ""))).name,
                "saida": Path(saida).name,
                "mascara": Path(mascara).name,
                "pegada": tuple(int(v) for v in pegada),
                "tipo": les.get("lesion_type", les.get("tipo_lesao", "")),
            })
    return out


def main() -> int:
    global SIGMA_TEXTURA, ANEL_LO, ANEL_HI, GANHO_MAX
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lote", help="pasta do lote gerado")
    ap.add_argument("--reais", default="/home/joao/medgan2/dados/birrads_5",
                    help="raiz dos recortes reais anotados que definem o alvo")
    ap.add_argument("--entrada", default=None,
                    help="pasta das mamografias de origem (padrao: <lote>/../../sem_anomalia)")
    ap.add_argument("--modo", choices=("densidade", "ganho"), default="densidade",
                    help="densidade (padrao): escala so a baixa frequencia do delta e preserva "
                         "a textura da lesao; ganho: escala o delta inteiro, identico a rodar o "
                         "pipeline com lesion_gain=g, mas custa granulacao e nitidez")
    ap.add_argument("--sigma-textura", type=float, default=3.0,
                    help="sigma em px nativos que separa densidade de textura (modo densidade)")
    ap.add_argument("--ganho-fixo", type=float, default=None,
                    help="aplica ESTE ganho em todas as lesoes, em vez de casar a distribuicao "
                         "das reais por quantil. Use quando o alvo tirado dos recortes reais nao "
                         "for confiavel: o enquadramento justo do recorte contamina o anel "
                         "proximo com lesao e o anel distante com a periferia gordurosa. "
                         "docs/FINETUNING.md mede razao sintetico/real 0,76-0,84 por comparacao "
                         "pareada, o que implica 1,19 a 1,32.")
    ap.add_argument("--anel", default="{},{}".format(ANEL_LO, ANEL_HI),
                    help="coroa de tecido de referencia, em multiplos do lado da janela de 55%% "
                         "(padrao 0.75,0.95 -- o anel imediato ainda e lesao no recorte real)")
    ap.add_argument("--ganho-max", type=float, default=None,
                    help="teto de ganho no casamento por quantil (padrao {}); acima de ~2 a lesao "
                         "tende a saturar em mancha branca".format(GANHO_MAX))
    ap.add_argument("--simular", action="store_true",
                    help="so mede e grava o CSV, sem reescrever as imagens")
    args = ap.parse_args()

    SIGMA_TEXTURA = float(args.sigma_textura)
    if args.ganho_max is not None:
        GANHO_MAX = float(args.ganho_max)
    ANEL_LO, ANEL_HI = [float(v) for v in args.anel.split(",")]
    lote = Path(args.lote).resolve()
    entrada = Path(args.entrada) if args.entrada else lote.parent.parent / "sem_anomalia"
    if not (lote / "metadata").is_dir():
        print("erro: {} nao parece um lote (sem metadata/)".format(lote), file=sys.stderr)
        return 2

    if args.ganho_fixo is not None:
        alvos_reais = np.asarray([])
        print("ganho fixo: {:.3f} em todas as lesoes (sem casar a distribuicao das reais)"
              .format(args.ganho_fixo))
    else:
        alvos_reais = perfil_real(args.reais)
        if len(alvos_reais) < 10:
            print("erro: menos de 10 recortes reais medidos em {}".format(args.reais), file=sys.stderr)
            return 2
        print("alvo: {} recortes reais anotados, crel mediano {:.3f} (IQR {:.3f}-{:.3f})".format(
            len(alvos_reais), np.median(alvos_reais), *np.percentile(alvos_reais, [25, 75])))

    lesoes = ler_lesoes(lote)
    print("{} lesoes no lote".format(len(lesoes)))
    cru = base_sem_calibracao(lote, args.simular)   # calibrar e sempre a partir do cru

    # 1a passada: crel de cada lesao como saiu do pipeline (ganho 1,0)
    cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = {}
    medidas: List[Optional[float]] = []
    ausentes: List[str] = []
    for les in lesoes:
        src = cv.imread(str(entrada / str(les["origem"])), cv.IMREAD_UNCHANGED)
        out = cv.imread(str(cru / str(les["saida"])), cv.IMREAD_UNCHANGED)
        msk = cv.imread(str(lote / "masks" / "original" / str(les["mascara"])), cv.IMREAD_UNCHANGED)
        if src is None or out is None or msk is None or src.shape != out.shape:
            # erro de entrada, nao propriedade da lesao: a lesao ficaria sem
            # calibrar e o lote sairia com codigo 0 -- falha no fim, alto
            ausentes.append(str(les["saida"]))
            medidas.append(None)
            continue
        wx0, wy0, wx1, wy1 = janela_lesao(out, les["pegada"], FATOR_JANELA)   # type: ignore[arg-type]
        if min(wy1 - wy0, wx1 - wx0) < 96 or abs((wy1 - wy0) - (wx1 - wx0)) > 2:
            medidas.append(None)                 # janela cortada pela borda: nao da para medir o anel
            continue
        a = src[wy0:wy1, wx0:wx1].astype(np.float32)
        d = out[wy0:wy1, wx0:wx1].astype(np.float32) - a
        m = (msk[wy0:wy1, wx0:wx1] > 127).astype(np.float32)
        maxv = 65535.0 if out.dtype == np.uint16 else 255.0
        pg = les["pegada"]                                                   # type: ignore[index]
        lado55 = max(pg[2] - pg[0], pg[3] - pg[1]) / FRAC_NUCLEO
        escala = lado55 / float(wy1 - wy0)
        cache[str(les["saida"])] = (a, d, m, maxv, escala)
        medidas.append(crel_de(a, d, m, 1.0, args.modo, maxv, escala))

    validas = [i for i, v in enumerate(medidas) if v is not None and np.isfinite(v)]
    print("{} lesoes mediveis".format(len(validas)))

    # 2a passada: casamento por quantil e ganho de cada lesao
    ordem = sorted(validas, key=lambda i: medidas[i])                    # type: ignore[arg-type,return-value]
    n = len(ordem)
    alvo_de: Dict[int, float] = {}
    for r, i in enumerate(ordem):
        alvo_de[i] = (float("nan") if args.ganho_fixo is not None
                      else float(np.quantile(alvos_reais, (r + 0.5) / n)))

    linhas, fora_faixa = [], 0
    for i, les in enumerate(lesoes):
        nome = str(les["saida"])
        if i not in alvo_de or nome not in cache:
            linhas.append(dict(saida=nome, origem=les["origem"], tipo=les["tipo"], ganho="",
                               crel_antes="", crel_alvo="", crel_depois="",
                               estado="arquivo_ausente" if nome in ausentes else "nao_medida"))
            continue
        a, d, m, maxv, escala = cache[nome]
        alvo = alvo_de[i]
        if args.ganho_fixo is not None:
            g, estado = float(args.ganho_fixo), "ganho_fixo"
            antes = crel_de(a, d, m, 1.0, args.modo, maxv, escala)
            depois = crel_de(a, d, m, g, args.modo, maxv, escala)
        else:
            g, antes, depois, estado = ganho_para(a, d, m, alvo, args.modo, maxv, escala)
        fora_faixa += int(estado not in ("ok", "ganho_fixo"))

        if not args.simular:
            src = cv.imread(str(entrada / str(les["origem"])), cv.IMREAD_UNCHANGED)
            out = cv.imread(str(cru / nome), cv.IMREAD_UNCHANGED)
            msk = cv.imread(str(lote / "masks" / "original" / str(les["mascara"])), cv.IMREAD_UNCHANGED)
            maxv_i = 65535.0 if out.dtype == np.uint16 else 255.0
            af = src.astype(np.float32)
            novo = compor(af, out.astype(np.float32) - af,
                          (msk > 127).astype(np.float32), g, args.modo, maxv_i).astype(out.dtype)
            # garantia dura: onde o pipeline nao mexeu, o pixel continua o do original
            igual = out == src
            novo[igual] = src[igual]
            cv.imwrite(str(lote / "original" / nome), novo)

        linhas.append(dict(saida=nome, origem=les["origem"], tipo=les["tipo"],
                           ganho=round(g, 4), crel_antes=round(antes, 4), crel_alvo=round(alvo, 4),
                           crel_depois=round(depois, 4), estado=estado))

    destino = lote / "calibracao_contraste.csv"
    with destino.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["saida", "origem", "tipo", "ganho", "crel_antes",
                                           "crel_alvo", "crel_depois", "estado"])
        w.writeheader()
        w.writerows(linhas)

    aplicados = [float(l["ganho"]) for l in linhas if l["ganho"] != ""]
    depois = [float(l["crel_depois"]) for l in linhas if l["crel_depois"] != ""]
    if aplicados:
        estados: Dict[str, int] = {}
        for l in linhas:
            estados[str(l["estado"])] = estados.get(str(l["estado"]), 0) + 1
        print("ganho aplicado: mediana {:.3f} (p10 {:.3f}, p90 {:.3f})".format(
            np.median(aplicados), *np.percentile(aplicados, [10, 90])))
        print("estados: " + ", ".join("{} {}".format(v, k) for k, v in sorted(estados.items())))
        if len(alvos_reais):
            print("crel depois: mediana {:.3f} (IQR {:.3f}-{:.3f})  |  real: {:.3f} ({:.3f}-{:.3f})".format(
                np.median(depois), *np.percentile(depois, [25, 75]),
                np.median(alvos_reais), *np.percentile(alvos_reais, [25, 75])))
        else:
            print("crel: {:.3f} -> {:.3f} (mediana)".format(
                np.median([float(l["crel_antes"]) for l in linhas if l["crel_antes"] != ""]),
                np.median(depois)))
    print("modo: {} (sigma de textura {:.1f} px) | anel {:.2f}-{:.2f} x lado".format(
        args.modo, SIGMA_TEXTURA, ANEL_LO, ANEL_HI))
    print("gravado {}".format(destino))
    if ausentes:
        print("erro: {} lesao(oes) sem origem, saida ou mascara legivel -- ficaram SEM calibrar: {}".format(
            len(ausentes), ", ".join(ausentes[:5])), file=sys.stderr)
        return 2
    return 1 if fora_faixa else 0


if __name__ == "__main__":
    raise SystemExit(main())
