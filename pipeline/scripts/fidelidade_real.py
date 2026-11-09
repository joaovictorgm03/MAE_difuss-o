#!/usr/bin/env python
"""Fidelidade das lesoes geradas as lesoes REAIS anotadas.

Complementa ``auditar_dataset.py``. Aquele confere faixas absolutas (contraste
entre 0,05 e 1,50, diametro entre 5 e 60 mm, zero pixel fora da mascara); este
confere se cada lesao **cai na distribuicao das lesoes reais anotadas**, que e
outra pergunta: uma lesao de contraste 1,2 passa na faixa absoluta e mesmo assim
e duas vezes mais conspicua que qualquer BI-RADS 5 do conjunto real.

Os descritores sao os da avaliacao de 2026-09-12 (``descritores.medir``),
aplicados de forma identica aos dois lados:

* recorte real anotado -- a imagem inteira, nucleo aproximado pela elipse
  central de 55% do lado (os recortes nao trazem mascara);
* lesao sintetica -- uma janela em volta da pegada dimensionada para a lesao
  ocupar a MESMA fracao de 55%, mesmo descritor, mesma elipse.

Criterios, todos comparados a faixa p5-p95 das reais:

===================  ==========================================================
crel                 contraste da lesao em relacao ao tecido em volta
conspic              contraste dividido pelo desvio do tecido
sigma_ratio          ruido de alta frequencia da lesao / do tecido (granulacao)
grad_borda           gradiente na borda / do tecido (costura visivel)
psd_core             inclinacao do espectro de potencia (textura fina)
radialidade          estrutura radial / tangencial (espiculas convergentes)
===================  ==========================================================

Alem dos criterios por lesao, mede a separabilidade real x sintetico de cada
descritor (AUC; 0,5 = indistinguivel) e a duplicacao contra um lote de
referencia ja entregue, que ``auditar_dataset.py`` nao ve por so comparar
variantes dentro do proprio lote.

**Ressalva sobre ``crel`` e ``conspic`` aqui.** ``descritores.medir`` usa o anel
IMEDIATAMENTE em volta do nucleo, e no recorte real anotado esse anel ainda e
lesao (margem, espiculas) -- o contraste real sai deprimido. Como os dois lados
sao medidos do mesmo jeito e a faixa p5-p95 resultante e larga (0,055 a 0,681
em ``crel``), o criterio continua valendo para pegar o extremo: lesao que sumiu
ou lesao absurda. Mas ele NAO serve para definir o alvo de contraste -- para
isso ``calibrar_contraste.py`` usa a coroa afastada (0,75-0,95 do lado), onde
real e sintetico sao de fato comparaveis. Usar o anel imediato como alvo deixa
a lesao sintetica com metade do contraste da real.

    .venv/bin/python scripts/fidelidade_real.py ../mamas_com_lesao/lote_variacoes_v2 \\
        --reais /home/joao/medgan2/dados/birrads_5 \\
        --referencia ../dataset_escolhido/datasetfinal_MAE

Saidas em ``<lote>/fidelidade/``: ``fidelidade.csv`` e ``FIDELIDADE.md``.
Codigo de saida 0 se toda lesao foi medida; 1 se alguma reprovou.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DESCRITORES = Path("/home/joao/medgan2/MedGan/avaliacao-2026-09-12/scripts")
for p in (str(REPO_ROOT), str(DESCRITORES), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import descritores  # noqa: E402
import calibrar_contraste as calib  # noqa: E402
from calibrar_contraste import FRAC_NUCLEO, janela_lesao, ler_lesoes  # noqa: E402

#: descritor -> (rotulo, severidade, o que um valor fora da faixa significa)
#:
#: A severidade vem da separabilidade medida no lote anterior (520 lesoes,
#: `lote_variacoes_v1/fidelidade/`), nao de preferencia:
#:
#: * `crel` e `conspic` sao BLOQUEANTES, mas como GUARDA DE EXTREMO, nao como
#:   alvo: fora da faixa aqui e lesao que sumiu no tecido ou lesao absurda.
#:   O alvo de contraste quem define e `calibrar_contraste.py`, com a coroa
#:   afastada -- ver a ressalva no cabecalho deste arquivo. Medido no anel
#:   imediato a AUC sai 0,68/0,69, mas esse numero carrega o vies do recorte
#:   real justo e nao deve ser lido como "a sintetica tem contraste demais".
#: * `sigma_ratio` e `grad_borda` sao AVISO. AUC 0,50 e 0,47 -- a mediana
#:   sintetica ja bate com a real; o que sai da faixa sai pela cauda, e a faixa
#:   real e estreita (+-6% em `grad_borda`). Reprovar por isso descartaria lesao
#:   boa por dispersao, nao por defeito.
#: * `psd_core` e AVISO: AUC 0,55, faixa larga, quase ninguem sai dela.
#: * `radialidade` e AVISO COM DESLOCAMENTO CONHECIDO. AUC 0,14: as lesoes
#:   sinteticas sao sistematicamente menos espiculadas que as reais (1,30 contra
#:   1,61). E a limitacao aberta do gerador, nao ruido de medida -- fica
#:   registrada por lesao e no relatorio, e conta para a regra composta.
#:
#: Alem dos criterios individuais, `fora_da_faixa_real` (3 ou mais descritores
#: fora ao mesmo tempo) e bloqueante: um desvio isolado e cauda, tres desvios
#: independentes e lesao fora da distribuicao real.
CRITERIOS: Dict[str, Tuple[str, str, str]] = {
    "crel":        ("contraste", "bloqueante",
                    "lesao apagada ou densa demais para um BI-RADS 4/5 real"),
    "conspic":     ("conspicuidade", "bloqueante",
                    "lesao salta ou some em relacao ao ruido do tecido"),
    "sigma_ratio": ("granulacao", "aviso",
                    "textura lisa demais (veu do VAE) ou granulada demais"),
    "grad_borda":  ("borda", "aviso",
                    "degrau na borda da lesao: costura visivel"),
    "psd_core":    ("espectro", "aviso",
                    "textura fina fora da escala das lesoes reais"),
    "radialidade": ("espiculas", "aviso",
                    "massa lisa demais (sem espiculas) ou estrutura radial exagerada"),
}
#: quantis das reais que delimitam a faixa aceita
Q_LO, Q_HI = 5.0, 95.0
#: quantos descritores fora da faixa ao mesmo tempo reprovam a lesao
MAX_FORA_DA_FAIXA = 2
#: granulado "sal e pimenta": ruido de alta frequencia da lesao / do tecido. A
#: faixa das reais vai ate ~1,13 e passar dela e so aviso; acima deste teto e
#: bloqueante. Posto olhando as entregues da entrega 1 do lote_birads1_v1: das 16
#: acima de 1,8, 15 eram granulado nitido dentro da caixa; entre 1,58 e 1,66 havia
#: massa plausivel e bola granulada lado a lado -- por isso o teto nao e mais baixo.
GRANULADO_MAX = 1.8


def medir_reais(raiz: str) -> Tuple[List[Dict[str, float]], Dict[str, Tuple[float, float]]]:
    """Descritores de cada recorte real anotado e a faixa p5-p95 de cada um."""
    regs = []
    for f in sorted(glob.glob(os.path.join(raiz, "**", "*.png"), recursive=True)):
        im = cv.imread(f, cv.IMREAD_UNCHANGED)
        if im is None or im.ndim != 2 or min(im.shape[:2]) < 64:
            continue
        d = descritores.medir(im.astype(np.float32), None)
        if d is not None:
            regs.append({k: float(d[k]) for k in CRITERIOS})
    faixas = {}
    for k in CRITERIOS:
        v = np.array([r[k] for r in regs], float)
        v = v[np.isfinite(v)]
        faixas[k] = (float(np.percentile(v, Q_LO)), float(np.percentile(v, Q_HI)))
    return regs, faixas


def medir_lote(lote: Path, entrada: Path) -> List[Dict[str, object]]:
    """Descritores de cada lesao entregue, na janela de 55% e com o mesmo descritor."""
    regs = []
    for les in ler_lesoes(lote):
        out = cv.imread(str(lote / "original" / str(les["saida"])), cv.IMREAD_UNCHANGED)
        if out is None:
            continue
        wx0, wy0, wx1, wy1 = janela_lesao(out, les["pegada"])            # type: ignore[arg-type]
        if min(wy1 - wy0, wx1 - wx0) < 96:
            continue
        d = descritores.medir(out[wy0:wy1, wx0:wx1].astype(np.float32), None)
        if d is None:
            continue
        regs.append(dict(saida=str(les["saida"]), origem=str(les["origem"]), tipo=str(les["tipo"]),
                         pegada=les["pegada"], **{k: float(d[k]) for k in CRITERIOS}))
    return regs


def crel_afastado(lote: Path, entrada: Path, les: Dict[str, object]) -> float:
    """Contraste nucleo x coroa AFASTADA (0,75-0,95 do lado), a medida da calibracao.

    E a mesma medida que ``calibrar_contraste.py`` usa como alvo: nela real e
    sintetico sao comparaveis, porque a coroa afastada do recorte real ja e
    tecido, e nao margem da lesao como o anel imediato de ``descritores.medir``.
    """
    src = cv.imread(str(entrada / str(les["origem"])), cv.IMREAD_UNCHANGED)
    out = cv.imread(str(lote / "original" / str(les["saida"])), cv.IMREAD_UNCHANGED)
    msk = cv.imread(str(lote / "masks" / "original" / str(les["mascara"])), cv.IMREAD_UNCHANGED)
    if src is None or out is None or msk is None or src.shape != out.shape:
        return float("nan")
    pg = les["pegada"]
    wx0, wy0, wx1, wy1 = janela_lesao(out, pg, calib.FATOR_JANELA)          # type: ignore[arg-type]
    if min(wy1 - wy0, wx1 - wx0) < 96 or abs((wy1 - wy0) - (wx1 - wx0)) > 2:
        return float("nan")                    # janela cortada pela borda: sem coroa completa
    a = src[wy0:wy1, wx0:wx1].astype(np.float32)
    d = out[wy0:wy1, wx0:wx1].astype(np.float32) - a
    m = (msk[wy0:wy1, wx0:wx1] > 127).astype(np.float32)
    maxv = 65535.0 if out.dtype == np.uint16 else 255.0
    escala = max(pg[2] - pg[0], pg[3] - pg[1]) / FRAC_NUCLEO / float(wy1 - wy0)   # type: ignore[index]
    return float(calib.crel_de(a, d, m, 1.0, "densidade", maxv, escala))


def auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(b > a), por postos. 0,5 = as duas amostras nao se separam."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    todos = np.concatenate([a, b])
    r = np.argsort(np.argsort(todos)) + 1.0
    rb = r[len(a):].sum()
    return float((rb - len(b) * (len(b) + 1) / 2.0) / (len(a) * len(b)))


def iou(p: Tuple[int, int, int, int], q: Tuple[int, int, int, int]) -> float:
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    ap = (p[2] - p[0]) * (p[3] - p[1])
    aq = (q[2] - q[0]) * (q[3] - q[1])
    return float(inter) / float(ap + aq - inter)


def pegadas_referencia(ref: Optional[str]) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """Pegadas por mamografia de origem num dataset ja entregue (manifest.csv)."""
    if not ref:
        return {}
    man = Path(ref) / "manifest.csv"
    if not man.is_file():
        return {}
    por_origem: Dict[str, List[Tuple[int, int, int, int]]] = {}
    for r in csv.DictReader(man.open(encoding="utf-8")):
        por_origem.setdefault(r["mamografia_origem"], []).append(
            (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])))
    return por_origem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lote")
    ap.add_argument("--reais", default="/home/joao/medgan2/dados/birrads_5")
    ap.add_argument("--entrada", default=None)
    ap.add_argument("--referencia", default=None,
                    help="datasetfinal_MAE de um lote ja entregue, para duplicacao entre lotes")
    ap.add_argument("--contraste-afastado", action="store_true",
                    help="julga o contraste na coroa afastada (a regua da calibracao): crel_afastado "
                         "fora de p5-p95 das reais reprova; crel/conspic do anel imediato so reprovam "
                         "lesao apagada (abaixo de p5) -- acima de p95 vira aviso, porque ali o recorte "
                         "real ainda e lesao e o teto sai rebaixado")
    ap.add_argument("--iou-max", type=float, default=0.50,
                    help="sobreposicao maxima com uma lesao do lote de referencia")
    args = ap.parse_args()

    lote = Path(args.lote).resolve()
    entrada = Path(args.entrada) if args.entrada else lote.parent.parent / "sem_anomalia"
    reais, faixas = medir_reais(args.reais)
    print("{} recortes reais anotados definem as faixas".format(len(reais)))
    for k, (lo, hi) in faixas.items():
        print("  {:12s} p5={:8.3f}  p95={:8.3f}".format(k, lo, hi))

    sint = medir_lote(lote, entrada)
    print("{} lesoes sinteticas medidas".format(len(sint)))
    ref = pegadas_referencia(args.referencia)

    faixa_afastado = None
    if args.contraste_afastado:
        reais_af = calib.perfil_real(args.reais)
        faixa_afastado = (float(np.percentile(reais_af, Q_LO)), float(np.percentile(reais_af, Q_HI)))
        print("crel afastado das reais: p5 {:.3f}, mediana {:.3f}, p95 {:.3f} ({} recortes)".format(
            faixa_afastado[0], float(np.median(reais_af)), faixa_afastado[1], len(reais_af)))
        les_por_saida = {str(l["saida"]): l for l in ler_lesoes(lote)}

    linhas = []
    for s in sint:
        fora = [k for k in CRITERIOS
                if not np.isfinite(s[k]) or not (faixas[k][0] <= s[k] <= faixas[k][1])]
        falhas = [k for k in fora if CRITERIOS[k][1] == "bloqueante"]
        avisos = [k for k in fora if CRITERIOS[k][1] == "aviso"]
        s["crel_afastado"] = float("nan")
        if faixa_afastado is not None:
            # anel imediato: so o lado de baixo continua bloqueante
            for k in ("crel", "conspic"):
                if k in falhas and np.isfinite(s[k]) and s[k] > faixas[k][1]:
                    falhas.remove(k)
                    avisos.append(k)
                    fora.remove(k)          # nem conta para fora_da_faixa_real: e a regua enviesada
            s["crel_afastado"] = crel_afastado(lote, entrada, les_por_saida[s["saida"]])
            if not (np.isfinite(s["crel_afastado"])
                    and faixa_afastado[0] <= s["crel_afastado"] <= faixa_afastado[1]):
                falhas.append("crel_afastado")
        if len(fora) > MAX_FORA_DA_FAIXA:
            falhas.append("fora_da_faixa_real")
        if np.isfinite(s["sigma_ratio"]) and s["sigma_ratio"] > GRANULADO_MAX:
            falhas.append("granulado")
        iou_ref = max([iou(s["pegada"], q) for q in ref.get(s["origem"], [])] or [0.0])
        if iou_ref > args.iou_max:
            falhas.append("iou_lote_referencia")
        linhas.append(dict(saida=s["saida"], origem=s["origem"], tipo=s["tipo"],
                           **{k: round(s[k], 4) for k in CRITERIOS},
                           crel_afastado=round(s["crel_afastado"], 4),
                           iou_lote_referencia=round(iou_ref, 4),
                           n_fora_da_faixa=len(fora), falhas=";".join(falhas),
                           avisos=";".join(avisos),
                           veredito="aprovada" if not falhas else "reprovada"))

    dest = lote / "fidelidade"
    dest.mkdir(exist_ok=True)
    campos = (["saida", "origem", "tipo"] + list(CRITERIOS) +
              ["crel_afastado", "iou_lote_referencia", "n_fora_da_faixa", "falhas", "avisos", "veredito"])
    with (dest / "fidelidade.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)

    aprov = sum(1 for l in linhas if l["veredito"] == "aprovada")
    contagem: Dict[str, int] = {}
    fora_total: Dict[str, int] = {}
    for l in linhas:
        for f in str(l["falhas"]).split(";"):
            if f:
                contagem[f] = contagem.get(f, 0) + 1
        for c in str(l["falhas"]).split(";") + str(l["avisos"]).split(";"):
            if c in CRITERIOS:
                fora_total[c] = fora_total.get(c, 0) + 1

    md = ["# Fidelidade as lesoes reais anotadas", "",
          "Faixas p{:.0f}-p{:.0f} de {} recortes reais de `{}`, medidos com o mesmo "
          "descritor geometrico das sinteticas.".format(Q_LO, Q_HI, len(reais), args.reais), "",
          "| lesoes medidas | aprovadas | reprovadas |", "|---|---:|---:|",
          "| {} | {} | {} |".format(len(linhas), aprov, len(linhas) - aprov), "",
          "## Separabilidade real x sintetico", "",
          "AUC 0,5 = as duas distribuicoes nao se separam; quanto mais longe de 0,5, "
          "maior a diferenca.", "",
          "| descritor | severidade | real (mediana) | sintetico (mediana) | faixa aceita | AUC | fora da faixa |",
          "|---|---|---:|---:|---|---:|---:|"]
    for k, (rot, sev, _) in CRITERIOS.items():
        r = np.array([x[k] for x in reais], float)
        s = np.array([x[k] for x in sint], float)
        md.append("| {} (`{}`) | {} | {:.3f} | {:.3f} | {:.3f} a {:.3f} | {:.2f} | {} |".format(
            rot, k, sev, float(np.nanmedian(r)), float(np.nanmedian(s)),
            faixas[k][0], faixas[k][1], auc(r, s), fora_total.get(k, 0)))
    md += ["", "## Motivos de reprovacao", "", "| criterio | lesoes | o que significa |", "|---|---:|---|"]
    extra = {"iou_lote_referencia": ("duplicacao entre lotes", "bloqueante",
                                     "ocupa quase o mesmo lugar que uma lesao do lote de referencia"),
             "crel_afastado": ("contraste na coroa afastada", "bloqueante",
                               "contraste fora da faixa p5-p95 das reais, na regua da calibracao"),
             "granulado": ("granulado", "bloqueante",
                           "sigma_ratio acima de {}: textura sal e pimenta dentro da lesao".format(GRANULADO_MAX)),
             "fora_da_faixa_real": ("fora da distribuicao real", "bloqueante",
                                    "mais de {} descritores fora da faixa das reais ao mesmo "
                                    "tempo".format(MAX_FORA_DA_FAIXA))}
    for k, n in sorted(contagem.items(), key=lambda kv: -kv[1]):
        rot, _sev, sig = CRITERIOS.get(k, extra.get(k, (k, "", "")))
        md.append("| `{}` ({}) | {} | {} |".format(k, rot, n, sig))
    if not contagem:
        md.append("| — | 0 | nenhuma lesao reprovou |")
    (dest / "FIDELIDADE.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n{} aprovadas, {} reprovadas".format(aprov, len(linhas) - aprov))
    for k, n in sorted(contagem.items(), key=lambda kv: -kv[1]):
        print("  {:22s} {}".format(k, n))
    print("gravado {}".format(dest))
    return 1 if aprov < len(linhas) else 0


if __name__ == "__main__":
    raise SystemExit(main())
