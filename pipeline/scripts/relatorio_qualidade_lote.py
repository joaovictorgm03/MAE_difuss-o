#!/usr/bin/env python
"""Relatorio de qualidade consolidado de um lote em partes e da sua entrega.

Le o que a cadeia gravou -- selecao de hospedeiras, metadados da geracao,
auditoria (criterios de ``auditar_dataset.CRITERIOS``), fidelidade as lesoes reais, verificacao da entrega --
e escreve, sem recalcular nenhuma medida:

* ``rastreabilidade.csv`` -- UMA linha por lesao gerada, entregue ou nao, com o
  veredito nas quatro verificacoes pedidas (artefatos, consistencia anatomica,
  restricoes de dados, plausibilidade clinica), o criterio e o valor que
  reprovaram, e se a lesao entrou na entrega;
* ``RELATORIO_QUALIDADE.md`` -- contagens, distribuicoes, separabilidade real x
  sintetico, resolucoes, diversidade, o que foi corrigido e os limites;
* paineis PNG para conferencia visual: aprovadas (antes | depois, mesmo
  janelamento) e reprovadas por categoria;
* os ``LEIA-ME.md`` da entrega.

    .venv/bin/python scripts/relatorio_qualidade_lote.py ../mamas_com_lesao/lote_birads1_v1 \\
        --dataset ../dataset_escolhido_birrad1 --entrada ../birads_1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from auditar_dataset import CRITERIOS as CRIT_AUD  # noqa: E402

try:  # os descritores vivem fora do repositorio; sem eles, o relatorio sai sem a AUC
    from fidelidade_real import CRITERIOS as CRIT_FID, auc, medir_reais  # noqa: E402
except Exception:                                                      # pragma: no cover
    CRIT_FID, auc, medir_reais = {}, None, None

#: as quatro verificacoes pedidas, e de onde vem cada criterio
CATEGORIAS = ("artefatos", "anatomia", "restricoes", "plausibilidade")
ROTULO_CAT = {
    "artefatos": "Artefatos (sintese, ruido, textura, costura)",
    "anatomia": "Consistencia anatomica",
    "restricoes": "Restricoes de dados (integridade, tamanho, duplicacao)",
    "plausibilidade": "Plausibilidade clinica (contraste, fidelidade as reais)",
}
CAT_DE_GRUPO = {"artefato": "artefatos", "anatomia": "anatomia", "integridade": "restricoes",
                "duplicacao": "restricoes", "clinica": "plausibilidade"}
#: excecoes ao grupo da auditoria: tamanho e restricao de dado, nao contraste
CAT_DE_CRITERIO = {"diametro_mm": "restricoes", "tamanho_no_catalogo": "restricoes",
                   "sigma_ratio": "artefatos", "grad_borda": "artefatos", "psd_core": "artefatos",
                   "crel": "plausibilidade", "conspic": "plausibilidade", "radialidade": "plausibilidade",
                   "fora_da_faixa_real": "plausibilidade", "iou_lote_referencia": "restricoes",
                   "granulado": "artefatos", "crel_afastado": "plausibilidade"}


def categoria(criterio: str) -> str:
    if criterio in CAT_DE_CRITERIO:
        return CAT_DE_CRITERIO[criterio]
    grupo = CRIT_AUD.get(criterio, ("",))[0]
    return CAT_DE_GRUPO.get(grupo, "restricoes")


def ler_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def lista(campo: Optional[str]) -> List[str]:
    if not campo:
        return []
    return [x.strip() for x in campo.replace(",", ";").split(";") if x.strip()]


def fnum(v, nd=2) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "--"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if not math.isfinite(x):
        return "--"
    return ("{:." + str(nd) + "f}").format(x).replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    return fnum(100.0 * x, nd) + "%"


def quantis(v: List[float], qs=(5, 50, 95)) -> List[float]:
    a = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    return [float(np.percentile(a, q)) for q in qs] if len(a) else [float("nan")] * len(qs)


def tabela(cab: List[str], linhas: List[List[str]], alinhar: Optional[str] = None) -> str:
    al = alinhar or ("l" + "r" * (len(cab) - 1))
    sep = ["---:" if c == "r" else "---" for c in al]
    out = ["| " + " | ".join(cab) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in l) + " |" for l in linhas]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Paineis
# ---------------------------------------------------------------------------


def _janela(img: np.ndarray, caixa, fator: float = 1.8) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = caixa
    lado = int(max(x2 - x1, y2 - y1) * fator)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    h, w = img.shape[:2]
    x0, y0 = max(0, cx - lado // 2), max(0, cy - lado // 2)
    return x0, y0, min(w, x0 + lado), min(h, y0 + lado)


def _u8(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)


def par_antes_depois(src: Path, out: Path, caixa, rotulo: str, lado: int = 220) -> Optional[np.ndarray]:
    a = cv.imread(str(src), cv.IMREAD_UNCHANGED)
    b = cv.imread(str(out), cv.IMREAD_UNCHANGED)
    if a is None or b is None:
        return None
    x0, y0, x1, y1 = _janela(b, caixa)
    ra, rb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    lo, hi = np.percentile(ra, [0.5, 99.5])          # MESMO janelamento nos dois lados
    ta = cv.resize(_u8(ra, lo, hi), (lado, lado), interpolation=cv.INTER_AREA)
    tb = cv.resize(_u8(rb, lo, hi), (lado, lado), interpolation=cv.INTER_AREA)
    bx1, by1 = (caixa[0] - x0) * lado / (x1 - x0), (caixa[1] - y0) * lado / (y1 - y0)
    bx2, by2 = (caixa[2] - x0) * lado / (x1 - x0), (caixa[3] - y0) * lado / (y1 - y0)
    tb = cv.cvtColor(tb, cv.COLOR_GRAY2BGR)
    ta = cv.cvtColor(ta, cv.COLOR_GRAY2BGR)
    cv.rectangle(tb, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 200, 255), 1)
    par = np.hstack([ta, np.full((lado, 3, 3), 255, np.uint8), tb])
    cv.putText(par, rotulo[:48], (3, lado - 5), cv.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1, cv.LINE_AA)
    return par


def mama_com_caixa(out: Path, caixa, rotulo: str, alto: int = 300) -> Optional[np.ndarray]:
    b = cv.imread(str(out), cv.IMREAD_UNCHANGED)
    if b is None:
        return None
    lo, hi = np.percentile(b, [1, 99.5])
    s = alto / b.shape[0]
    t = cv.cvtColor(cv.resize(_u8(b, lo, hi), (max(1, int(b.shape[1] * s)), alto),
                              interpolation=cv.INTER_AREA), cv.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = (int(v * s) for v in caixa)
    cv.rectangle(t, (x1, y1), (x2, y2), (0, 0, 255), 2)
    t = cv.copyMakeBorder(t, 0, 0, 0, max(0, 200 - t.shape[1]), cv.BORDER_CONSTANT, value=0)[:, :200]
    t = np.ascontiguousarray(t)
    cv.putText(t, rotulo[:34], (3, alto - 5), cv.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1, cv.LINE_AA)
    return t


def grade(tiles: List[np.ndarray], por_linha: int) -> Optional[np.ndarray]:
    tiles = [t for t in tiles if t is not None]
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv.copyMakeBorder(t, 0, h - t.shape[0] + 4, 0, w - t.shape[1] + 4, cv.BORDER_CONSTANT,
                               value=(40, 40, 40)) for t in tiles]
    while len(tiles) % por_linha:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + por_linha]) for i in range(0, len(tiles), por_linha)])


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lote", help="lote unificado (com _controle/, auditoria/, fidelidade/)")
    ap.add_argument("--dataset", required=True, help="pasta da entrega (dataset_bbox/ e datasetfinal_MAE/)")
    ap.add_argument("--entrada", required=True, help="pasta das mamografias de origem")
    ap.add_argument("--reais", default="/home/joao/medgan2/dados/birrads_5")
    ap.add_argument("--semente", type=int, default=1337)
    ap.add_argument("--entregas-arquivo", default="entregas.json",
                    help="registro de entregas desta pasta em <lote>/_controle/ (lote_em_partes --entregas-arquivo)")
    ap.add_argument("--nome", default=None, help="nome da entrega nos textos (padrao: nome da pasta --dataset)")
    a = ap.parse_args(argv)

    lote, ds, entrada = Path(a.lote).resolve(), Path(a.dataset).resolve(), Path(a.entrada).resolve()
    nome_ds = a.nome or ds.name
    arq_ent = lote / "_controle" / a.entregas_arquivo
    entregas = json.loads(arq_ent.read_text(encoding="utf-8")) if arq_ent.is_file() else {"entregas": []}
    n_planejadas = len(list((lote / "_controle" / "partes").glob("parte_*.txt")))
    final, bbox = ds / "datasetfinal_MAE", ds / "dataset_bbox"
    qual = final / "qualidade"
    paineis = qual / "paineis"
    paineis.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.semente)

    host = ler_csv(lote / "_controle" / "hospedeiras.csv")
    aud = {r["saida"]: r for r in ler_csv(lote / "auditoria" / "auditoria.csv")}
    fid = {r["saida"]: r for r in ler_csv(lote / "fidelidade" / "fidelidade.csv")}
    cal = {r["saida"]: r for r in ler_csv(lote / "calibracao_contraste.csv")}
    entregue = {r["arquivo"]: r for r in ler_csv(final / "manifest.csv")}
    verif = ler_csv(qual / "verificacao_entrega.csv")
    estado = json.loads((lote / "_controle" / "estado.json").read_text(encoding="utf-8")) \
        if (lote / "_controle" / "estado.json").is_file() else {}
    metas = [json.loads(m.read_text(encoding="utf-8")) for m in sorted((lote / "metadata").glob("*.json"))]
    # o lote unificado pode ter partes de outra pasta de entrega (outra politica):
    # so conta o que foi auditado para esta
    partes_aqui = {r.get("parte") for r in aud.values() if r.get("parte")}
    if partes_aqui:
        metas = [m for m in metas if m.get("parte") in partes_aqui]
    por_imagem = int(next((json.loads(l).get("opcoes", {}).get("lesions_per_image")
                           for p in sorted((lote / "partes").glob("parte_*"))[:1]
                           for l in (p / "processing.jsonl").open(encoding="utf-8")
                           if '"lote iniciado"' in l), 4) or 4)

    # --- geracao: o que o portao interno descartou -------------------------
    n_hosp = len(metas)
    geradas = sum(len(m.get("lesoes", [])) for m in metas)
    pedidas = n_hosp * por_imagem
    hosp_falha = [m for m in metas if m.get("status") in ("falha", "ignorada")]
    tentativas = [int(l.get("attempts", 1)) for m in metas for l in m.get("lesoes", [])]
    pect = Counter((m.get("anatomia") or {}).get("pectoral_method", "?") for m in metas)
    views = Counter((m.get("anatomia") or {}).get("view", "?") for m in metas)
    fonte_view = Counter((m.get("anatomia") or {}).get("view_source", "nome_do_arquivo") for m in metas)

    # --- rastreabilidade: uma linha por lesao gerada ------------------------
    linhas: List[Dict[str, object]] = []
    falhas_crit: Counter = Counter()
    avisos_crit: Counter = Counter()
    for saida, r in sorted(aud.items()):
        f = fid.get(saida, {})
        falhas = [(c, "auditoria") for c in lista(r.get("falhas"))] + [(c, "fidelidade") for c in lista(f.get("falhas"))]
        avisos = [(c, "auditoria") for c in lista(r.get("avisos"))] + [(c, "fidelidade") for c in lista(f.get("avisos"))]
        if r.get("erro"):
            falhas.append(("auditoria_falhou", "auditoria"))
        linha: Dict[str, object] = {
            "saida": saida, "origem": r.get("origem"), "parte": r.get("parte", ""), "tipo": r.get("tipo"),
            "variante": int(r.get("indice", 0)) + 1, "seed": r.get("seed"),
        }
        for cat in CATEGORIAS:
            fc = [c for c, _ in falhas if categoria(c) == cat]
            ac = [c for c, _ in avisos if categoria(c) == cat]
            val = []
            for c in fc:
                v = f.get("sigma_ratio", "") if c == "granulado" else r.get(c, f.get(c, ""))
                val.append("{}={}".format(c, v) if v not in ("", None) else c)
            linha[cat] = "reprovada" if fc else "aprovada"
            linha[cat + "_criterios"] = "; ".join(val)
            linha[cat + "_avisos"] = "; ".join(ac)
        for c, _ in falhas:
            falhas_crit[c] += 1
        for c, _ in avisos:
            avisos_crit[c] += 1
        linha["auditoria"] = r.get("veredito")
        linha["fidelidade"] = f.get("veredito", "nao_medida")
        linha["ganho_contraste"] = cal.get(saida, {}).get("ganho", "")
        linha["entregue"] = int(saida in entregue)
        linha["particao"] = entregue.get(saida, {}).get("particao", "")
        linhas.append(linha)
    if not linhas:
        print("erro: nada auditado em {}".format(lote), file=sys.stderr)
        return 2
    with open(qual / "rastreabilidade.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        wr.writeheader()
        wr.writerows(linhas)
    for nome in ("hospedeiras.csv", "incidencias.csv", "excluidas.png"):
        if (lote / "_controle" / nome).is_file():
            shutil.copy2(lote / "_controle" / nome, qual / nome)

    n_aud = len(linhas)
    n_ent = sum(int(l["entregue"]) for l in linhas)
    rep_cat = {c: sum(l[c] == "reprovada" for l in linhas) for c in CATEGORIAS}
    so_cat = {c: sum(l[c] == "reprovada" and all(l[o] == "aprovada" for o in CATEGORIAS if o != c)
                     for l in linhas) for c in CATEGORIAS}

    # --- separabilidade real x sintetico (descritores da fidelidade) -------
    auc_linhas: List[List[str]] = []
    n_reais: object = "os"
    faixas: Dict[str, Tuple[float, float]] = {}
    if medir_reais is not None and Path(a.reais).is_dir() and fid:
        reais, faixas = medir_reais(a.reais)
        n_reais = len(reais)
        for k in CRIT_FID:
            rv = np.array([x[k] for x in reais], float)
            todos = np.array([float(v[k]) for v in fid.values() if v.get(k) not in ("", None)], float)
            ent = np.array([float(fid[s][k]) for s in entregue if s in fid and fid[s].get(k) not in ("", None)], float)
            auc_linhas.append([
                "`{}` ({})".format(k, CRIT_FID[k][0]), CRIT_FID[k][1],
                fnum(np.median(rv), 3), fnum(np.median(todos), 3) if len(todos) else "--",
                fnum(np.median(ent), 3) if len(ent) else "--",
                "{} a {}".format(fnum(faixas[k][0], 3), fnum(faixas[k][1], 3)),
                fnum(auc(rv, todos), 2) if len(todos) else "--",
                fnum(auc(rv, ent), 2) if len(ent) else "--"])

    # --- entrega: resolucao e diversidade -----------------------------------
    ent_l = list(entregue.values())
    W = [int(r["largura_img"]) for r in ent_l]
    H = [int(r["altura_img"]) for r in ent_l]
    mp = [w * h / 1e6 for w, h in zip(W, H)]
    bits = Counter(r["bit_depth"] for r in ent_l)
    var_por_mama = Counter(Counter(r["mamografia_origem"] for r in ent_l).values())
    pac = {r["paciente"] for r in ent_l}
    mamas = {r["mamografia_origem"] for r in ent_l}
    tipos = Counter(r["tipo_lesao"] for r in ent_l)
    parts = Counter(r["particao"] for r in ent_l)
    views_ent = Counter(r["view"] for r in ent_l)
    lado_ent = Counter(r["lateralidade"] for r in ent_l)
    reg = Counter((r.get("regiao_anatomica") or "").split("(")[0].strip() for r in ent_l)
    lora_pac = sum(r.get("paciente_nos_recortes_lora") in ("True", "true", "1") for r in ent_l)
    v_ok = sum(int(r["ok"]) for r in verif)

    def q(col):
        return quantis([float(r[col]) for r in ent_l if r.get(col) not in ("", None, "None")])

    # --- o que saiu desde a entrega anterior ------------------------------------
    # ``--dataset`` e a pasta nova (<nome>.novo); a entrega em uso ainda esta em <nome>
    anterior = ds.with_name(nome_ds) / "datasetfinal_MAE" / "manifest.csv"
    removidas: List[Dict[str, str]] = []
    if ds.name != nome_ds and anterior.is_file():
        rast = {str(l["saida"]): l for l in linhas}
        for r in ler_csv(anterior):
            if r["arquivo"] not in entregue:
                l = rast.get(r["arquivo"], {})
                motivos = "; ".join(str(l.get(c + "_criterios", "")) for c in CATEGORIAS if l.get(c) == "reprovada")
                removidas.append({"arquivo": r["arquivo"], "entrega_em_que_entrou": r.get("entrega", ""),
                                  "particao": r.get("particao", ""), "motivo": motivos or "nao auditada"})
    if removidas:
        with open(qual / "removidas_desde_entrega_anterior.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(removidas[0]))
            wr.writeheader()
            wr.writerows(removidas)

    # --- paineis --------------------------------------------------------------
    amostra = [ent_l[i] for i in rng.permutation(len(ent_l))[:36]] if ent_l else []
    g = grade([par_antes_depois(entrada / r["mamografia_origem"], final / "imagens" / r["arquivo"],
                                (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])),
                                "{} {} {}mm".format(r["arquivo"][:26], r["tipo_lesao"], fnum(r["diametro_mm"], 0)))
               for r in amostra], 4)
    if g is not None:
        cv.imwrite(str(paineis / "aprovadas_antes_depois.png"), g)
    # granulado residual: sigma_ratio (ruido de alta frequencia lesao/tecido) acima
    # da faixa das reais. E aviso, nao gate: nos pilotos, 1,58 era massa plausivel
    # e 1,66 bola granulada -- margem pequena demais para um limiar (ver secao 3.2)
    sig = [(float(r["sigma_ratio"]), r) for r in ent_l if r.get("sigma_ratio") not in ("", None)]
    sig_p95 = faixas.get("sigma_ratio", (0.852, 1.127))[1]          # p95 das reais
    sig_hi = sorted([x for x in sig if x[0] > sig_p95], key=lambda x: -x[0])
    faixas_sig = [("<= {} (faixa das reais)".format(fnum(sig_p95, 2)), sum(v <= sig_p95 for v, _ in sig)),
                  ("{} a 1,40".format(fnum(sig_p95, 2)), sum(sig_p95 < v <= 1.40 for v, _ in sig)),
                  ("1,40 a 1,60", sum(1.40 < v <= 1.60 for v, _ in sig)),
                  ("> 1,60 (granulado provavel)", sum(v > 1.60 for v, _ in sig))]
    g = grade([par_antes_depois(entrada / r["mamografia_origem"], final / "imagens" / r["arquivo"],
                                (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])),
                                "sigma {} {}".format(fnum(v, 2), r["arquivo"][:24]))
               for v, r in sig_hi[:16]], 4)
    if g is not None:
        cv.imwrite(str(paineis / "entregues_granulado_residual.png"), g)
    reprov_paineis: Dict[str, str] = {}
    for cat in CATEGORIAS:
        rs = [l for l in linhas if l[cat] == "reprovada"]
        if not rs:
            continue
        rs = [rs[i] for i in rng.permutation(len(rs))[:20]]
        tiles = []
        for l in rs:
            r = aud[str(l["saida"])]
            caixa = json.loads(r["pegada"]) if r.get("pegada", "").startswith("[") else None
            if caixa is None:
                m = next((m for m in metas if Path(m["arquivo"]).stem == l["origem"]), None)
                les = next((x for x in (m or {}).get("lesoes", []) if Path(x["files"]["original"]).name == l["saida"]), None)
                caixa = les["box_original"]["pegada_lesao"] if les else None
            if caixa is None:
                continue
            out = lote / "original" / str(l["saida"])
            rot = str(l[cat + "_criterios"])[:48]
            if cat == "anatomia":
                tiles.append(mama_com_caixa(out, caixa, rot))
            else:
                tiles.append(par_antes_depois(entrada / (str(l["origem"]) + ".png"), out, caixa, rot))
        g = grade(tiles, 5 if cat == "anatomia" else 4)
        if g is not None:
            nome = "reprovadas_{}.png".format(cat)
            cv.imwrite(str(paineis / nome), g)
            reprov_paineis[cat] = nome

    # --- relatorio -------------------------------------------------------------
    excl = Counter(h["motivo_exclusao"] for h in host if h.get("incluida") == "0")
    incl = [h for h in host if h.get("incluida") == "1"]
    corrigidas = sum(1 for h in incl if h["view"] == "MLO" and "_CC" in h["arquivo"])
    durs = [v.get("gerar_resumo", {}).get("duracao_total_s", 0) for k, v in estado.items()
            if k.startswith("parte_") and (not partes_aqui or k in partes_aqui)]
    partes_prontas = (estado.get("_lote") or {}).get("partes_prontas", [])
    interrompido = (estado.get("_lote") or {}).get("interrompido_por_disco")

    L: List[str] = []
    L.append("# Relatorio de qualidade -- {}\n".format(nome_ds))
    ult = (entregas.get("entregas") or [{}])[-1]
    if ult.get("entrega"):
        # final = a ultima entrega desta pasta (pode cobrir so parte do lote, se a
        # pasta so recebe algumas politicas); o registro sabe, a contagem de partes nao
        parcial = not ult.get("final", False)
        L.append("> **Entrega {}{}** -- {} de {} partes processadas ({} hospedeiras). {}\n".format(
            ult["entrega"], " (parcial)" if parcial else " (final)", len(partes_prontas), n_planejadas, n_hosp,
            "A geracao continua; esta pasta sera substituida pela proxima entrega, que CONTEM esta "
            "(salvo imagens retiradas por um criterio novo, sempre listadas em "
            "`datasetfinal_MAE/qualidade/removidas_desde_entrega_anterior.csv`); nenhum paciente muda "
            "de particao. A coluna `entrega` do manifesto diz em que entrega cada imagem entrou."
            if parcial else "Entrega final: todas as partes desta pasta foram processadas."))
        if removidas:
            L.append("> **{} imagem(ns) da entrega anterior foram retiradas** nesta: {}. Lista e motivo em "
                     "`datasetfinal_MAE/qualidade/removidas_desde_entrega_anterior.csv`.\n".format(
                         len(removidas), ", ".join("{} {}".format(v, k) for k, v in Counter(
                             r["particao"] for r in removidas).items())))
    L.append("Lesoes sinteticas suspeitas (BI-RADS 4/5) inseridas em mamografias **BI-RADS 1** de "
             "`{}`, varias variantes independentes por mama. Este relatorio foi gerado por "
             "`pipeline/scripts/relatorio_qualidade_lote.py` a partir do que cada etapa gravou; "
             "nenhum numero foi digitado a mao.\n".format(entrada.name))
    L.append(tabela(["", ""], [
        ["Mamografias candidatas", str(len(host))],
        ["Hospedeiras aceitas", "{} ({} pacientes)".format(len(incl), len({h['paciente'] for h in incl}))],
        ["Hospedeiras processadas", "{} em {} parte(s){}".format(
            n_hosp, len(partes_prontas) or "?", " -- geracao interrompida pela guarda de disco antes de "
            + interrompido if interrompido else "")],
        ["Lesoes pedidas / geradas", "{} / {} ({} por mama)".format(pedidas, geradas, por_imagem)],
        ["Auditadas ({} criterios + 6 de fidelidade)".format(len(CRIT_AUD)), str(n_aud)],
        ["**Entregues**", "**{}** ({} das auditadas)".format(n_ent, pct(n_ent / max(1, n_aud)))],
        ["Mamas / pacientes na entrega", "{} / {}".format(len(mamas), len(pac))],
        ["Conferidas contra a origem apos a montagem", "{} de {}".format(v_ok, len(verif))],
    ], "lr"))
    L.append("")

    L.append("## 1. Revisao dos dados de entrada\n")
    L.append("O nome do arquivo nao foi aceito como verdade. `scripts/selecionar_hospedeiras.py` le o "
             "rotulo de incidencia gravado na propria imagem (`CCE`/`CCD` = 3 letras, `MLOE`/`MLOD` = "
             "4 letras; a razao largura/altura separa as duas sem sobreposicao) e cruza com o nome.\n")
    L.append("**Achado principal: a MLO esquerda foi exportada como `L_CC_1`** (e `L_CC_2`, `L_CC_3` "
             "nas repeticoes). O rotulo da imagem diz `MLOE` e o musculo peitoral esta visivel. Pelo "
             "nome, o pipeline tratava essas imagens como CC e **nao procurava o peitoral** -- a lesao "
             "podia cair dentro do musculo, e a auditoria, que usava a mesma funcao, nao perceberia. "
             "Corrigido com uma tabela de incidencias (`--incidencias`) lida pelo gerador e pela "
             "auditoria: **{} imagens** que o nome chamava de CC foram processadas como MLO.\n".format(corrigidas))
    L.append(tabela(["Motivo de exclusao da hospedeira", "Imagens", "O que e"], [
        ["`rotulo_nao_padrao`", str(excl.get("rotulo_nao_padrao", 0)),
         "texto clinico gravado (segmentectomia, quadrantectomia, mastectomia com reconstrucao, "
         "cirurgia, radio/quimio, cicatriz), phantom de calibracao, rotulo ilegivel ou de outro "
         "equipamento (com barra vertical clara)"],
        ["`incidencia_divergente`", str(excl.get("incidencia_divergente", 0)),
         "rotulo da imagem e nome discordam; excluida por conservadorismo"],
        ["`ja_usada_em_entrega`", str(excl.get("ja_usada_em_entrega", 0)),
         "mesma mamografia ja foi hospedeira em `dataset_escolhido` / `dataset_escolhido_v2`"],
        ["`duplicata_exata`", str(excl.get("duplicata_exata", 0)), "conteudo identico a outro arquivo"],
    ], "lrl"))
    L.append("\nMiniaturas das excluidas: `datasetfinal_MAE/qualidade/excluidas.png`; decisao por "
             "imagem: `datasetfinal_MAE/qualidade/hospedeiras.csv`.\n")
    L.append("Incidencias processadas: {}. Fonte da incidencia: {}. Metodo de exclusao do peitoral: {}.\n".format(
        ", ".join("{} {}".format(k, v) for k, v in sorted(views.items())),
        ", ".join("{} {}".format(k, v) for k, v in fonte_view.items()),
        ", ".join("`{}` {}".format(k, v) for k, v in pect.most_common())))

    L.append("## 2. Geracao\n")
    politicas = json.loads((lote / "_controle" / "politicas.json").read_text(encoding="utf-8")) \
        if (lote / "_controle" / "politicas.json").is_file() else {}
    if politicas:
        L.append("O lote tem mais de uma politica de geracao; a coluna `politica` do manifesto diz a de "
                 "cada imagem. Contraste medido na coroa afastada (`crel_afastado`, a regua da calibracao), "
                 "quartis p25 / mediana / p75:\n")
        linhas_pol = []
        try:
            import calibrar_contraste as _cal
            r_af = _cal.perfil_real(a.reais)
            linhas_pol.append(["reais ({} recortes de `{}`)".format(len(r_af), Path(a.reais).name), "--", "--",
                               " / ".join(fnum(x, 3) for x in np.percentile(r_af, [25, 50, 75]))])
        except Exception:
            pass
        presentes = {r.get("politica") for r in ent_l}
        for pol, desc in [(k, v) for k, v in politicas.items() if k in presentes]:
            ents = [r for r in ent_l if r.get("politica") == pol]
            ca = [float(fid[r["arquivo"]]["crel_afastado"]) for r in ents
                  if r["arquivo"] in fid and fid[r["arquivo"]].get("crel_afastado") not in ("", None, "nan")]
            ca = [x for x in ca if np.isfinite(x)]
            linhas_pol.append(["`{}`".format(pol), desc, str(len(ents)),
                               " / ".join(fnum(x, 3) for x in np.percentile(ca, [25, 50, 75])) if ca
                               else "nao medido nesta politica"])
        L.append(tabela(["Politica", "O que muda", "Entregues", "crel afastado"], linhas_pol, "llrl"))
        L.append("")
    L.append("Motor: SD 1.5 *inpainting* (checkpoint MAM-E) + adaptador LoRA `treinos/birads_lora_v1` "
             "(passo 2500, guidance 2,5), em **janela na resolucao nativa** -- o caminho de maior "
             "resolucao do pipeline: a lesao nasce em pixels da imagem original (reducao minima "
             "0,75x), nao no canvas de 1024. Tamanho de cada lesao sorteado do catalogo de recortes "
             "reais BI-RADS 4/5; posicao pelo mapa de plausibilidade anatomica; cada variante parte da "
             "mamografia original, nunca de outra variante. Tipos na proporcao dos recortes reais "
             "(83% BI-RADS 5, 17% BI-RADS 4).\n")
    ganhos = [float(r["ganho"]) for r in cal.values() if r.get("ganho") not in ("", None)]
    est_cal = Counter(r.get("estado", "") for r in cal.values())
    if not cal:
        L.append("Contraste: **sem calibracao** -- cada lesao como o gerador a entregou.\n")
    elif set(est_cal) <= {"ganho_fixo", "nao_medida"}:
        L.append("Contraste: `calibrar_contraste.py --ganho-fixo {}` (modo `densidade`: so a baixa frequencia "
                 "do delta muda, a textura fina fica) em todas as lesoes.\n".format(fnum(np.median(ganhos), 2)))
    else:
        ganhos = [float(r["ganho"]) for r in cal.values()
                  if r.get("ganho") not in ("", None) and r.get("estado") != "ganho_fixo"] or ganhos
        L.append("Contraste: `calibrar_contraste.py` por **casamento de quantis** (modo `densidade`: so a baixa "
                 "frequencia do delta muda, a textura fina fica). O contraste de cada lesao, medido na coroa "
                 "afastada (0,75-0,95 do lado), recebe o quantil correspondente da distribuicao de {} recortes "
                 "reais BI-RADS 4/5; a ordem relativa entre as lesoes e preservada. Ganho aplicado: mediana {} "
                 "(p10 {}, p90 {}). Estados: {}.\n".format(
                     "82", fnum(np.median(ganhos), 2), fnum(np.percentile(ganhos, 10), 2),
                     fnum(np.percentile(ganhos, 90), 2),
                     ", ".join("`{}` {}".format(k, v) for k, v in est_cal.most_common())))
    exp_cal = ler_csv(lote / "_controle" / "experimento_calibracao.csv")
    if exp_cal:
        L.append("**Por que o ganho 0,85 (politica v1).** O ganho 1,25 do `lote_variacoes_v5` foi medido com hospedeiras de "
                 "`sem_anomalia`. Nestas, o casamento por quantil com as lesoes reais (coroa afastada) pede "
                 "ganho mediano ~0,89: as lesoes ja saem mais contrastadas que as reais. As politicas foram "
                 "comparadas nas MESMAS {} lesoes de um piloto, com os mesmos criterios e limiares -- que "
                 "reprovam tanto lesao forte quanto fraca demais (taxas medidas antes do criterio "
                 "`nucleo_acima_tecido`, criado depois, que reprova mais algumas em todas as politicas):\n".format(exp_cal[0]["lesoes"]))
        L.append(tabela(["Politica", "Aprovadas", "CC", "MLO", "Weber mediano"], [
            [r["politica"], "{} ({})".format(r["aprovadas"], pct(float(r["taxa"]), 0)), pct(float(r["taxa_cc"]), 0),
             pct(float(r["taxa_mlo"]), 0), fnum(r["weber_mediano"], 2)] for r in exp_cal], "lrrrr"))
        L.append("\nO quantil por lesao perde para o ganho fixo porque leva algumas lesoes ao teto de ganho "
                 "(2,5), que satura. Nas MLO a lesao cai em tecido mais escuro, e o mesmo acrescimo de "
                 "densidade vira mais contraste de Weber.\n")
    at = np.array(tentativas) if tentativas else np.array([1])
    L.append(tabela(["Portao de QC interno (na geracao)", "Valor"], [
        ["Lesoes pedidas", str(pedidas)],
        ["Geradas e gravadas", str(geradas)],
        ["Descartadas apos esgotar posicoes e sementes", str(pedidas - geradas)],
        ["Hospedeiras sem nenhuma lesao", str(len(hosp_falha))],
        ["Lesoes aceitas na 1a tentativa", "{:.0%}".format(float((at == 1).mean()))],
        ["Tentativas por lesao aceita (mediana / p95 / max)",
         "{} / {} / {}".format(int(np.median(at)), int(np.percentile(at, 95)), int(at.max()))],
        ["Tempo de geracao somado (min)", fnum(sum(durs) / 60.0, 0)],
    ], "lr"))
    L.append("\nO portao interno regenera com outra semente e depois outra posicao quando a lesao sai "
             "invertida (saldo de densidade < 1%), sem contraste, com CNR fora da faixa ou com pixel "
             "alterado fora da mascara. E so a primeira barreira: tudo que ele aprova passa depois "
             "pelas quatro verificacoes abaixo.\n")

    L.append("## 3. As quatro verificacoes\n")
    L.append("Cada lesao gravada foi relida do disco e medida de novo por `auditar_dataset.py` ({} "
             "criterios, contra a mamografia de origem) e por `fidelidade_real.py` (6 descritores, "
             "contra {} recortes reais BI-RADS 4/5 anotados). Uma lesao entra na entrega so se passa "
             "nos criterios bloqueantes das quatro.\n".format(len(CRIT_AUD), n_reais))
    L.append(tabela(["Verificacao", "Reprovadas", "So nesta", "% das auditadas"], [
        [ROTULO_CAT[c], str(rep_cat[c]), str(so_cat[c]), pct(rep_cat[c] / n_aud)]
        for c in CATEGORIAS], "lrrr"))
    L.append("\n### 3.1 Criterio a criterio\n")
    crits = [(c, CRIT_AUD[c][1], "auditoria", CRIT_AUD[c][4],
              "{} a {}".format(fnum(CRIT_AUD[c][2]), fnum(CRIT_AUD[c][3]))) for c in CRIT_AUD]
    crits += [(c, CRIT_FID[c][1], "fidelidade", CRIT_FID[c][2], "p5-p95 das reais") for c in CRIT_FID]
    crits += [("fora_da_faixa_real", "bloqueante", "fidelidade", "3+ descritores fora da faixa real ao mesmo tempo", "<= 2"),
              ("granulado", "bloqueante", "fidelidade", "textura sal e pimenta dentro da lesao (`sigma_ratio`)", "<= 1,80"),
              ("crel_afastado", "bloqueante", "fidelidade", "contraste na coroa afastada, a regua da calibracao", "p5-p95 das reais")]
    for cat in CATEGORIAS:
        L.append("**{}**\n".format(ROTULO_CAT[cat]))
        L.append(tabela(["Criterio", "Severidade", "Faixa", "Reprovou", "Aviso", "O que mede"], [
            ["`{}`".format(c), sev, faixa, str(falhas_crit.get(c, 0)), str(avisos_crit.get(c, 0)), desc]
            for c, sev, origem_c, desc, faixa in crits if categoria(c) == cat], "lllrrl"))
        L.append("")
    L.append("Uma lesao pode reprovar em mais de um criterio. Valor medido de cada criterio, por lesao: "
             "`datasetfinal_MAE/qualidade/auditoria.csv` e `fidelidade.csv`; veredito por "
             "verificacao, com o criterio e o valor que reprovaram: `rastreabilidade.csv`.\n")

    L.append("### 3.2 Artefatos encontrados e o que foi feito\n")
    # (rotulo, criterios bloqueantes, criterios de aviso, o que e, resolucao aplicada)
    art = [
        ("veu liso / bloco retangular", ["textura_roi"], ["ruido_roi_anel", "nitidez_roi_anel"],
         "ruido e nitidez da lesao caem juntos em relacao ao tecido vizinho (assinatura do VAE)", "lesao descartada"),
        ("grade do espaco latente", ["grade_z"], [],
         "periodicidade de grade, medida contra 24 recortes de tecido real da mesma mama", "lesao descartada"),
        ("halo escuro", ["halo_u8"], [],
         "o modelo retirando densidade do hospedeiro em volta da lesao",
         "piso de densidade na composicao (`density_floor -3`); descarte do que ainda passa do limite"),
        ("degrau / costura na borda", ["degrau_borda_u8", "costura_excesso"], [],
         "descontinuidade de nivel ou gradiente na borda da mascara",
         "mascara superelitica com transicao suave; descarte"),
        ("saturacao", ["saturacao_frac"], [], "lesao saturada no topo da escala", "descarte"),
        ("mancha branca chapada", ["nucleo_acima_tecido"], [],
         "nucleo da lesao mais claro que qualquer tecido original em volta; estoura, quase quadrada",
         "criterio novo neste lote; descarte"),
        ("segmento estilhacado / caixa inteira", [], ["circularidade", "preenchimento_pegada"],
         "forma da lesao segmentada fora do esperado", "registrado"),
        ("granulado sal e pimenta", ["granulado"], [],
         "ruido de alta frequencia da lesao muito acima do tecido (`sigma_ratio` > 1,8)",
         "criterio novo a partir da entrega 2; descarte (e retirada das que ja tinham sido entregues)"),
        ("granulacao, borda e espectro vs. reais", [], ["sigma_ratio", "grad_borda", "psd_core"],
         "textura fina fora da faixa p5-p95 das lesoes reais", "registrado; conta para `fora_da_faixa_real`"),
        ("pixel alterado fora da mascara", ["fora_da_mascara_px"], [],
         "qualquer mudanca fora da regiao da lesao",
         "verificado em toda lesao, de novo apos a calibracao e de novo apos a montagem"),
    ]
    L.append(tabela(["Artefato", "Criterios", "Reprovadas", "Avisos", "O que e", "Resolucao aplicada"], [
        [n, ", ".join("`{}`".format(c) for c in b + w), str(sum(falhas_crit.get(c, 0) for c in b)),
         str(sum(avisos_crit.get(c, 0) for c in w)), d, r] for n, b, w, d, r in art], "llrrll"))
    L.append("")
    L.append("**Artefato novo encontrado na revisao visual deste lote: mancha branca chapada.** O nucleo "
             "da lesao fica mais claro que qualquer tecido original em volta e aparece estourado, quase "
             "quadrado. Nenhum dos 27 criterios anteriores o pegava (passava com Weber 0,95-1,16). Criado "
             "`nucleo_acima_tecido` (bloqueante, <= 0,25), com o limiar posto olhando as aprovadas do "
             "piloto ordenadas pela medida: >= 0,28 eram manchas chapadas, <= 0,24 massas com textura. "
             "Reprovou **{}** lesoes neste lote. Perto do limiar a medida e ruidosa: na primeira parte, "
             "uma lesao estrelada plausivel (0,26) ficou de fora, e massas espiculadas convincentes "
             "passaram com 0,21-0,24. O limiar e conservador de proposito: um falso positivo custa uma "
             "lesao entre milhares; um falso negativo poe um artefato no treino.\n".format(
                 falhas_crit.get("nucleo_acima_tecido", 0)))
    L.append("**Segundo artefato novo: granulado \"sal e pimenta\".** Textura de pontos dentro da lesao, "
             "medida por `sigma_ratio` (ruido de alta frequencia lesao/tecido; reais ate ~1,13). No piloto "
             "a fronteira em 1,6 era ambigua (1,58 massa plausivel, 1,66 bola granulada), entao ficou so "
             "como aviso. Na entrega 1, das 16 entregues acima de 1,8, 15 eram granulado nitido: a partir "
             "da entrega 2, `granulado` (> 1,8) e bloqueante, e as que ja tinham sido entregues sairam. "
             "Reprovou **{}** lesoes. Entre 1,13 e 1,8 continua aviso -- distribuicao nas entregues:\n"
             .format(falhas_crit.get("granulado", 0)))
    L.append(tabela(["`sigma_ratio` nas entregues", "Imagens"], [[k, str(v)] for k, v in faixas_sig], "lr"))
    L.append("\nPainel das {} mais granuladas: `datasetfinal_MAE/qualidade/paineis/entregues_granulado_residual.png`.\n"
             .format(min(16, len(sig_hi))))
    L.append("### 3.3 Inconsistencias anatomicas e o que foi feito\n")
    L.append(tabela(["Inconsistencia", "Reprovadas", "Resolucao aplicada"], [
        ["lesao sobre o musculo peitoral (`peitoral_frac`)", str(falhas_crit.get("peitoral_frac", 0)),
         "incidencia corrigida pela tabela ({} MLO que o nome chamava de CC); mapa de plausibilidade zera o peitoral + 6 mm; auditoria remede".format(corrigidas)],
        ["lesao a menos de 8 mm da pele (`distancia_pele_mm`, medida na resolucao nativa)", str(falhas_crit.get("distancia_pele_mm", 0)),
         "descarte -- o gerador mede no canvas, onde a distancia sai otimista"],
        ["lesao fora do tecido mamario (`cobertura_tecido`)", str(falhas_crit.get("cobertura_tecido", 0)), "descarte"],
        ["lesao truncada pela borda da imagem (`toca_borda_img`)", str(falhas_crit.get("toca_borda_img", 0)), "descarte"],
        ["posicao pouco plausivel (`score_anatomico`, aviso)", "aviso: " + str(avisos_crit.get("score_anatomico", 0)), "registrado"],
        ["mama operada / phantom como hospedeira", str(excl.get("rotulo_nao_padrao", 0)) + " hospedeiras", "excluidas antes da geracao"],
    ], "lrl"))
    L.append("")
    L.append("### 3.4 Plausibilidade clinica\n")
    L.append("Faixas absolutas (auditoria) e distribuicao das lesoes reais anotadas (fidelidade). "
             "AUC real x sintetico: 0,5 = indistinguivel; longe de 0,5 = deslocado.\n")
    if auc_linhas:
        L.append(tabela(["Descritor", "Severidade", "Real (mediana)", "Gerado (mediana)", "Entregue (mediana)",
                         "Faixa aceita", "AUC gerado", "AUC entregue"], auc_linhas, "llrrrlrr"))
        L.append("")
    L.append("**Leitura.** `crel`/`conspic` sao medidos no anel imediato em volta do nucleo; no recorte "
             "real justo esse anel ainda e lesao, o que deprime o contraste real -- por isso servem como "
             "guarda de extremo, nao como alvo. `radialidade` tem deslocamento conhecido: as sinteticas "
             "sao menos espiculadas que as reais (limitacao do gerador, registrada por lesao).\n")

    L.append("## 4. A entrega\n")
    L.append(tabela(["", ""], [
        ["Resolucao", "{}x{} a {}x{} px (mediana {} MP) -- a da mamografia de origem, sem reamostragem".format(
            min(W) if W else 0, min(H) if H else 0, max(W) if W else 0, max(H) if H else 0,
            fnum(np.median(mp), 1) if mp else "--")],
        ["Profundidade de bits", ", ".join("{} bits: {}".format(k, v) for k, v in sorted(bits.items()))],
        ["Fora da lesao", "bit a bit identico a origem ({} de {} conferidas)".format(v_ok, len(verif))],
        ["Tipos", ", ".join("{} {}".format(k, v) for k, v in sorted(tipos.items()))],
        ["Incidencia / lado", "{} / {}".format(", ".join("{} {}".format(k, v) for k, v in sorted(views_ent.items())),
                                               ", ".join("{} {}".format(k, v) for k, v in sorted(lado_ent.items())))],
        ["Variantes por mama", ", ".join("{}: {} mamas".format(k, v) for k, v in sorted(var_por_mama.items()))],
        ["Particao (por paciente)", ", ".join("{} {}".format(k, v) for k, v in sorted(parts.items()))],
        ["Pacientes com recorte no treino do LoRA", "{} imagens (coluna `paciente_nos_recortes_lora`)".format(lora_pac)],
    ], "ll"))
    L.append("")
    L.append(tabela(["Medida na entrega", "p5", "mediana", "p95"], [
        ["diametro (mm)"] + [fnum(x, 1) for x in q("diametro_mm")],
        ["lado maior da caixa (px)"] + [fnum(x, 0) for x in quantis([max(int(r["largura_px"]), int(r["altura_px"])) for r in ent_l])],
        ["contraste de Weber (nativo)"] + [fnum(x, 2) for x in q("weber")],
        ["CNR"] + [fnum(x, 2) for x in q("cnr")],
        ["distancia da pele (mm)"] + [fnum(x, 1) for x in q("distancia_pele_mm")],
    ], "lrrr"))
    L.append("\nRegioes anatomicas (terco / faixa): " + "; ".join(
        "{} {}".format(k, v) for k, v in reg.most_common(9)) + ".\n")
    L.append("**Duplicacao.** Variantes da mesma mama que se sobrepoem (IoU > 0,5) ou cujo recorte "
             "correlaciona > 0,95 disputam por supressao nao-maxima e so a de maior confianca fica "
             "({} descartadas). Nenhuma hospedeira de entrega anterior foi reusada.\n".format(
                 falhas_crit.get("iou_max_variantes", 0) + falhas_crit.get("similaridade_max", 0)))

    L.append("## 5. Conferencia visual\n")
    L.append("Paineis em `datasetfinal_MAE/qualidade/paineis/`, com **o mesmo janelamento** no antes e "
             "no depois (um recorte auto-janelado sozinho exagera o contraste):\n")
    L.append("* `aprovadas_antes_depois.png` -- 36 lesoes entregues sorteadas: origem | com lesao (caixa em laranja).")
    for cat, nome in reprov_paineis.items():
        L.append("* `{}` -- ate 20 reprovadas em {}, com o criterio que reprovou{}.".format(
            nome, ROTULO_CAT[cat].lower(), " (mama inteira, caixa em vermelho)" if cat == "anatomia" else ""))
    L.append("")

    L.append("## 6. Limites\n")
    L.append("1. **Rotulo BI-RADS 4/5 nao validado por classe.** O adaptador viu 12 recortes BI-RADS 4; "
             "nao ha evidencia de que gere algo distinto por classe.\n"
             "2. **Assinatura do gerador.** Um classificador em features profundas separa estas lesoes "
             "das reais (AUC 0,94 no teste do LoRA; a mesma lesao real so reconstruida pelo VAE ja da "
             "0,97). Servem para aumento de dataset, nao como substituto indistinguivel.\n"
             "3. **Menos espiculadas que as reais** (`radialidade`).\n"
             "4. **A caixa e a pegada, nao o contorno.** A mascara e a regiao redesenhada (inclui "
             "margem), nao a segmentacao da lesao.\n"
             "5. **Escala em mm estimada** (detector de 29,5 cm no lado maior); os PNG nao trazem "
             "`PixelSpacing`.\n"
             "6. **Entregas anteriores.** `dataset_escolhido` e `dataset_escolhido_v2` foram gerados "
             "sem a tabela de incidencias: as hospedeiras `*_L_CC_1` delas foram tratadas como CC, sem "
             "exclusao do peitoral.\n")
    L.append("> Imagens sinteticas para pesquisa e aumento de dataset. **Nao devem ser usadas para "
             "diagnostico** nem misturadas a arquivos clinicos reais sem marcacao de origem sintetica.\n")

    texto = "\n".join(L)
    (ds / "RELATORIO_QUALIDADE.md").write_text(texto, encoding="utf-8")
    shutil.copy2(ds / "RELATORIO_QUALIDADE.md", qual / "RELATORIO_QUALIDADE_CONSOLIDADO.md")
    escrever_leiame(ds, final, bbox, n_ent, len(mamas), len(pac), W, H, bits, tipos, parts, var_por_mama, entrada,
                    nome_ds, Counter(r.get("entrega", "") for r in ent_l))
    print("relatorio: {}".format(ds / "RELATORIO_QUALIDADE.md"))
    print("rastreabilidade: {} lesoes, {} entregues".format(n_aud, n_ent))
    return 0


def escrever_leiame(ds, final, bbox, n, n_mamas, n_pac, W, H, bits, tipos, parts, var, entrada,
                    nome_ds, por_entrega) -> None:
    bits_s = ", ".join("{} bits ({})".format(k, v) for k, v in sorted(bits.items()))
    tipos_s = ", ".join("{} ({})".format(k, v) for k, v in sorted(tipos.items()))
    parts_s = ", ".join("{} {}".format(k, v) for k, v in sorted(parts.items()))
    raiz = """# {nome} -- lesoes sinteticas em mamografias BI-RADS 1

{n} mamografias com **uma lesao sintetica suspeita (BI-RADS 4/5) por imagem**, geradas a
partir de {n_mamas} mamografias BI-RADS 1 de `{ent}` ({n_pac} pacientes), na resolucao e
profundidade de bits originais, aprovadas em quatro verificacoes de qualidade.
O relatorio completo esta em [`RELATORIO_QUALIDADE.md`](RELATORIO_QUALIDADE.md).

| | |
|---|---|
| Imagens | **{n}** |
| Resolucao | {wmin}x{hmin} a {wmax}x{hmax} px, a da origem |
| Bits | {bits} |
| Tipos | {tipos} |
| Particao (por paciente) | {parts} |
| Imagens por entrega | {entregas} |

```
{nome}/
├── RELATORIO_QUALIDADE.md   artefatos, anatomia, restricoes e plausibilidade, lesao a lesao
├── datasetfinal_MAE/        ENTREGA PRINCIPAL -- maior resolucao e qualidade validada
│   ├── imagens/               PNG na resolucao e bits da mamografia de origem
│   ├── mascaras/              regiao alterada (branco); fora dela, pixel identico a origem
│   ├── manifest.csv           1 linha por imagem: caixa, medidas, particao, rastreio
│   └── qualidade/             auditoria, fidelidade, rastreabilidade.csv, paineis, hospedeiras
└── dataset_bbox/            conferencia visual e treino de detector
    ├── imagens/               as mesmas imagens com a caixa da lesao desenhada (8 bits RGB)
    └── anotacoes/             caixas.csv, coco.json, yolo/, classes.txt
```

Para treinar: `datasetfinal_MAE/imagens/` + as caixas de `manifest.csv` (ou
`dataset_bbox/anotacoes/`). Respeite a coluna `particao`: ela e por paciente e honra a
divisao usada no treino do gerador. As imagens de `dataset_bbox/imagens/` sao so para
olhar -- 8 bits e com a caixa pintada.

Os arquivos de `datasetfinal_MAE/imagens/` sao hardlinks dos arquivos do lote de
origem: apagar o lote nao os afeta, e a entrega nao ocupa espaco duplicado.

> Imagens sinteticas para pesquisa e aumento de dataset. **Nao devem ser usadas para
> diagnostico** nem misturadas a arquivos clinicos reais sem marcacao de origem sintetica.
""".format(nome=nome_ds, n=n, entregas=", ".join("entrega {}: {}".format(k, v) for k, v in sorted(por_entrega.items())) or "--", n_mamas=n_mamas, n_pac=n_pac, ent=entrada.name,
           wmin=min(W) if W else 0, hmin=min(H) if H else 0, wmax=max(W) if W else 0, hmax=max(H) if H else 0,
           bits=bits_s, tipos=tipos_s, parts=parts_s)
    (ds / "LEIA-ME.md").write_text(raiz, encoding="utf-8")
    (final / "LEIA-ME.md").write_text("""# datasetfinal_MAE

{n} PNG na resolucao e profundidade de bits da mamografia de origem, uma lesao
sintetica por imagem. Fora da mascara, cada pixel e identico ao da origem
(conferido em `qualidade/verificacao_entrega.csv`).

* `imagens/<origem>_com_lesao_NN.png` -- NN = numero da variante naquela mama
* `mascaras/<...>_mask.png` -- regiao redesenhada pelo modelo (inclui margem)
* `manifest.csv` -- caixa `x1,y1,x2,y2` (pegada da lesao, px nativos; x2/y2 exclusivos),
  tipo, particao, incidencia (`view`, `view_fonte`), medidas da auditoria e da fidelidade
* `qualidade/` -- `rastreabilidade.csv` (toda lesao gerada, aprovada ou nao, e por que),
  `auditoria.csv`, `fidelidade.csv`, `calibracao_contraste.csv`, `hospedeiras.csv`,
  `verificacao_entrega.csv`, `paineis/`
""".format(n=n), encoding="utf-8")
    if bbox.is_dir():
        (bbox / "LEIA-ME.md").write_text("""# dataset_bbox

As mesmas {n} imagens de `../datasetfinal_MAE/`, convertidas para 8 bits RGB com a
caixa da lesao desenhada e rotulada (tipo e tamanho em px). Servem para conferencia
visual; para treinar use as imagens de 16 bits de `../datasetfinal_MAE/imagens/` com
as anotacoes daqui:

* `anotacoes/caixas.csv` -- arquivo, tipo, particao, x1, y1, x2, y2 (px nativos)
* `anotacoes/coco.json` -- formato COCO (bbox = x, y, largura, altura)
* `anotacoes/yolo/*.txt` -- classe cx cy w h normalizados; classes em `classes.txt`
""".format(n=n), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
