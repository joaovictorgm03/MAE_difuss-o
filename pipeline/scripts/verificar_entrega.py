#!/usr/bin/env python
"""Confere cada arquivo ENTREGUE contra a mamografia de origem.

``verify_dataset.py`` confere o lote gerado; este confere a pasta de entrega
depois da montagem (``datasetfinal_MAE/``), para pegar qualquer coisa que a
montagem, a calibracao ou a juncao das partes tenha estragado:

* mesmas dimensoes e mesmo ``dtype`` da origem (resolucao e bits nativos);
* zero pixels alterados fora da mascara, e algum alterado dentro;
* caixa do manifesto dentro da imagem e dentro da mascara;
* nenhum nome repetido e nenhuma imagem sem linha no manifesto (ou o contrario).

    .venv/bin/python scripts/verificar_entrega.py ../dataset_escolhido_birrad1/datasetfinal_MAE \\
        --entrada ../birads_1

Grava ``qualidade/verificacao_entrega.csv``. Codigo de saida 0 se tudo passou.
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_dataset import read_gray, verify_pair  # noqa: E402


def conferir(args) -> Dict[str, object]:
    linha, pasta, entrada = args
    img = pasta / "imagens" / linha["arquivo"]
    mask = pasta / "mascaras" / linha["mascara"]
    src = entrada / linha["mamografia_origem"]
    problemas: List[str] = []
    faltam = [str(p.name) for p in (img, mask, src) if not p.is_file()]
    if faltam:
        return {"arquivo": linha["arquivo"], "ok": 0, "problemas": "ausente: " + ", ".join(faltam)}
    ok, problemas = verify_pair(src, img, mask)
    m = read_gray(mask) > 127
    h, w = m.shape
    x1, y1, x2, y2 = (int(linha[k]) for k in ("x1", "y1", "x2", "y2"))
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        problemas.append("caixa fora da imagem")
    elif m[y1:y2, x1:x2].mean() < 0.95:
        problemas.append("caixa sai da mascara ({:.0%} dentro)".format(float(m[y1:y2, x1:x2].mean())))
    if int(linha["largura_img"]) != w or int(linha["altura_img"]) != h:
        problemas.append("dimensoes do manifesto divergem")
    return {"arquivo": linha["arquivo"], "ok": int(not problemas), "problemas": "; ".join(problemas)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta", help="datasetfinal_MAE/")
    ap.add_argument("--entrada", required=True, help="pasta das mamografias de origem")
    ap.add_argument("--processos", type=int, default=8)
    a = ap.parse_args(argv)
    pasta, entrada = Path(a.pasta).resolve(), Path(a.entrada).resolve()
    with open(pasta / "manifest.csv", newline="", encoding="utf-8") as fh:
        linhas = list(csv.DictReader(fh))

    nomes = [l["arquivo"] for l in linhas]
    extras: List[str] = []
    if len(set(nomes)) != len(nomes):
        extras.append("nomes repetidos no manifesto")
    em_disco = {p.name for p in (pasta / "imagens").glob("*.png")}
    if em_disco != set(nomes):
        extras.append("{} imagem(ns) sem linha no manifesto, {} linha(s) sem imagem".format(
            len(em_disco - set(nomes)), len(set(nomes) - em_disco)))

    with ProcessPoolExecutor(a.processos) as ex:
        res = list(ex.map(conferir, [(l, pasta, entrada) for l in linhas], chunksize=4))
    (pasta / "qualidade").mkdir(exist_ok=True)
    with open(pasta / "qualidade" / "verificacao_entrega.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["arquivo", "ok", "problemas"])
        wr.writeheader()
        wr.writerows(res)
    ruins = [r for r in res if not r["ok"]]
    for r in ruins[:20]:
        print("FALHA {}: {}".format(r["arquivo"], r["problemas"]))
    for e in extras:
        print("FALHA: " + e)
    print("{} de {} arquivos entregues conferem com a origem (dimensoes, bits, zero pixel fora da mascara)"
          .format(len(res) - len(ruins), len(res)))
    return 0 if not ruins and not extras else 1


if __name__ == "__main__":
    raise SystemExit(main())
