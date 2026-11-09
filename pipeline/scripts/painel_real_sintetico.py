#!/usr/bin/env python
"""Painel visual: lesoes reais anotadas x lesoes sinteticas, no mesmo enquadramento.

A comparacao visual so e honesta se os dois lados forem recortados e exibidos do
mesmo jeito. Aqui:

* o recorte e a janela em que a lesao ocupa 55% do lado, dos dois lados
  (``calibrar_contraste.janela_lesao``);
* o janelamento de exibicao e o mesmo -- percentis 0,5 e 99,5 do proprio
  recorte, que e como um recorte real e visto quando aberto sozinho.

Esse segundo ponto e o que costuma enganar a olho: um recorte real aberto
sozinho e auto-janelado e a lesao salta; a mesma lesao dentro da mamografia
inteira parece sutil. Sem igualar o janelamento, a sintetica sempre parece ter
menos contraste que a real, independentemente do que os numeros digam.

    .venv/bin/python scripts/painel_real_sintetico.py ../mamas_com_lesao/lote_variacoes_v2 \\
        --reais /home/joao/medgan2/dados/birrads_5 --saida painel.png

Grava um painel com duas faixas: reais em cima, sinteticas embaixo.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from calibrar_contraste import janela_lesao, ler_lesoes  # noqa: E402

LADO = 224          # lado de cada quadro no painel
MARGEM = 4


def janelar(x: np.ndarray, lo_p: float = 0.5, hi_p: float = 99.5) -> np.ndarray:
    """Mesmo janelamento de exibicao dos dois lados: percentis do proprio recorte."""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [lo_p, hi_p])
    hi = max(hi, lo + 1.0)
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def quadro(x: np.ndarray, rotulo: str, cor) -> np.ndarray:
    q = cv.cvtColor(cv.resize(janelar(x), (LADO, LADO), interpolation=cv.INTER_AREA),
                    cv.COLOR_GRAY2BGR)
    cv.rectangle(q, (0, 0), (LADO - 1, LADO - 1), cor, 2)
    cv.rectangle(q, (0, LADO - 18), (LADO - 1, LADO - 1), (0, 0, 0), -1)
    cv.putText(q, rotulo[:26], (3, LADO - 5), cv.FONT_HERSHEY_SIMPLEX, 0.38, cor, 1, cv.LINE_AA)
    return q


def faixa(quadros: List[np.ndarray], titulo: str, largura: int) -> np.ndarray:
    cab = np.zeros((26, largura, 3), np.uint8)
    cv.putText(cab, titulo, (6, 18), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)
    linha = np.full((LADO, largura, 3), 20, np.uint8)
    for i, q in enumerate(quadros):
        x = i * (LADO + MARGEM)
        linha[:, x:x + LADO] = q
    return np.vstack([cab, linha])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lote")
    ap.add_argument("--reais", default="/home/joao/medgan2/dados/birrads_5")
    ap.add_argument("--saida", default=None, help="padrao: <lote>/fidelidade/painel_real_sintetico.png")
    ap.add_argument("--n", type=int, default=10, help="quadros por faixa")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lote = Path(args.lote).resolve()
    rng = np.random.default_rng(args.seed)

    fs = sorted(glob.glob(os.path.join(args.reais, "**", "*.png"), recursive=True))
    fs = [f for f in fs if (cv.imread(f, cv.IMREAD_UNCHANGED) is not None)]
    escolha = rng.permutation(len(fs))[:args.n]
    q_reais = []
    for i in escolha:
        im = cv.imread(fs[i], cv.IMREAD_UNCHANGED)
        q_reais.append(quadro(im, os.path.basename(fs[i])[:22], (90, 230, 90)))

    lesoes = ler_lesoes(lote)
    sel = rng.permutation(len(lesoes))[:args.n]
    q_sint = []
    for i in sel:
        les = lesoes[int(i)]
        out = cv.imread(str(lote / "original" / str(les["saida"])), cv.IMREAD_UNCHANGED)
        if out is None:
            continue
        wx0, wy0, wx1, wy1 = janela_lesao(out, les["pegada"])            # type: ignore[arg-type]
        q_sint.append(quadro(out[wy0:wy1, wx0:wx1], str(les["saida"])[:22], (60, 160, 255)))

    n = min(len(q_reais), len(q_sint))
    largura = n * (LADO + MARGEM) - MARGEM
    painel = np.vstack([
        faixa(q_reais[:n], "REAIS anotadas (BI-RADS 5) -- {}".format(args.reais), largura),
        np.full((8, largura, 3), 20, np.uint8),
        faixa(q_sint[:n], "SINTETICAS deste lote -- {}".format(lote.name), largura)])

    destino = Path(args.saida) if args.saida else lote / "fidelidade" / "painel_real_sintetico.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(destino), painel)
    print("painel com {} pares em {}".format(n, destino))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
