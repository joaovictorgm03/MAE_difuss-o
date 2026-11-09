#!/usr/bin/env python
"""Mede o "remendo" em volta das lesoes de um lote do modo nativo.

Compara o tecido **regenerado** (dentro da mascara, fora da pegada da lesao) com
o tecido **original** logo fora da mascara, na imagem entregue, e normaliza pela
mesma razao na imagem de entrada -- 1,00 = a margem tem a mesma textura que o
tecido original em volta. Motivado por uma observacao visual: a regiao da lesao
parecia mais lisa que o entorno.

    .venv/bin/python scripts/analise_remendo.py ../mamas_com_lesao/lote_birads_lora
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image


def hp(a, s):
    return a - cv.GaussianBlur(a, (0, 0), s)


def bp(a, s1, s2):
    return cv.GaussianBlur(a, (0, 0), s1) - cv.GaussianBlur(a, (0, 0), s2)


def grad(a):
    b = cv.GaussianBlur(a, (0, 0), 1.0)
    return np.hypot(cv.Sobel(b, cv.CV_32F, 1, 0), cv.Sobel(b, cv.CV_32F, 0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lote")
    args = ap.parse_args()
    res = {16: [], 8: []}
    for f in sorted(glob.glob(str(Path(args.lote) / "metadata/*.json"))):
        m = json.load(open(f, encoding="utf-8"))
        for l in m["lesoes"]:
            if "pegada_lesao" not in l["box_original"]:
                print("lote nao e do modo nativo", file=sys.stderr)
                return 2
            orig_full = np.asarray(Image.open(m["arquivo"]))
            out_full = np.asarray(Image.open(l["files"]["original"]))
            mask_full = np.asarray(Image.open(l["files"]["mask_original"])) > 127
            bits = 16 if orig_full.dtype == np.uint16 else 8
            ref = 65535.0 if bits == 16 else 255.0
            g = l["box_original"]
            x1, y1, x2, y2 = g["x1"], g["y1"], g["x2"], g["y2"]
            side = max(x2 - x1, y2 - y1)
            pad = max(24, int(0.35 * side))
            X0, Y0 = max(0, x1 - pad), max(0, y1 - pad)
            X1, Y1 = min(orig_full.shape[1], x2 + pad), min(orig_full.shape[0], y2 + pad)
            o = orig_full[Y0:Y1, X0:X1].astype(np.float32) / ref
            u = out_full[Y0:Y1, X0:X1].astype(np.float32) / ref
            mk = mask_full[Y0:Y1, X0:X1]
            fx0, fy0, fx1, fy1 = g["pegada_lesao"]
            foot = np.zeros_like(mk)
            foot[max(0, fy0 - Y0):fy1 - Y0, max(0, fx0 - X0):fx1 - X0] = True
            din = cv.distanceTransform(mk.astype(np.uint8), cv.DIST_L2, 5)
            dout = cv.distanceTransform((~mk).astype(np.uint8), cv.DIST_L2, 5)
            A = mk & ~foot
            B = (~mk) & (dout <= max(12, int(0.10 * side)))
            edge = ((din <= 2) & mk) | ((dout <= 2) & ~mk)
            if A.sum() < 200 or B.sum() < 200:
                continue

            def ratio(fn):
                return (fn(u)[A].std() / fn(u)[B].std()) / (fn(o)[A].std() / fn(o)[B].std())

            ref255 = 255.0 / ref
            dlow = cv.GaussianBlur((u - o) / ref255 * ref255 * 255.0, (0, 0), 6)
            ccx, ccy = (fx0 + fx1) / 2 - X0, (fy0 + fy1) / 2 - Y0
            yy, xx = np.mgrid[0:u.shape[0], 0:u.shape[1]]
            rr = np.maximum(np.abs(xx - ccx) / ((fx1 - fx0) / 2), np.abs(yy - ccy) / ((fy1 - fy0) / 2))
            periphery = (rr >= 0.5) & (rr < 1.0)
            res[bits].append({
                "periferia escurecida >= 8 niveis": float((dlow[periphery] <= -8).mean()),
                "ruido (hp 1,5 px)": ratio(lambda a: hp(a, 1.5)),
                "textura fina (bp 1-3 px)": ratio(lambda a: bp(a, 1, 3)),
                "contraste local (bp 3-12 px)": ratio(lambda a: bp(a, 3, 12)),
                "nivel medio (desvios)": float((u[A].mean() - o[A].mean()) / (o[B].std() + 1e-6)),
                "degrau na borda (gradiente)": float(grad(u)[edge].mean() / (grad(o)[edge].mean() + 1e-9)),
            })
    for bits, rows in res.items():
        if not rows:
            continue
        print("== %d bits (n=%d) -- razao gerado/original, 1,00 = igual ao original" % (bits, len(rows)))
        for k in rows[0]:
            v = np.array([r[k] for r in rows])
            print("  %-30s mediana %.2f  [p10 %.2f  p90 %.2f]" % (k, np.median(v), *np.percentile(v, [10, 90])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
