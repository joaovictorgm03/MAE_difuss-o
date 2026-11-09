#!/usr/bin/env python
"""Desenha a caixa (bbox) de cada lesao sobre as imagens de um lote, numa pasta ``bbox/``.

Para conferencia visual: as imagens entregues em ``original/`` continuam
intocadas. Cada imagem de ``bbox/`` e a mesma mamografia com resolucao original,
convertida para 8 bits RGB, com um retangulo e o tipo da lesao.

Qual caixa e desenhada:

* modo nativo (``--lora``): a **pegada da lesao** -- o tamanho real sorteado do
  catalogo --, que e onde a lesao esta. Com ``--mascara`` desenha tambem, em
  tracejado, a area da mascara (a pegada + a margem que o modelo redesenha);
* modo canvas: a area da mascara, a unica caixa que esse modo registra.

Grava tambem ``bbox/caixas.csv`` (uma linha por lesao, coordenadas em pixels da
imagem original: x1, y1 inclusivos; x2, y2 exclusivos).

    .venv/bin/python scripts/desenhar_bbox.py ../mamas_com_lesao/lote_birads_lora
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image

CORES = {"birads5": (255, 60, 60), "birads4": (255, 170, 0), "mass": (255, 60, 60),
         "nodule": (60, 200, 255), "density": (120, 255, 120)}


def to_rgb8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        ref = float(img.max()) or 1.0
        img = np.clip(img.astype(np.float32) / ref * 255.0, 0, 255).astype(np.uint8)
    return cv.cvtColor(img, cv.COLOR_GRAY2RGB)


def dashed_rect(img, p0, p1, color, thick, dash):
    x0, y0 = p0
    x1, y1 = p1
    for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        for s in range(0, length, 2 * dash):
            t0, t1 = s / max(length, 1), min(s + dash, length) / max(length, 1)
            q0 = (int(a[0] + (b[0] - a[0]) * t0), int(a[1] + (b[1] - a[1]) * t0))
            q1 = (int(a[0] + (b[0] - a[0]) * t1), int(a[1] + (b[1] - a[1]) * t1))
            cv.line(img, q0, q1, color, thick, cv.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lote", help="pasta do lote (contem metadata/ e original/)")
    ap.add_argument("--mascara", action="store_true", help="desenha tambem a area da mascara (tracejado)")
    args = ap.parse_args()

    lote = Path(args.lote)
    meta_dir = lote / "metadata"
    if not meta_dir.is_dir():
        print("erro: {} nao tem metadata/".format(lote), file=sys.stderr)
        return 2
    out_dir = lote / "bbox"
    out_dir.mkdir(exist_ok=True)

    rows, n_img = [], 0
    for meta_path in sorted(meta_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lesions = meta.get("lesoes", [])
        if not lesions:
            continue
        by_file = {}
        for l in lesions:
            by_file.setdefault(l["files"]["original"], []).append(l)
        for image_path, items in by_file.items():
            img = to_rgb8(np.asarray(Image.open(image_path)))
            h, w = img.shape[:2]
            thick = max(3, int(round(0.0025 * max(h, w))))
            font = max(1.0, 0.0006 * max(h, w))
            for l in items:
                g = l["box_original"]
                color = CORES.get(l["lesion_type"], (255, 255, 0))
                if "pegada_lesao" in g:
                    x1, y1, x2, y2 = g["pegada_lesao"]
                    origem = "pegada_lesao"
                    if args.mascara:
                        dashed_rect(img, (g["x1"], g["y1"]), (g["x2"] - 1, g["y2"] - 1), color, max(2, thick // 2),
                                    dash=4 * thick)
                else:
                    x1, y1, x2, y2 = g["x1"], g["y1"], g["x2"], g["y2"]
                    origem = "mascara"
                cv.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), color, thick, cv.LINE_AA)
                label = "{} {}x{} px".format(l["lesion_type"], x2 - x1, y2 - y1)
                (tw, th), base = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, font, max(2, thick // 2))
                ty = y1 - thick - 6 if y1 - thick - 6 - th > 0 else y2 + th + thick + 6
                tx = min(max(0, x1), w - tw - 2)
                cv.rectangle(img, (tx - 2, ty - th - 4), (tx + tw + 2, ty + base), (0, 0, 0), -1)
                cv.putText(img, label, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, font, color, max(2, thick // 2), cv.LINE_AA)
                rows.append({"arquivo": Path(image_path).name, "tipo_lesao": l["lesion_type"], "caixa": origem,
                             "x1": x1, "y1": y1, "x2": x2, "y2": y2, "largura_px": x2 - x1, "altura_px": y2 - y1,
                             "largura_img": w, "altura_img": h})
            Image.fromarray(img).save(out_dir / (Path(image_path).stem + "_bbox.png"), optimize=True)
            n_img += 1

    with open(out_dir / "caixas.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["arquivo"])
        writer.writeheader()
        writer.writerows(rows)
    print("{} imagem(ns), {} caixa(s) -> {}".format(n_img, len(rows), out_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
