#!/usr/bin/env python
"""Quanto a colagem das amostras de treino deixa borda visivel, e quanto contraste preserva.

Justifica ``paste.PasteConfig.alpha_shape``, ``level_match`` e
``native_window.paste_feather_frac``. Sem GPU.

``excesso_borda``
    gradiente medio (imagem suavizada) sobre as arestas do retangulo do recorte,
    dividido pelo gradiente num anel de tecido logo fora dele; a mesma razao no
    hospedeiro sem colagem e o denominador. 1,0 = a colagem nao criou aresta; o
    artefato visto nas amostras (bloco retangular) aparece como valores > 1.
``retencao_contraste``
    (nucleo - anel externo) na janela composta dividido pelo mesmo contraste no
    recorte original. 1,0 = a lesao chega ao treino com o contraste real.

    .venv/bin/python scripts/analise_colagem.py --dados treinos/dados_birads
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2 as cv                                                     # noqa: E402
import numpy as np                                                   # noqa: E402

from mamelesion.birads_data import LesionCrop, load_json, parse_ids, read_crop   # noqa: E402
from mamelesion.config import Config                                 # noqa: E402
from mamelesion.paste import PasteConfig, make_sample, prepare_host  # noqa: E402

CONFIGS = {
    "retangulo_mediana_0.08": dict(alpha_shape="retangulo", level_match="mediana", paste_feather_frac=0.08),
    "superelipse_plano_0.08": dict(alpha_shape="superelipse", level_match="plano", paste_feather_frac=0.08),
    "superelipse_plano_0.15": dict(alpha_shape="superelipse", level_match="plano", paste_feather_frac=0.15),
    "superelipse_plano_0.25": dict(alpha_shape="superelipse", level_match="plano", paste_feather_frac=0.25),
}


def grad(img):
    b = cv.GaussianBlur(img, (0, 0), 2.0)
    return np.hypot(cv.Sobel(b, cv.CV_32F, 1, 0), cv.Sobel(b, cv.CV_32F, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="treinos/dados_birads")
    ap.add_argument("--amostras", type=int, default=150)
    args = ap.parse_args()
    cfg = Config.load()
    split = load_json(Path(args.dados) / "split.json")
    hosts = [prepare_host(Path(split["hospedeiros_pasta"]) / n, cfg) for n in split["hospedeiros"]["train"]]
    crops = []
    for n in split["recortes"]["train"]:
        p = Path(split["caminho_de"][n])
        crops.append(LesionCrop(n, p, split["classe_de"][n], *parse_ids(n), read_crop(p)))
    base = PasteConfig.from_native_window(cfg.native_window)

    out = {}
    for name, kw in CONFIGS.items():
        pc = replace(base, **kw)
        excess, retention = [], []
        for i in range(args.amostras):
            rng = np.random.default_rng([11, i])           # mesmas posicoes em todas as configs
            crop = crops[i % len(crops)]
            host = hosts[(i * 7) % len(hosts)]
            try:
                s = make_sample(crop, host, rng, pc, keep_native=True)
            except ValueError:
                continue
            w = s.window
            N = s.native_composite
            H = host.raw[w.y0:w.y1, w.x0:w.x1].astype(np.float32) / host.ref
            fx0, fy0, fx1, fy1 = (v - o for v, o in zip(s.footprint, (w.x0, w.y0, w.x0, w.y0)))
            yy, xx = np.mgrid[0:w.side, 0:w.side]
            inside = (xx >= fx0) & (xx < fx1) & (yy >= fy0) & (yy < fy1)
            d_edge = np.minimum.reduce([np.abs(xx - fx0), np.abs(xx - fx1), np.abs(yy - fy0), np.abs(yy - fy1)])
            edge = (d_edge <= 3) & (xx >= fx0 - 3) & (xx <= fx1 + 3) & (yy >= fy0 - 3) & (yy <= fy1 + 3)
            outer = (~inside) & (xx >= fx0 - 24) & (xx < fx1 + 24) & (yy >= fy0 - 24) & (yy < fy1 + 24) & (d_edge >= 12)
            if edge.sum() < 50 or outer.sum() < 50:
                continue
            gN, gH = grad(N), grad(H)
            excess.append(float((gN[edge].mean() / gN[outer].mean()) / (gH[edge].mean() / gH[outer].mean())))
            lh, lw = fy1 - fy0, fx1 - fx0
            c = crop.image if crop.image.shape == (lh, lw) else cv.resize(crop.image, (lw, lh), interpolation=cv.INTER_AREA)
            core = (slice(int(.35 * lh), int(.65 * lh)), slice(int(.35 * lw), int(.65 * lw)))
            ring_c = np.ones((lh, lw), bool); ring_c[int(.1 * lh):int(.9 * lh), int(.1 * lw):int(.9 * lw)] = False
            real_contrast = c[core].mean() - c[ring_c].mean()
            comp = N[fy0:fy1, fx0:fx1]
            if abs(real_contrast) > 0.01:
                retention.append(float((comp[core].mean() - H[outer].mean()) / real_contrast))
        out[name] = {"n": len(excess),
                     "excesso_borda_mediana": round(float(np.median(excess)), 3),
                     "excesso_borda_p90": round(float(np.percentile(excess, 90)), 3),
                     "retencao_contraste_mediana": round(float(np.median(retention)), 3)}
        print("{:26s} n={:3d} | excesso de borda mediana {:.3f} p90 {:.3f} | retencao de contraste {:.3f}".format(
            name, out[name]["n"], out[name]["excesso_borda_mediana"], out[name]["excesso_borda_p90"],
            out[name]["retencao_contraste_mediana"]))
    (Path(args.dados) / "analise_colagem.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
