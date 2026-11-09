#!/usr/bin/env python
"""Quanto detalhe fino da lesao sobrevive ao VAE em cada fator de contexto da janela nativa.

Justifica ``native_window.context_factor``. Para cada recorte de treino, monta a
amostra exatamente como o treino (colagem em hospedeiro de treino), passa a
janela de 512 pelo VAE congelado do checkpoint, amplia de volta a resolucao
nativa e compara a pegada da lesao com a composicao nativa original.

Contexto maior da ao modelo mais tecido para condicionar a geracao, mas reduz
mais a lesao. Este script mede o custo dessa reducao.

    .venv/bin/python scripts/analise_escala_vae.py --dados treinos/dados_birads
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2 as cv                                           # noqa: E402
import numpy as np                                         # noqa: E402
import torch                                               # noqa: E402

from mamelesion.birads_data import load_json, read_crop, LesionCrop, parse_ids   # noqa: E402
from mamelesion.config import Config                       # noqa: E402
from mamelesion.inpainting import resolve_model_path       # noqa: E402
from mamelesion.paste import PasteConfig, make_sample, prepare_host              # noqa: E402


def fino(a):
    return cv.GaussianBlur(a, (0, 0), 1.0) - cv.GaussianBlur(a, (0, 0), 3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="treinos/dados_birads")
    ap.add_argument("--fatores", default="1.25,1.5,2.0,2.5,3.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-scale", type=float, default=0.5, help="teto de reducao (0 desliga)")
    args = ap.parse_args()

    cfg = Config.load()
    split = load_json(Path(args.dados) / "split.json")
    hosts_dir = Path(split["hospedeiros_pasta"])
    hosts = [prepare_host(hosts_dir / n, cfg) for n in split["hospedeiros"]["train"]]
    crops = []
    for n in split["recortes"]["train"]:
        p = Path(split["caminho_de"][n])
        pat, st = parse_ids(n)
        crops.append(LesionCrop(n, p, split["classe_de"][n], pat, st, read_crop(p)))

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(resolve_model_path(cfg.paths.model_id), subfolder="vae",
                                        torch_dtype=torch.float32).to("cuda").eval()

    resultados = {}
    for fator in [float(x) for x in args.fatores.split(",")]:
        pc = PasteConfig(context_factor=fator, min_scale=args.min_scale, scale_jitter=(1.0, 1.0),
                         flip_horizontal=False, flip_vertical=False)
        rng = np.random.default_rng(args.seed)
        corrs, rets, escalas = [], [], []
        for i, crop in enumerate(crops):
            s = make_sample(crop, hosts[i % len(hosts)], rng, pc, keep_native=True)
            x = torch.from_numpy(s.image_u8.astype(np.float32) / 127.5 - 1)[None, None].repeat(1, 3, 1, 1).cuda()
            with torch.no_grad():
                rec = vae.decode(vae.encode(x).latent_dist.mode()).sample.clamp(-1, 1)[0].mean(0).cpu().numpy()
            rec01 = (rec + 1) / 2
            side = s.window.side
            if side != 512:
                rec01 = cv.resize(rec01, (side, side), interpolation=cv.INTER_CUBIC)
            fx0, fy0, fx1, fy1 = s.footprint
            ox, oy = fx0 - s.window.x0, fy0 - s.window.y0
            ref = s.native_composite[oy:oy + fy1 - fy0, ox:ox + fx1 - fx0]
            got = rec01[oy:oy + fy1 - fy0, ox:ox + fx1 - fx0]
            fr, fg = fino(ref), fino(got)
            corrs.append(float(np.corrcoef(fr.ravel(), fg.ravel())[0, 1]))
            rets.append(float(np.std(fg) / max(np.std(fr), 1e-8)))
            escalas.append(s.window.scale)
        resultados[fator] = {"correlacao_detalhe_mediana": round(float(np.median(corrs)), 3),
                             "correlacao_detalhe_p10": round(float(np.percentile(corrs, 10)), 3),
                             "retencao_detalhe_mediana": round(float(np.median(rets)), 3),
                             "escala_mediana": round(float(np.median(escalas)), 3),
                             "fracao_em_escala_nativa": round(float(np.mean(np.isclose(escalas, 1.0))), 3)}
        print("fator {:4.2f} | escala mediana {:.2f} | nativas {:.0%} | correlacao detalhe {:.3f} (p10 {:.3f}) | retencao {:.3f}".format(
            fator, resultados[fator]["escala_mediana"], resultados[fator]["fracao_em_escala_nativa"],
            resultados[fator]["correlacao_detalhe_mediana"], resultados[fator]["correlacao_detalhe_p10"],
            resultados[fator]["retencao_detalhe_mediana"]))
    out = Path(args.dados) / "analise_escala_vae_min_scale_{}.json".format(args.min_scale)
    out.write_text(json.dumps(resultados, indent=2), encoding="utf-8")
    print("gravado:", out)


if __name__ == "__main__":
    main()
