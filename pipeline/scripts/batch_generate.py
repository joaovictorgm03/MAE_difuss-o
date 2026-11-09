"""Geracao em lote de mamografias com lesoes sinteticas (saida 1024x1024).

Exemplos
--------
    python scripts/batch_generate.py --per-image 2
    python scripts/batch_generate.py --types mass,density --per-image 3 --seed 7
    python scripts/batch_generate.py --set diffusion.guidance_scale=6 --no-qc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mamelesion import Config, list_images, prepare_case, save_result, synthesize
from mamelesion.generator import random_box
from mamelesion.inpainting import LesionInpainter
from mamelesion.quality import evaluate, passes_qc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="caminho do config.yaml")
    p.add_argument("--input-dir", default=None, help="pasta com mamografias saudaveis")
    p.add_argument("--output-dir", default=None, help="pasta de saida")
    p.add_argument("--types", default="mass,nodule,density",
                   help="tipos de lesao separados por virgula")
    p.add_argument("--per-image", type=int, default=1,
                   help="quantas lesoes por imagem e por tipo")
    p.add_argument("--seed", type=int, default=1337, help="semente base")
    p.add_argument("--no-qc", action="store_true", help="desativa o portao de qualidade")
    p.add_argument("--set", action="append", dest="overrides", default=[],
                   metavar="CHAVE=VALOR", help="sobrescreve qualquer campo do config")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config.load(args.config).apply_overrides(args.overrides)
    if args.input_dir:
        cfg.paths.input_dir = args.input_dir
    if args.output_dir:
        cfg.paths.output_dir = args.output_dir
    if args.no_qc:
        cfg.quality.qc_enabled = False

    input_dir = cfg.resolve_input_dir()
    out_dir = cfg.resolve_output_dir()
    images = list_images(input_dir)
    if not images:
        print("nenhuma imagem encontrada em {}".format(input_dir))
        return 1

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    print("imagens: {} | tipos: {} | por imagem: {}".format(len(images), types, args.per_image))
    print("modelo: {}".format(cfg.paths.model_id))

    inpainter = LesionInpainter(cfg)
    rng = np.random.default_rng(args.seed)
    records = []
    rejected = 0

    for image_path in images:
        case = prepare_case(image_path, cfg)
        print("\n[{}] canvas {}px | mm/px {:.4f} | lado {}".format(
            case.name, case.size, case.mm_per_px, case.side))
        for lesion_type in types:
            for k in range(args.per_image):
                attempts = cfg.quality.qc_max_attempts if cfg.quality.qc_enabled else 1
                for attempt in range(attempts):
                    seed = int(rng.integers(0, 2 ** 31 - 1))
                    box = random_box(case, cfg, lesion_type, rng)
                    if box is None:
                        print("  {} #{}: nenhuma regiao valida".format(lesion_type, k))
                        break
                    result = synthesize(case, inpainter, box, cfg,
                                        lesion_type=lesion_type, seed=seed)
                    ev = evaluate(result, case, cfg.quality, lesion_type)
                    ok, failed = (True, []) if not cfg.quality.qc_enabled else \
                        passes_qc(ev, cfg.quality.qc_required)
                    if ok:
                        tag = "{:02d}".format(k)
                        paths = save_result(result, case, cfg, out_dir, tag)
                        records.append({
                            "case": case.name, "lesion_type": lesion_type, "index": k,
                            "seed": seed, "attempts": attempt + 1,
                            "diameter_mm": result.params["lesion_diameter_mm"],
                            "weber_contrast": ev["metrics"]["weber_contrast"],
                            "cnr": ev["metrics"]["cnr"],
                            "criteria_passed": ev["plausibility"]["n_passed"],
                            "criteria_total": ev["plausibility"]["n_total"],
                            "image": paths["image"],
                        })
                        print("  {} #{}: OK (tentativa {}) contraste {:.3f} CNR {:.2f} -> {}"
                              .format(lesion_type, k, attempt + 1,
                                      ev["metrics"]["weber_contrast"], ev["metrics"]["cnr"],
                                      Path(paths["image"]).name))
                        break
                    rejected += 1
                    print("  {} #{}: rejeitado ({}) - regerando".format(
                        lesion_type, k, ", ".join(failed)))
                else:
                    print("  {} #{}: descartado apos {} tentativas".format(
                        lesion_type, k, attempts))

    if records:
        manifest = pd.DataFrame(records)
        manifest_path = out_dir / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print("\n{} imagens geradas | {} rejeitadas pelo QC".format(len(records), rejected))
        print("manifesto: {}".format(manifest_path))
        print(manifest.groupby("lesion_type")[["diameter_mm", "weber_contrast", "cnr"]].mean())
    else:
        print("\nnenhuma imagem gerada")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
