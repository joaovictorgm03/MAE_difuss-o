#!/usr/bin/env python
"""Por que um classificador separa lesoes sinteticas das reais -- e o que reduz isso sem retreino.

Usa so recortes de **treino e validacao** (o teste fica intocado). Mede o AUC de
um classificador real x sintetico (``lesion_eval.classifier_auc``) em controles:

``real_val x real_treino``
    duas amostras reais de pacientes diferentes -- o piso de ruido do teste.
``real_val x real_val_vae``
    o mesmo recorte real, antes e depois de passar pelo VAE na janela nativa:
    quanto o VAE sozinho denuncia uma imagem.
``real_val x sint_<variante>``
    lesoes geradas pelo checkpoint escolhido, com variantes de **composicao**
    aplicadas a mesma saida do modelo (nada e regerado):

    - ``atual``: delta = saida - original (como ``window.window_delta``);
    - ``passa_baixa_s``: delta = G_s * (saida - original). A imagem entregue
      mantem o ruido nativo do detector dentro da lesao e recebe so o sinal da
      lesao ate a escala ``s`` (px na entrada do modelo).

Para cada variante tambem mede razao de contraste e distancia de textura, porque
passar-baixa demais apaga espiculas: menos separavel nao basta se a lesao some.

    .venv/bin/python scripts/analise_separabilidade.py --run treinos/birads_lora_v1
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2 as cv                                                        # noqa: E402
import numpy as np                                                      # noqa: E402

from mamelesion.birads_data import CLASS_PROMPTS, LesionCrop, load_json, parse_ids, read_crop, save_json  # noqa: E402
from mamelesion.config import Config                                    # noqa: E402
from mamelesion.finetune import RUN_FILE, load_run                      # noqa: E402
from mamelesion.inpainting import LesionInpainter                       # noqa: E402
from mamelesion.lesion_eval import (DinoFeatures, classifier_auc, contrast_ratio, plan_jobs,  # noqa: E402
                                    texture_distances, vae_ceiling_crops)
from mamelesion.nativegen import generate_native                        # noqa: E402
from mamelesion.paste import prepare_host                               # noqa: E402
from mamelesion.window import anatomical_center, window_delta, writeback  # noqa: E402

SIGMAS = (0.75, 1.5, 3.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--passos", type=int, default=30)
    args = ap.parse_args()

    run_dir = Path(args.run)
    info = load_json(run_dir / RUN_FILE)
    run = load_run(run_dir)
    cfg = Config.load()
    run.apply_geometry(cfg.native_window)
    guidance = float(info.get("melhor_guidance") or 4.0)
    split = load_json(run_dir / "split.json")

    def crops_of(names):
        return [LesionCrop(n, Path(split["caminho_de"][n]), split["classe_de"][n], *parse_ids(n),
                           read_crop(Path(split["caminho_de"][n]))) for n in names]

    val, train = crops_of(split["recortes"]["val"]), crops_of(split["recortes"]["train"])
    hosts = [prepare_host(Path(split["hospedeiros_pasta"]) / n, cfg) for n in split["hospedeiros"]["val"]]
    dino = DinoFeatures()
    f_val, f_train = dino([c.image for c in val]), dino([c.image for c in train])
    results = {"checkpoint": str(run.lora_dir), "guidance": guidance}

    def row(name, f_other, crops_other=None):
        auc = classifier_auc(f_val, f_other)
        r = {"auc": auc["media"], "ic95": auc["ic95"]}
        if crops_other is not None:
            r["razao_contraste"] = contrast_ratio([c.image for c in val], crops_other)
            r["textura_w1"] = texture_distances([c.image for c in val], crops_other)["mediana_w1_normalizada"]
        results[name] = r
        extra = "" if crops_other is None else " | contraste {:.2f} | textura {:.2f}".format(
            r["razao_contraste"], r["textura_w1"])
        print("{:34s} AUC {:.2f} [{:.2f}-{:.2f}]{}".format(name, r["auc"], *r["ic95"], extra), flush=True)

    row("real_val x real_treino", f_train)

    # --- controle do VAE: recorte real colado na janela do hospedeiro, ida e volta ---------------
    vae_crops = vae_ceiling_crops(val, hosts, cfg)
    row("real_val x real_val_pelo_vae", dino(vae_crops), vae_crops)

    # --- sinteticos do checkpoint escolhido, variantes de composicao sobre a mesma saida --------
    inp = LesionInpainter(cfg, lora_path=str(run.lora_dir))
    jobs = plan_jobs([(c.name, c.lesion_class, c.wh) for c in val], len(hosts), args.sementes, 777)
    variants = {"atual": []}
    variants.update({"passa_baixa_{}".format(s): [] for s in SIGMAS})
    nw = cfg.native_window
    for job in jobs:
        h = hosts[job.host_index]
        center, _ = anatomical_center(h.pmap, h.framing, h.mm_per_px_canvas, job.wh,
                                      np.random.default_rng(job.place_seed))
        nl = generate_native(inp.generate_window, h.raw, h.ref, center, job.wh, nw,
                             CLASS_PROMPTS[job.lesion_class], job.seed, args.passos, guidance)
        variants["atual"].append(nl.lesion_crop(h.ref))
        diff = nl.generated_u8.astype(np.float32) - nl.original_u8.astype(np.float32)
        for s in SIGMAS:
            gen = nl.original_u8.astype(np.float32) + cv.GaussianBlur(diff, (0, 0), s)
            delta = window_delta(nl.original_u8, gen, nl.mask_u8, nl.box, nw.feather_frac)
            out, _ = writeback(h.raw, nl.window, delta, nl.mask_u8, h.ref)
            x0, y0, x1, y1 = nl.footprint
            variants["passa_baixa_{}".format(s)].append(out[y0:y1, x0:x1].astype(np.float32) / h.ref)
    inp.unload()
    for name, crops in variants.items():
        row("real_val x sint_{}".format(name), dino(crops), crops)

    out = run_dir / "selecao" / "separabilidade.json"
    save_json(results, out)
    print("gravado:", out)


if __name__ == "__main__":
    main()
