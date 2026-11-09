#!/usr/bin/env python
"""Fases 1, 2 e 4: o quanto cada forma de gerar se aproxima das lesoes BI-RADS reais.

Tres bracos, sobre os **mesmos** recortes reais, hospedeiros, posicoes e sementes:

``atual``
    o pipeline de antes deste trabalho: canvas 1024 reduzido, checkpoint original,
    prompt generico, preset ``mass``. E a linha de base (fase 1).
``nativo_base``
    janela em resolucao nativa com o checkpoint original (fase 2): isola o ganho
    da mudanca de escala.
``nativo_lora``
    janela nativa + adaptador LoRA do treino, prompt da classe (fase 4): isola o
    ganho do ajuste fino.

Cada lesao sintetica tem o tamanho exato de um recorte real da particao e e
gerada num hospedeiro da mesma particao -- pacientes e mamografias que o treino
nunca viu. Use ``--particao test`` uma unica vez, no fim; comparacoes durante o
desenvolvimento vao em ``val``.

Saida em ``--out``: ``avaliacao.json``, ``AVALIACAO.md``, ``grade.png``.

    .venv/bin/python scripts/evaluate_birads.py --run treinos/birads_lora_v1 --particao test
"""

from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                                      # noqa: E402
from PIL import Image                                                   # noqa: E402

from mamelesion.birads_data import (CLASS_PROMPTS, GENERIC_PROMPT, LesionCrop, load_json,  # noqa: E402
                                    parse_ids, read_crop, save_json)
from mamelesion.compose import compose                                  # noqa: E402
from mamelesion.config import Config                                    # noqa: E402
from mamelesion.finetune import load_run                                # noqa: E402
from mamelesion.inpainting import LesionInpainter                       # noqa: E402
from mamelesion.lesion_eval import (DinoFeatures, centroid_cosine, classifier_auc, comparison_grid,  # noqa: E402
                                    contrast_ratio, kid, memorization, plan_jobs, texture_distances,
                                    vae_ceiling_crops)
from mamelesion.masks import LesionBox, mask_from_box, snap_to_latent   # noqa: E402
from mamelesion.native import build_breast_canvas, lesion_delta, read_source, writeback_native  # noqa: E402
from mamelesion.nativegen import evaluate_window, generate_native       # noqa: E402
from mamelesion.paste import prepare_host                               # noqa: E402
from mamelesion.window import anatomical_center, native_footprint       # noqa: E402

ARMS = ("atual", "nativo_base", "nativo_lora")
EVAL_BASE_SEED = 4242


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="pasta do treino (usa melhor_checkpoint e split.json dela)")
    ap.add_argument("--checkpoint", default=None, help="checkpoint especifico em vez do melhor")
    ap.add_argument("--particao", choices=("val", "test"), default="test")
    ap.add_argument("--bracos", default=",".join(ARMS))
    ap.add_argument("--sementes", type=int, default=3, help="lesoes por recorte real")
    ap.add_argument("--passos", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=None,
                    help="forca a mesma guidance em todos os bracos (padrao: 4,0 nos bracos do checkpoint "
                         "original, melhor_guidance do treino no braco LoRA)")
    ap.add_argument("--lora-escala", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="padrao: <run>/avaliacao_<particao>")
    return ap.parse_args(argv)


class CanvasArm:
    """O modo canvas exatamente como ``SyntheticLesionPipeline`` o executa, sem o sorteio de caixa."""

    def __init__(self, cfg, inpainter):
        self.cfg, self.inpainter = cfg, inpainter
        self.cache = {}
        preset = cfg.lesion.preset("mass")
        self.comp = deepcopy(cfg.composition)
        self.comp.detail_transfer, self.comp.lesion_gain = preset.detail_transfer, preset.lesion_gain

    def generate(self, host_path, center, wh, seed, steps, guidance):
        if host_path not in self.cache:
            source = read_source(host_path, self.cfg.preprocessing.max_value)
            canvas, framing, _ = build_breast_canvas(source.gray_u8, 1024,
                                                     breast_threshold=self.cfg.preprocessing.breast_threshold)
            self.cache[host_path] = (source, canvas, framing)
        source, canvas, framing = self.cache[host_path]
        fx0, fy0, fx1, fy1 = native_footprint(center, wh, source.raw.shape)
        cx0, cy0 = framing.orig_to_canvas(fx0, fy0)
        cx1, cy1 = framing.orig_to_canvas(fx1, fy1)
        box = LesionBox(int(np.floor(cx0)), int(np.floor(cy0)), int(np.ceil(cx1)), int(np.ceil(cy1)))
        if self.cfg.lesion.snap_to_latent:
            box = snap_to_latent(box, 1024, self.cfg.preprocessing.model_resolution)
        mask = mask_from_box(box, 1024, self.cfg.lesion.mask_shape)
        result = self.inpainter.generate(canvas, mask, box, lesion_type="mass", seed=seed, steps=steps,
                                         guidance=guidance, prompt=GENERIC_PROMPT)
        result.composed = compose(canvas, result.generated_model_res, mask, self.comp, box)
        delta = lesion_delta(result, self.comp)
        native, _, mask_native = writeback_native(source, framing, delta, box, result.mask)
        ref = source.scale_to_u8 * 255.0
        changed_outside = int(((native != source.raw) & (mask_native == 0)).sum())
        return native[fy0:fy1, fx0:fx1].astype(np.float32) / ref, changed_outside, None


def main(argv=None) -> int:
    args = parse_args(argv)
    arms = [a.strip() for a in args.bracos.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        print("erro: bracos desconhecidos: {}".format(sorted(unknown)), file=sys.stderr)
        return 2
    run_dir = Path(args.run)
    out = Path(args.out) if args.out else run_dir / "avaliacao_{}".format(args.particao)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config.load()
    checkpoint = None
    if args.checkpoint:
        # aceita caminho absoluto, relativo a pasta atual ou relativo a pasta do treino
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists() and (run_dir / checkpoint).exists():
            checkpoint = run_dir / checkpoint
    run = load_run(checkpoint or run_dir)
    diffs = run.apply_geometry(cfg.native_window)
    if diffs:
        print("geometria do treino aplicada (difere do config.yaml): " + "; ".join(diffs))
    split = load_json(run_dir / "split.json")

    def crops_of(names):
        res = []
        for n in names:
            p = Path(split["caminho_de"][n])
            res.append(LesionCrop(n, p, split["classe_de"][n], *parse_ids(n), read_crop(p)))
        return res

    reals = crops_of(split["recortes"][args.particao])
    train = crops_of(split["recortes"]["train"])
    hosts_dir = Path(split["hospedeiros_pasta"])
    host_names = split["hospedeiros"][args.particao]
    hosts = [prepare_host(hosts_dir / n, cfg) for n in host_names]
    jobs = plan_jobs([(c.name, c.lesion_class, c.wh) for c in reals], len(hosts), args.sementes, EVAL_BASE_SEED)
    centers = []
    for job in jobs:
        h = hosts[job.host_index]
        centers.append(anatomical_center(h.pmap, h.framing, h.mm_per_px_canvas, job.wh,
                                         np.random.default_rng(job.place_seed)))
    print("particao {}: {} recortes reais ({}), {} hospedeiros, {} lesoes por braco".format(
        args.particao, len(reals), dict((k, sum(c.lesion_class == k for c in reals)) for k in CLASS_PROMPTS),
        len(hosts), len(jobs)))

    base_guidance = args.guidance if args.guidance is not None else 4.0
    lora_guidance = args.guidance if args.guidance is not None else float(
        run.config.get("melhor_guidance") or 4.0)
    guidance_of = {"atual": base_guidance, "nativo_base": base_guidance, "nativo_lora": lora_guidance}
    synth = {a: [] for a in arms}
    by_ref = {a: {} for a in arms}
    integrity = {a: 0 for a in arms}
    qc = {a: [] for a in arms}
    started = time.time()

    base = None
    if "atual" in arms or "nativo_base" in arms:
        base = LesionInpainter(cfg)
    if "atual" in arms:
        canvas_arm = CanvasArm(cfg, base)
        for job, (center, _) in zip(jobs, centers):
            crop, outside, _ = canvas_arm.generate(hosts_dir / host_names[job.host_index], center, job.wh,
                                                   job.seed, args.passos, guidance_of["atual"])
            synth["atual"].append(crop)
            by_ref["atual"].setdefault(job.reference, []).append(crop)
            integrity["atual"] += outside
        print("  atual: {:.0f} s".format(time.time() - started))

    def native_arm(name, inpainter, prompt_of):
        t = time.time()
        for job, (center, _) in zip(jobs, centers):
            h = hosts[job.host_index]
            nl = generate_native(inpainter.generate_window, h.raw, h.ref, center, job.wh, cfg.native_window,
                                 prompt_of(job), job.seed, args.passos, guidance_of[name])
            crop = nl.lesion_crop(h.ref)
            synth[name].append(crop)
            by_ref[name].setdefault(job.reference, []).append(crop)
            integrity[name] += int(((nl.raw_out != h.raw) & (nl.mask_native == 0)).sum())
            ev = evaluate_window(nl, h.mm_per_px_canvas * h.framing.scale, cfg.quality, "mass",
                                 cfg.preprocessing.breast_threshold)
            qc[name].append(bool(ev["qc_ok"]))
        print("  {}: {:.0f} s".format(name, time.time() - t))

    if "nativo_base" in arms:
        native_arm("nativo_base", base, lambda job: GENERIC_PROMPT)
    if base is not None:
        base.unload()
    if "nativo_lora" in arms:
        lora = LesionInpainter(cfg, lora_path=str(run.lora_dir), lora_scale=args.lora_escala)
        native_arm("nativo_lora", lora, lambda job: CLASS_PROMPTS[job.lesion_class])
        lora.unload()

    # --- metricas ------------------------------------------------------------------
    ceiling = vae_ceiling_crops(reals, hosts, cfg)
    dino = DinoFeatures()
    f_real = dino([c.image for c in reals])
    f_train = dino([c.image for c in train])
    report = {"particao": args.particao, "run": str(run_dir), "checkpoint": str(run.lora_dir),
              "geometria": run.native_window, "sementes": args.sementes, "passos": args.passos,
              "guidance_por_braco": {a: guidance_of[a] for a in arms}, "lora_escala": args.lora_escala,
              "n_reais": len(reals), "hospedeiros": host_names, "bracos": {}}
    # referencia de ruido: quanto dois conjuntos de lesoes REAIS diferem entre si
    f_other = dino([c.image for c in crops_of(split["recortes"]["val" if args.particao == "test" else "test"])])
    report["referencia_real_x_real"] = {"kid": kid(f_real, f_other), "auc_classificador": classifier_auc(f_real, f_other)}
    f_ceil = dino(ceiling)
    report["teto_vae"] = {"kid": kid(f_real, f_ceil), "auc_classificador": classifier_auc(f_real, f_ceil),
                          "razao_contraste": contrast_ratio([c.image for c in reals], ceiling),
                          "textura_mediana_w1": texture_distances([c.image for c in reals], ceiling)[
                              "mediana_w1_normalizada"]}
    for a in arms:
        f = dino(synth[a])
        report["bracos"][a] = {
            "kid": kid(f_real, f),
            "auc_classificador": classifier_auc(f_real, f),
            "cosseno_centroides": centroid_cosine(f_real, f),
            "memorizacao": memorization(f_train, f_real, f),
            "textura": texture_distances([c.image for c in reals], synth[a]),
            "razao_contraste": contrast_ratio([c.image for c in reals], synth[a]),
            "pixels_alterados_fora_da_mascara": integrity[a],
            "taxa_qc": (float(np.mean(qc[a])) if qc[a] else None),
            "n_sinteticos": len(synth[a]),
        }
    report["duracao_s"] = round(time.time() - started, 1)
    save_json(report, out / "avaliacao.json")

    rows = [[c.image] + [by_ref[a][c.name][0] for a in arms] for c in reals[:14]]
    Image.fromarray(comparison_grid(rows, ["real"] + list(arms))).save(out / "grade.png")

    # --- relatorio -----------------------------------------------------------------
    ref = report["referencia_real_x_real"]
    lines = ["# Avaliacao BI-RADS -- particao `{}`".format(args.particao), "",
             "{} recortes reais de pacientes nunca vistos, {} lesoes sinteticas por braco ({} por recorte), "
             "hospedeiros {}. Checkpoint: `{}`.".format(len(reals), len(jobs), args.sementes,
                                                        ", ".join(host_names), run.lora_dir), "",
             "| Braco | KID (menor melhor) | AUC real x sint. | Razao de contraste (1 ideal) | Textura W1 (menor melhor) "
             "| Memorizacao (esperado {:.2f}) | QC | Pixels fora da mascara |".format(
                 report["bracos"][arms[0]]["memorizacao"]["esperado_sem_copia"]),
             "|---|---|---|---|---|---|---|---|"]
    t = report["teto_vae"]
    lines.append("| *referencia: reais reconstruidas pelo VAE* | {:.3f} +- {:.3f} | {:.2f} [{:.2f}-{:.2f}] | {:.2f} | {:.2f} | - | - | - |".format(
        t["kid"]["media"], t["kid"]["desvio"], t["auc_classificador"]["media"], *t["auc_classificador"]["ic95"],
        t["razao_contraste"], t["textura_mediana_w1"]))
    for a in arms:
        r = report["bracos"][a]
        lines.append("| `{}` | {:.3f} +- {:.3f} | {:.2f} [{:.2f}-{:.2f}] | {:.2f} | {:.2f} | {:.2f} | {} | {} |".format(
            a, r["kid"]["media"], r["kid"]["desvio"], r["auc_classificador"]["media"],
            r["auc_classificador"]["ic95"][0], r["auc_classificador"]["ic95"][1], r["razao_contraste"],
            r["textura"]["mediana_w1_normalizada"], r["memorizacao"]["fracao_acima"],
            "-" if r["taxa_qc"] is None else "{:.0%}".format(r["taxa_qc"]), r["pixels_alterados_fora_da_mascara"]))
    lines += ["", "Duas referencias para ler a tabela:", "",
              "- **Piso de ruido** -- lesoes reais de `{}` contra reais da outra particao: KID {:.3f}, "
              "AUC {:.2f}. Nenhum braco pode ficar abaixo disso.".format(
                  args.particao, ref["kid"]["media"], ref["auc_classificador"]["media"]),
              "- **Efeito do VAE** -- as mesmas lesoes reais, inteiras, depois de ida e volta pelo VAE "
              "congelado. Mostra quanto do AUC vem so da reconstrucao do VAE. Nao e um limite rigido: os "
              "bracos preservam pixels nativos na transicao e fora da mascara.", "",
              "## Textura por estatistica (mediana real -> sintetica)", "",
              "| Estatistica | Real | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    keys = [k for k in report["bracos"][arms[0]]["textura"] if k != "mediana_w1_normalizada"]
    for k in keys:
        realv = report["bracos"][arms[0]]["textura"][k]["real_mediana"]
        lines.append("| {} | {:.4g} | ".format(k, realv) + " | ".join(
            "{:.4g} (W1 {:.2f})".format(report["bracos"][a]["textura"][k]["sint_mediana"],
                                        report["bracos"][a]["textura"][k]["w1_normalizada"]) for a in arms) + " |")
    lines += ["", "`grade.png`: uma linha por recorte real; colunas na ordem da tabela.", ""]
    (out / "AVALIACAO.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[4:5 + len(arms) + 3]))
    print("saida:", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
