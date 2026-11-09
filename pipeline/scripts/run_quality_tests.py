"""Bateria completa de testes de qualidade sobre as lesoes sinteticas.

Executa, para um lote de geracoes:

1. metricas de imagem na regiao gerada (contraste, textura, forma, artefatos);
2. deteccao por CAD classico (LoG multiescala) antes e depois da lesao;
3. deteccao de anomalia em features DINOv2 ajustada apenas com tecido saudavel;
4. mapas de XAI (occlusion, saliency, Grad-CAM) sobre o escore de anomalia;
5. sondas lineares de indistinguibilidade real x sintetico;

e escreve figuras, CSV de metricas e um relatorio em Markdown.

Exemplo
-------
    python scripts/run_quality_tests.py --per-image 2 --xai-cases 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mamelesion import Config, list_images, prepare_case, synthesize
from mamelesion.generator import random_box
from mamelesion.inpainting import LesionInpainter
from mamelesion import cad, quality, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--output-dir", default=None, help="pasta do relatorio (padrao: outputs/quality_report)")
    p.add_argument("--types", default="mass,nodule,density")
    p.add_argument("--per-image", type=int, default=2)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--xai-cases", type=int, default=3,
                   help="quantos casos recebem os mapas de XAI (occlusion e caro)")
    p.add_argument("--no-qc", action="store_true",
                   help="avalia todas as geracoes, inclusive as que o QC rejeitaria")
    p.add_argument("--set", action="append", dest="overrides", default=[])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config.load(args.config).apply_overrides(args.overrides)
    out_dir = Path(args.output_dir) if args.output_dir else cfg.resolve_output_dir() / "quality_report"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    images = list_images(cfg.resolve_input_dir())
    if not images:
        print("nenhuma imagem em {}".format(cfg.resolve_input_dir()))
        return 1
    types = [t.strip() for t in args.types.split(",") if t.strip()]

    print("preparando {} casos...".format(len(images)))
    cases = [prepare_case(p, cfg) for p in images]
    inpainter = LesionInpainter(cfg)
    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------- geracao
    print("gerando {} lesoes...".format(len(cases) * len(types) * args.per_image))
    generations = []
    qc_rejected = {t: 0 for t in types}
    for case in cases:
        for lesion_type in types:
            for k in range(args.per_image):
                box = random_box(case, cfg, lesion_type, rng)
                if box is None:
                    continue
                seed = int(rng.integers(0, 2 ** 31 - 1))
                result = synthesize(case, inpainter, box, cfg, lesion_type=lesion_type, seed=seed)
                ev = quality.evaluate(result, case, cfg.quality, lesion_type)
                ok, failed = quality.passes_qc(ev, cfg.quality.qc_required)
                if not ok:
                    qc_rejected[lesion_type] += 1
                    if not args.no_qc:
                        continue
                generations.append({"case": case, "result": result, "eval": ev,
                                    "type": lesion_type, "qc_ok": ok, "qc_failed": failed})
    print("  {} geracoes retidas | rejeitadas pelo QC: {}".format(len(generations), qc_rejected))
    if not generations:
        print("nenhuma geracao valida")
        return 1
    inpainter.unload()

    # -------------------------------------------------------------- CAD classico
    print("rodando CAD classico (LoG multiescala)...")
    for gen in generations:
        case, result = gen["case"], gen["result"]
        blob = cad.BlobCAD(cfg.quality.cad_scales_mm, case.mm_per_px, cfg.quality.cad_top_k)
        gen["eval"]["cad"] = cad.detection_report(blob, result.original, result.composed,
                                                  case.breast, result.box)

    # ------------------------------------------------- anomalia em features DINOv2
    print("ajustando detector de anomalia DINOv2 com tecido saudavel...")
    anomaly = cad.Dinov2Anomaly(cfg.quality)
    anomaly.fit([c.canvas for c in cases], [c.breast for c in cases])
    for gen in generations:
        case, result = gen["case"], gen["result"]
        rep = cad.anomaly_report(anomaly, result.original, result.composed,
                                 case.breast, result.box)
        gen["heat"] = {"anomalia (Mahalanobis)": rep.pop("heat_synthetic")}
        rep.pop("heat_original", None)
        gen["eval"]["anomaly"] = rep

    # --------------------------------------------------------------------- XAI
    # escolhe os casos de XAI distribuidos entre os tipos de lesao, para que o
    # relatorio nao acabe com tres exemplos do mesmo tipo
    selected, seen_types = [], set()
    for gen in generations:
        if gen["type"] not in seen_types:
            selected.append(gen)
            seen_types.add(gen["type"])
    chosen_ids = {id(g) for g in selected}
    selected.extend(g for g in generations if id(g) not in chosen_ids)
    selected = selected[:min(args.xai_cases, len(generations))]

    print("calculando mapas de XAI em {} casos ({})...".format(
        len(selected), ", ".join(g["type"] for g in selected)))
    xai_records = []
    for gen in selected:
        case, result = gen["case"], gen["result"]
        maps = {
            "saliency": cad.saliency_map(anomaly, result.composed),
            "Grad-CAM": cad.gradcam_map(anomaly, result.composed),
            "occlusion": cad.occlusion_map(anomaly, result.composed, 96, 48),
        }
        xai_stats = {}
        for name, heat in maps.items():
            gen["heat"][name] = heat
            stats = cad.xai_hit_rate(heat, result.box, case.breast)
            for key, value in stats.items():
                xai_stats["{}_{}".format(name.split()[0].lower(), key)] = value
            xai_records.append({"metodo": name, "caso": case.name,
                                "tipo": gen["type"], **stats})
        gen["eval"]["xai"] = xai_stats
    xai_df = pd.DataFrame(xai_records)

    # ------------------------------------------------- sondas de indistinguibilidade
    print("treinando sondas lineares de indistinguibilidade...")
    # Duas sondas:
    #  1. lesao sintetica x tecido real -> AUC alta significa lesao detectavel;
    #  2. impressao digital do gerador: recortes pequenos, contidos na area
    #     reconstruida e fora da lesao, contra *o mesmo recorte na imagem
    #     original*. O controle pareado por coordenada elimina o vies de
    #     localizacao (densidade local, distancia da pele), deixando como unica
    #     diferenca entre os dois grupos a passagem pelo modelo generativo.
    patch = cfg.quality.patch_size
    probe_patch = cfg.quality.probe_patch_size
    prng = np.random.default_rng(args.seed + 1)
    lesion_patches, real_patches = [], []
    bg_synth, bg_real_paired = [], []

    for gen in generations:
        result, ev = gen["result"], gen["eval"]
        cx, cy = result.box.center
        crop = cad.crop_at(result.composed, (cy, cx), patch)
        if crop is not None:
            lesion_patches.append(crop)
        inside_not_lesion = (result.mask > 127) & ~ev["lesion_mask"]
        for center in cad.sample_coords_in_region(inside_not_lesion, probe_patch, 3, prng):
            a = cad.crop_at(result.composed, center, probe_patch)
            b = cad.crop_at(result.original, center, probe_patch)
            if a is not None and b is not None:
                bg_synth.append(a)
                bg_real_paired.append(b)

    per_case = max(1, cfg.quality.n_control_patches // len(cases))
    for case in cases:
        real_patches.extend(cad.sample_tissue_patches(
            case.canvas, case.breast, patch, per_case, rng=prng))

    print("  patches: lesao={} ({}px) | fundo reconstruido x original pareado={} ({}px) "
          "| tecido real={}".format(len(lesion_patches), patch, len(bg_synth),
                                    probe_patch, len(real_patches)))

    probes = {}
    probes["lesao sintetica x tecido real ({} px)".format(patch)] = dict(
        cad.linear_probe_auc(anomaly.embed(lesion_patches), anomaly.embed(real_patches)),
        reading="AUC alta = a lesao e detectavel (sensibilidade desejada)")
    if bg_synth:
        probes["fundo reconstruido x mesmo local na imagem real ({} px)".format(probe_patch)] = dict(
            cad.linear_probe_auc(anomaly.embed(bg_synth), anomaly.embed(bg_real_paired)),
            reading="AUC proxima de 0.5 = sem impressao digital do gerador")

        # Mesma sonda com o nivel medio de cada recorte removido: separa um
        # eventual desvio de intensidade (corrigivel) de uma diferenca real de
        # textura (intrinseca ao gerador).
        def _dc_free(patches):
            out = []
            for p in patches:
                f = p.astype(np.float32)
                out.append(np.clip(f - f.mean() + 128.0, 0, 255).astype(np.uint8))
            return out

        probes["idem, sem o nivel medio (so textura, {} px)".format(probe_patch)] = dict(
            cad.linear_probe_auc(anomaly.embed(_dc_free(bg_synth)),
                                 anomaly.embed(_dc_free(bg_real_paired))),
            reading="isola a textura: queda em relacao a linha acima indica que a "
                    "diferenca e de nivel medio, nao de microestrutura")
        deltas = np.array([float(a.astype(np.float32).mean() - b.astype(np.float32).mean())
                           for a, b in zip(bg_synth, bg_real_paired)])
        probes["fundo reconstruido x mesmo local na imagem real ({} px)".format(probe_patch)][
            "delta_intensidade_medio"] = float(deltas.mean())
    else:
        print("  aviso: nenhuma regiao reconstruida comporta um recorte de {} px; "
              "sonda de impressao digital nao executada".format(probe_patch))

    # ------------------------------------------------------------------ saidas
    print("escrevendo figuras e relatorio...")
    evaluations = [g["eval"] for g in generations]
    df = report.metrics_dataframe(evaluations)
    df.to_csv(out_dir / "metrics.csv", index=False)

    figures = {}
    dist_path = out_dir / "figures" / "distribuicoes.png"
    report.distribution_figure(df, dist_path)
    figures["Distribuicao das metricas por tipo"] = "figures/distribuicoes.png"

    shown = {}
    for gen in generations:
        key = gen["type"]
        if key in shown:
            continue
        shown[key] = True
        name = "caso_{}_{}".format(gen["eval"]["case"], key)
        path = out_dir / "figures" / (name + ".png")
        report.case_figure(gen["result"], gen["case"], gen["eval"], path,
                           gen.get("heat"))
        figures["Exemplo - {}".format(key)] = "figures/{}.png".format(name)

    context = {
        "modelo de inpainting": cfg.paths.model_id,
        "resolucao do modelo": "{}x{} (latente {}x{})".format(
            cfg.preprocessing.model_resolution, cfg.preprocessing.model_resolution,
            cfg.preprocessing.model_resolution // 8, cfg.preprocessing.model_resolution // 8),
        "resolucao de saida": "{}x{}".format(cfg.output.size, cfg.output.size),
        "prompt": cfg.lesion.prompt,
        "passos de difusao / guidance": "{} / {}".format(
            cfg.diffusion.num_inference_steps, cfg.diffusion.guidance_scale),
        "scheduler": cfg.diffusion.scheduler,
        "composicao": "{} + {} (detail_transfer={}, feather={}px)".format(
            cfg.composition.mode, cfg.composition.blend,
            cfg.composition.detail_transfer, cfg.composition.feather_px),
        "imagens de origem": len(cases),
        "geracoes avaliadas": len(generations),
        "rejeitadas pelo QC": qc_rejected,
        "anel de referencia (px)": cfg.quality.ring_px,
    }

    checks = report.checks_summary_frame(evaluations)
    report_path = out_dir / "RELATORIO_QUALIDADE.md"
    report.write_report(report_path, df, context, figures, probes, checks, xai_df)

    with open(out_dir / "probes.json", "w", encoding="utf-8") as fh:
        json.dump(probes, fh, indent=2, ensure_ascii=False)

    print("\nrelatorio: {}".format(report_path))
    print("metricas: {}".format(out_dir / "metrics.csv"))
    for name, info in probes.items():
        print("  sonda {}: AUC {:.3f}".format(name, info.get("auc", float("nan"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
