#!/usr/bin/env python
"""Reavalia todos os checkpoints de um treino na validacao e escolhe o melhor.

Por que existe separado do treino: a escolha precisa ser refeita quando o
criterio muda, sem retreinar. Foi o caso do primeiro treino -- o KID escolhia o
modelo base, que nao gera lesao nenhuma (ver ``lesion_eval.contrast_ratio``).

Para cada checkpoint (e o modelo base como referencia) gera as mesmas lesoes --
mesmos recortes de validacao, hospedeiros, posicoes e sementes -- e aplica
``lesion_eval.selection_key``. Opcionalmente varre a ``guidance`` de inferencia.
Grava ``selecao/selecao.json``, ``selecao/SELECAO.md`` e atualiza
``melhor_checkpoint`` e ``melhor_guidance`` em ``treino.json``.

    .venv/bin/python scripts/select_checkpoint.py --run treinos/birads_lora_v1 --guidance 2.5,4.0
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                                     # noqa: E402
from PIL import Image                                                  # noqa: E402

from mamelesion.birads_data import (CLASS_PROMPTS, GENERIC_PROMPT, LesionCrop, load_json,  # noqa: E402
                                    parse_ids, read_crop, save_json)
from mamelesion.config import Config                                   # noqa: E402
from mamelesion.finetune import LORA_WEIGHTS, RUN_FILE, load_run       # noqa: E402
from mamelesion.inpainting import LesionInpainter                      # noqa: E402
from mamelesion.lesion_eval import (DinoFeatures, comparison_grid, plan_jobs, select_best,  # noqa: E402
                                    selection_key, validation_summary, CONTRAST_RATIO_RANGE, MEMO_MAX)
from mamelesion.nativegen import generate_native                       # noqa: E402
from mamelesion.paste import prepare_host                              # noqa: E402
from mamelesion.window import anatomical_center                        # noqa: E402

SEL_BASE_SEED = 777


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--guidance", default="4.0", help="valores separados por virgula")
    ap.add_argument("--sementes", type=int, default=2)
    ap.add_argument("--passos", type=int, default=30)
    ap.add_argument("--checkpoints", default="", help="pastas especificas (padrao: todas com pesos)")
    ap.add_argument("--sem-base", action="store_true", help="nao avaliar o modelo base")
    ap.add_argument("--so-reescolher", action="store_true",
                    help="nao gera nada: reaplica o criterio atual ao selecao.json existente")
    args = ap.parse_args()

    run_dir = Path(args.run)
    info = load_json(run_dir / RUN_FILE)
    if args.so_reescolher:
        previous = load_json(run_dir / "selecao" / "selecao.json")
        return finish(run_dir, info, previous["resultados"], previous.get("n_validacao"),
                      previous.get("sementes"), previous.get("hospedeiros"))
    cfg = Config.load()
    for k, v in info["geometria"].items():
        setattr(cfg.native_window, k, type(getattr(cfg.native_window, k))(v))
    split = load_json(run_dir / "split.json")

    def crops_of(names):
        return [LesionCrop(n, Path(split["caminho_de"][n]), split["classe_de"][n], *parse_ids(n),
                           read_crop(Path(split["caminho_de"][n]))) for n in names]

    val, train = crops_of(split["recortes"]["val"]), crops_of(split["recortes"]["train"])
    hosts = [prepare_host(Path(split["hospedeiros_pasta"]) / n, cfg) for n in split["hospedeiros"]["val"]]
    jobs = plan_jobs([(c.name, c.lesion_class, c.wh) for c in val], len(hosts), args.sementes, SEL_BASE_SEED)
    centers = [anatomical_center(hosts[j.host_index].pmap, hosts[j.host_index].framing,
                                 hosts[j.host_index].mm_per_px_canvas, j.wh,
                                 np.random.default_rng(j.place_seed))[0] for j in jobs]
    dino = DinoFeatures()
    f_val, f_train = dino([c.image for c in val]), dino([c.image for c in train])

    if args.checkpoints:
        folders = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    else:
        folders = [ck["pasta"] for ck in info["checkpoints"] if (run_dir / ck["pasta"] / LORA_WEIGHTS).exists()]
    candidates = ([] if args.sem_base else [None]) + folders
    guidances = [float(g) for g in args.guidance.split(",")]

    out_dir = run_dir / "selecao"
    out_dir.mkdir(exist_ok=True)
    results = []
    for folder in candidates:
        inp = LesionInpainter(cfg, lora_path=str(run_dir / folder) if folder else None)
        for g in guidances:
            t = time.time()
            synth, first = [], {}
            for job, center in zip(jobs, centers):
                h = hosts[job.host_index]
                prompt = CLASS_PROMPTS[job.lesion_class] if folder else GENERIC_PROMPT
                nl = generate_native(inp.generate_window, h.raw, h.ref, center, job.wh, cfg.native_window,
                                     prompt, job.seed, args.passos, g)
                crop = nl.lesion_crop(h.ref)
                synth.append(crop)
                first.setdefault(job.reference, crop)
            summary = validation_summary(f_val, dino(synth), f_train, [c.image for c in val], synth)
            name = folder or "modelo_base"
            row = {"checkpoint": folder, "nome": name, "guidance": g, **summary,
                   "chave": list(selection_key(summary)), "duracao_s": round(time.time() - t, 1)}
            results.append(row)
            tag = "{}_g{}".format(Path(name).name, g)
            Image.fromarray(comparison_grid([[c.image, first[c.name]] for c in val[:10]], ["real", tag])).save(
                out_dir / "grade_{}.png".format(tag))
            print("{:22s} g={:.1f} | AUC {:.2f} [{:.2f}-{:.2f}] | contraste {:.2f} | KID {:.3f} | memo {:.2f} | "
                  "textura {:.2f} | elegivel {}".format(
                      Path(name).name, g, summary["auc_classificador"]["media"], *summary["auc_classificador"]["ic95"],
                      summary["razao_contraste"], summary["kid"]["media"], summary["memorizacao"]["fracao_acima"],
                      summary["textura_mediana_w1"], row["chave"][0] == 0), flush=True)
        inp.unload()

    finish(run_dir, info, results, len(val), args.sementes, split["hospedeiros"]["val"])


def finish(run_dir, info, results, n_val, seeds, host_names):
    out_dir = run_dir / "selecao"
    lora_rows = [r for r in results if r["checkpoint"]]
    best, trail = (None, [])
    if lora_rows:
        i, trail = select_best(lora_rows)
        best = lora_rows[i]
    save_json({"criterio": {"faixa_razao_contraste": CONTRAST_RATIO_RANGE, "memorizacao_max": MEMO_MAX,
                            "ordem": "elegivel > AUC (empate no IC95) > KID (empate em 1 desvio) > textura W1 "
                                     "> contraste proximo de 1"},
               "resultados": results, "melhor": best, "trilha": trail, "n_validacao": n_val,
               "sementes": seeds, "hospedeiros": host_names}, out_dir / "selecao.json")

    lines = ["# Selecao de checkpoint -- validacao", "",
             "{} recortes de validacao x {} sementes, hospedeiros {}. Criterio em cascata, com empate "
             "estatistico: elegivel (razao de contraste em {} e memorizacao <= {}) > AUC real x sintetico "
             "(empate dentro do IC95) > KID (empate dentro de 1 desvio) > textura W1 > contraste proximo "
             "de 1.".format(n_val, seeds, ", ".join(host_names or []), CONTRAST_RATIO_RANGE, MEMO_MAX), "",
             "| Checkpoint | Guidance | AUC real x sint. | Razao contraste | KID | Memorizacao | Textura W1 | Elegivel |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append("| {} | {:.1f} | {:.2f} [{:.2f}-{:.2f}] | {:.2f} | {:.3f} | {:.2f} | {:.2f} | {} |".format(
            Path(r["nome"]).name, r["guidance"], r["auc_classificador"]["media"], *r["auc_classificador"]["ic95"],
            r["razao_contraste"], r["kid"]["media"], r["memorizacao"]["fracao_acima"], r["textura_mediana_w1"],
            "sim" if r["chave"][0] == 0 else "nao"))
    if best:
        lines += ["", "**Escolhido:** `{}` com guidance {:.1f}.".format(best["checkpoint"], best["guidance"]), "",
                  "Como a decisao foi tomada:", ""] + ["{}. {}".format(k + 1, t) for k, t in enumerate(trail)]
    (out_dir / "SELECAO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if best:
        info["melhor_checkpoint"] = best["checkpoint"]
        info["melhor_guidance"] = best["guidance"]
        info["melhor_criterio"] = {"origem": "scripts/select_checkpoint.py",
                                   "auc_classificador": best["auc_classificador"],
                                   "razao_contraste": best["razao_contraste"], "kid": best["kid"]["media"],
                                   "memorizacao_fracao_acima": best["memorizacao"]["fracao_acima"],
                                   "elegivel": best["chave"][0] == 0}
        save_json(info, run_dir / RUN_FILE)
        print("\nmelhor: {} guidance {:.1f}".format(best["checkpoint"], best["guidance"]))
        for t in trail:
            print("  -", t)


if __name__ == "__main__":
    main()
