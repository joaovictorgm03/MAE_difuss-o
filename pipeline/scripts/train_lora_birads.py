#!/usr/bin/env python
"""Fase 3: ajuste fino LoRA do checkpoint MAM-E sobre recortes reais BI-RADS 4/5.

Cada amostra e um recorte de treino colado, com borda suave, numa janela em
resolucao nativa de uma mamografia saudavel de treino (``mamelesion.paste``); o
UNet aprende a regenerar a lesao sob a mascara. So os adaptadores LoRA das
camadas de atencao sao treinados; VAE e codificador de texto ficam congelados.

A cada ``--validar-cada`` passos (e no passo 0, que e o modelo base, porque o
LoRA comeca com saida nula) o script gera lesoes para os recortes de
**validacao**, em hospedeiros de validacao, e mede KID, memorizacao e textura
contra os recortes reais (``mamelesion.lesion_eval``). O checkpoint com menor
KID sem sinal de memorizacao vira ``melhor_checkpoint`` em ``treino.json``. O
conjunto de teste nao e tocado aqui -- ele e da fase 4.

Saida em ``--run``::

    treino.json                 hiperparametros, geometria, ambiente, checkpoints, melhor
    split.json, catalogo_tamanhos.json   copias dos dados usados
    log.jsonl                   perda por passo
    checkpoints/passo_000000/   validacao do modelo base (sem pesos)
    checkpoints/passo_NNNNNN/   pytorch_lora_weights.safetensors, validacao.json,
                                grade.png, estado_treino.pt (para retomar)

Exemplos
--------
    # teste rapido de ponta a ponta (minutos)
    .venv/bin/python scripts/train_lora_birads.py --run treinos/fumaca --passos 40 \\
        --validar-cada 20 --val-sementes 1 --val-limite 4

    # treino completo
    .venv/bin/python scripts/train_lora_birads.py --run treinos/birads_lora_v1

    # retomar do ultimo checkpoint
    .venv/bin/python scripts/train_lora_birads.py --run treinos/birads_lora_v1 --retomar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2 as cv                                                        # noqa: E402
import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402
import torch.nn.functional as F                                         # noqa: E402
from PIL import Image                                                   # noqa: E402

from mamelesion.birads_data import (CLASS_PROMPTS, LesionCrop, load_json, parse_ids,  # noqa: E402
                                    read_crop, save_json)
from mamelesion.config import Config                                    # noqa: E402
from mamelesion.finetune import (LORA_WEIGHTS, RUN_FILE, PasteDataset, PromptPolicy,  # noqa: E402
                                 add_lora, collate, save_lora)
from mamelesion.inpainting import resolve_model_path, run_inpaint       # noqa: E402
from mamelesion.lesion_eval import (DinoFeatures, comparison_grid, plan_jobs,  # noqa: E402
                                    select_best, selection_key, validation_summary)
from mamelesion.nativegen import generate_native                        # noqa: E402
from mamelesion.paste import PasteConfig, prepare_host                  # noqa: E402
from mamelesion.window import anatomical_center                         # noqa: E402

VAL_BASE_SEED = 777


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", default="treinos/dados_birads", help="saida de prepare_birads_data.py")
    ap.add_argument("--run", required=True, help="pasta do treino (criada; nao sobrescreve)")
    ap.add_argument("--retomar", action="store_true", help="continua do ultimo checkpoint em --run")
    g = ap.add_argument_group("otimizacao")
    g.add_argument("--passos", type=int, default=4000)
    g.add_argument("--lote", type=int, default=4)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--aquecimento", type=int, default=200, help="passos de aquecimento do lr")
    g.add_argument("--rank", type=int, default=16)
    g.add_argument("--alpha", type=int, default=16)
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--peso-fora", type=float, default=0.10,
                   help="peso da perda fora da mascara (dentro = 1). O contexto e visivel ao modelo")
    g.add_argument("--min-snr", type=float, default=5.0, help="gamma do Min-SNR; 0 desliga")
    g.add_argument("--seed", type=int, default=1337)
    g.add_argument("--workers", type=int, default=6)
    d = ap.add_argument_group("amostras")
    d.add_argument("--p-birads4", type=float, default=0.30,
                   help="probabilidade de sortear BI-RADS 4 (natural: 12/59 = 0,20)")
    d.add_argument("--p-vazio", type=float, default=0.10)
    d.add_argument("--p-generico", type=float, default=0.20)
    v = ap.add_argument_group("validacao")
    v.add_argument("--validar-cada", type=int, default=500)
    v.add_argument("--val-sementes", type=int, default=2, help="lesoes por recorte de validacao")
    v.add_argument("--val-passos", type=int, default=30, help="passos de difusao na validacao")
    v.add_argument("--val-guidance", type=float, default=4.0)
    v.add_argument("--val-limite", type=int, default=0, help="usa so os N primeiros recortes (teste)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_crops(split, names):
    out = []
    for n in names:
        p = Path(split["caminho_de"][n])
        patient, study = parse_ids(n)
        out.append(LesionCrop(n, p, split["classe_de"][n], patient, study, read_crop(p)))
    return out


def worker_init(_):
    cv.setNumThreads(1)
    torch.set_num_threads(1)


class Validator:
    """Gera lesoes para os recortes de validacao e mede contra os reais."""

    def __init__(self, args, cfg, val_crops, train_crops, val_hosts, device):
        self.args, self.cfg, self.device = args, cfg, device
        crops = val_crops[:args.val_limite] if args.val_limite else val_crops
        self.crops = crops
        self.hosts = val_hosts
        self.jobs = plan_jobs([(c.name, c.lesion_class, c.wh) for c in crops], len(val_hosts),
                              args.val_sementes, VAL_BASE_SEED)
        self.dino = DinoFeatures(device=device)
        self.real_feats = self.dino([c.image for c in crops])
        self.train_feats = self.dino([c.image for c in train_crops])

    def run(self, pipe, out_dir: Path, step: int):
        started = time.time()
        nw = self.cfg.native_window
        gen = lambda im, m, prompt, seed, steps, guid, neg: run_inpaint(      # noqa: E731
            pipe, im, m, prompt, seed, steps, guid, neg, self.device)
        synth, by_ref, placements = [], {}, []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for job in self.jobs:
                host = self.hosts[job.host_index]
                center, info = anatomical_center(host.pmap, host.framing, host.mm_per_px_canvas, job.wh,
                                                 np.random.default_rng(job.place_seed))
                placements.append(info["placement"])
                nl = generate_native(gen, host.raw, host.ref, center, job.wh, nw, CLASS_PROMPTS[job.lesion_class],
                                     job.seed, self.args.val_passos, self.args.val_guidance)
                crop = nl.lesion_crop(host.ref)
                synth.append(crop)
                by_ref.setdefault(job.reference, []).append(crop)
        summary = validation_summary(self.real_feats, self.dino(synth), self.train_feats,
                                     [c.image for c in self.crops], synth)
        result = {
            "passo": step,
            **summary,
            "chave_selecao": list(selection_key(summary)),
            "n_recortes_reais": len(self.crops),
            "n_sinteticos": len(synth),
            "posicionamento_fallback": placements.count("fallback"),
            "duracao_s": round(time.time() - started, 1),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(result, out_dir / "validacao.json")
        rows = [[c.image] + by_ref[c.name] for c in self.crops[:12]]
        header = ["real"] + ["sintetico s{}".format(s) for s in range(self.args.val_sementes)]
        Image.fromarray(comparison_grid(rows, header)).save(out_dir / "grade.png")
        return result


def build_pipe(path, vae, text_encoder, tokenizer, unet, noise_scheduler):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_config(noise_scheduler.config),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def choose_best(run_dir: Path, info: dict) -> None:
    """Aplica ``lesion_eval.select_best`` aos checkpoints com pesos; o passo 0 so e referencia.

    Criterio definitivo: ``scripts/select_checkpoint.py``, que usa mais sementes e
    varre a guidance. Este e o provisorio, durante o treino.
    """
    rows = [ck for ck in info["checkpoints"] if ck.get("validacao") and ck["passo"] != 0
            and "auc_classificador" in ck["validacao"]]
    if not rows:
        return
    i, trail = select_best([ck["validacao"] for ck in rows])
    info["melhor_checkpoint"] = rows[i]["pasta"]
    info["melhor_criterio"] = {"origem": "train_lora_birads.py (provisorio)", "trilha": trail}


def main(argv=None) -> int:
    args = parse_args(argv)
    device = "cuda"
    if not torch.cuda.is_available():
        print("erro: treino exige GPU CUDA", file=sys.stderr)
        return 2

    run_dir = Path(args.run)
    dados = Path(args.dados)
    if (run_dir / RUN_FILE).exists() and not args.retomar:
        print("erro: {} ja existe. Use outro --run ou --retomar.".format(run_dir), file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.load()
    nw = cfg.native_window
    paste_cfg = PasteConfig.from_native_window(nw, cfg.preprocessing.model_resolution)
    policy = PromptPolicy(p_empty=args.p_vazio, p_generic=args.p_generico)
    class_probs = {"birads4": args.p_birads4, "birads5": 1.0 - args.p_birads4}

    split = load_json(dados / "split.json")
    model_path = resolve_model_path(cfg.paths.model_id, cfg.paths.model_local_path)

    # --- estado do treino ----------------------------------------------------------
    start_step = 0
    resume_state = None
    if args.retomar:
        info = load_json(run_dir / RUN_FILE)
        cks = [ck for ck in info["checkpoints"] if (run_dir / ck["pasta"] / "estado_treino.pt").exists()]
        if not cks:
            print("erro: nenhum checkpoint com estado para retomar", file=sys.stderr)
            return 2
        last = max(cks, key=lambda ck: ck["passo"])
        resume_state = torch.load(run_dir / last["pasta"] / "estado_treino.pt", map_location="cpu",
                                  weights_only=True)
        start_step = int(resume_state["passo"])
        if file_hash(dados / "split.json") != info["dados"]["split_sha256"]:
            print("erro: split.json mudou desde o inicio do treino", file=sys.stderr)
            return 2
        print("retomando do passo", start_step)
    else:
        from mamelesion.pipeline import environment_info
        shutil.copy2(dados / "split.json", run_dir / "split.json")
        shutil.copy2(dados / "catalogo_tamanhos.json", run_dir / "catalogo_tamanhos.json")
        info = {
            "criado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
            "argumentos": vars(args),
            "geometria": {k: getattr(nw, k) for k in ("context_factor", "min_scale", "min_context",
                                                      "mask_margin_frac", "feather_frac", "paste_feather_frac")},
            "amostras": {**asdict(paste_cfg), "probabilidade_classe": class_probs, "prompts": asdict(policy)},
            "prompts_classe": CLASS_PROMPTS,
            "modelo_base": {"id": cfg.paths.model_id, "caminho": str(model_path)},
            "dados": {"pasta": str(dados.resolve()), "split_sha256": file_hash(dados / "split.json"),
                      "treino": len(split["recortes"]["train"]), "validacao": len(split["recortes"]["val"]),
                      "hospedeiros_treino": split["hospedeiros"]["train"],
                      "hospedeiros_validacao": split["hospedeiros"]["val"]},
            "ambiente": environment_info(),
            "checkpoints": [],
            "melhor_checkpoint": None,
        }
        save_json(info, run_dir / RUN_FILE)

    # --- dados ---------------------------------------------------------------------
    t0 = time.time()
    train_crops = load_crops(split, split["recortes"]["train"])
    val_crops = load_crops(split, split["recortes"]["val"])
    hosts_dir = Path(split["hospedeiros_pasta"])
    train_hosts = [prepare_host(hosts_dir / n, cfg) for n in split["hospedeiros"]["train"]]
    val_hosts = [prepare_host(hosts_dir / n, cfg) for n in split["hospedeiros"]["val"]]
    print("dados: {} recortes treino, {} validacao, {}+{} hospedeiros ({:.0f} s)".format(
        len(train_crops), len(val_crops), len(train_hosts), len(val_hosts), time.time() - t0))

    total_samples = args.passos * args.lote
    dataset = PasteDataset(train_crops, train_hosts, paste_cfg, policy, class_probs, total_samples,
                           args.seed, start_index=start_step * args.lote)
    # "spawn", nao "fork": neste ponto o processo ja tem dezenas de threads (pools do
    # OpenCV/OpenMP da preparacao dos hospedeiros). Um fork herda locks presos por
    # essas threads e os workers travam para sempre no primeiro cv.resize -- foi o
    # que aconteceu no primeiro teste (workers com 1 thread em futex_wait, GPU a 0%).
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.lote, shuffle=False, num_workers=args.workers, collate_fn=collate,
        pin_memory=True, drop_last=True, worker_init_fn=worker_init,
        multiprocessing_context="spawn" if args.workers > 0 else None,
        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)

    # --- modelos ---------------------------------------------------------------------
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params, compute_snr
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    torch.manual_seed(args.seed)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",
                                                 torch_dtype=torch.float16).to(device).eval()
    text_encoder.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    vae.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet",
                                                torch_dtype=torch.bfloat16).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    params = add_lora(unet, args.rank, args.alpha, args.dropout)
    cast_training_params(unet, dtype=torch.float32)
    params = [p for p in unet.parameters() if p.requires_grad]
    unet.enable_gradient_checkpointing()
    n_params = sum(p.numel() for p in params)
    print("LoRA: {:.2f} M parametros treinaveis (rank {})".format(n_params / 1e6, args.rank))

    with torch.no_grad():
        embeds = {}
        for prompt in policy.prompts():
            ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt").input_ids.to(device)
            embeds[prompt] = text_encoder(ids)[0][0].to(torch.bfloat16)

    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=args.aquecimento,
                                 num_training_steps=args.passos)
    if resume_state is not None:
        set_peft_model_state_dict(unet, resume_state["lora"])
        optimizer.load_state_dict(resume_state["otimizador"])
        lr_scheduler.load_state_dict(resume_state["agendador"])

    validator = Validator(args, cfg, val_crops, train_crops, val_hosts, device)
    pipe = build_pipe(model_path, vae, text_encoder, tokenizer, unet, noise_scheduler)
    sf = vae.config.scaling_factor
    prediction = noise_scheduler.config.prediction_type

    def checkpoint(step: int, save_weights: bool) -> None:
        folder = "checkpoints/passo_{:06d}".format(step)
        ck_dir = run_dir / folder
        if save_weights:
            save_lora(unet, ck_dir)
            torch.save({"passo": step, "lora": get_peft_model_state_dict(unet),
                        "otimizador": optimizer.state_dict(), "agendador": lr_scheduler.state_dict()},
                       ck_dir / "estado_treino.pt")
        unet.eval()
        result = validator.run(pipe, ck_dir, step)
        unet.train()
        info["checkpoints"] = [ck for ck in info["checkpoints"] if ck["passo"] != step]
        info["checkpoints"].append({"passo": step, "pasta": folder, "tem_pesos": save_weights,
                                    "validacao": {k: result[k] for k in (
                                        "auc_classificador", "kid", "razao_contraste", "cosseno_centroides",
                                        "memorizacao", "textura_mediana_w1", "chave_selecao")}})
        info["checkpoints"].sort(key=lambda ck: ck["passo"])
        choose_best(run_dir, info)
        save_json(info, run_dir / RUN_FILE)
        print("[validacao passo {}] AUC real x sint. {:.2f} | contraste {:.2f} | KID {:.3f} | memorizacao {:.2f} "
              "| textura W1 {:.2f} | {:.0f} s | melhor: {}".format(
                  step, result["auc_classificador"]["media"], result["razao_contraste"], result["kid"]["media"],
                  result["memorizacao"]["fracao_acima"], result["textura_mediana_w1"],
                  result["duracao_s"], info.get("melhor_checkpoint")), flush=True)

    if start_step == 0 and not any(ck["passo"] == 0 for ck in info["checkpoints"]):
        checkpoint(0, save_weights=False)

    # --- laco --------------------------------------------------------------------------
    unet.train()
    log = open(run_dir / "log.jsonl", "a", encoding="utf-8")
    step = start_step
    window_loss, t_last = [], time.time()
    fallback = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        fallback += batch["posicionamento"].count("fallback")
        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample() * sf
            masked = pixels * (mask < 0.5)                    # como o StableDiffusionInpaintPipeline
            masked_latents = vae.encode(masked).latent_dist.sample() * sf
            mask_lat = F.interpolate(mask, size=latents.shape[-2:])
            cond = torch.stack([embeds[p] for p in batch["prompt"]])

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        model_in = torch.cat([noisy, mask_lat, masked_latents], dim=1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(model_in.to(torch.bfloat16), timesteps, cond).sample
        target = noise if prediction == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)

        # perda concentrada na lesao: a margem de 1 pixel latente cobre a transicao
        weight = args.peso_fora + (1 - args.peso_fora) * F.max_pool2d(mask_lat, 3, 1, 1)
        err = ((pred.float() - target.float()) ** 2).mean(1, keepdim=True)
        per_sample = (err * weight).sum((1, 2, 3)) / weight.sum((1, 2, 3))
        if args.min_snr > 0:
            snr = compute_snr(noise_scheduler, timesteps)
            w = torch.clamp(snr, max=args.min_snr) / (snr if prediction == "epsilon" else snr + 1)
            per_sample = per_sample * w
        loss = per_sample.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        value = float(loss.detach())
        if not math.isfinite(value):
            print("erro: perda nao finita no passo {}".format(step), file=sys.stderr)
            return 1
        window_loss.append(value)
        log.write(json.dumps({"passo": step, "perda": value, "lr": lr_scheduler.get_last_lr()[0]}) + "\n")
        if step % 25 == 0 or step == args.passos:
            dt = time.time() - t_last
            print("passo {:5d}/{} | perda {:.4f} (media 25: {:.4f}) | lr {:.2e} | {:.2f} it/s | "
                  "GPU {:.1f} GB | fallback {}".format(
                      step, args.passos, value, float(np.mean(window_loss)), lr_scheduler.get_last_lr()[0],
                      len(window_loss) / max(dt, 1e-6), torch.cuda.max_memory_allocated() / 2 ** 30, fallback),
                  flush=True)
            log.flush()
            window_loss, t_last = [], time.time()
        if step % args.validar_cada == 0 or step == args.passos:
            checkpoint(step, save_weights=True)
            t_last = time.time()
        if step >= args.passos:
            break
    log.close()

    info["concluido_em"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(info, run_dir / RUN_FILE)
    crit = info.get("melhor_criterio", {})
    print("\nmelhor checkpoint: {} | {}".format(info.get("melhor_checkpoint"), crit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
