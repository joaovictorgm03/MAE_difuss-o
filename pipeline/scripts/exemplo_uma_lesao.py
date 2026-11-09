#!/usr/bin/env python
"""Gera UMA lesao sintetica, passo a passo, para entender o pipeline.

Este script nao reimplementa nada: chama, na mesma ordem, as funcoes que o
gerador de producao (``mamelesion/pipeline.py``, metodo ``_one_lesion_native``)
usa. Cada passo imprime o que calculou e grava uma figura em ``--saida``.

    .venv/bin/python scripts/exemplo_uma_lesao.py ../birads_1/10047_32948_L_CC_1.png
    .venv/bin/python scripts/exemplo_uma_lesao.py ../birads_1/10047_32948_L_CC_1.png --seed 7 --guidance 4

O que fica de fora (roda depois, sobre o lote inteiro, em scripts separados):
calibracao de contraste (calibrar_contraste.py), auditoria de 31 criterios
(auditar_dataset.py) e fidelidade as lesoes reais (fidelidade_real.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mamelesion.anatomy import AnatomyConfig, plausibility_map          # noqa: E402
from mamelesion.config import Config                                    # noqa: E402
from mamelesion.finetune import load_run                                # noqa: E402
from mamelesion.inpainting import LesionInpainter                       # noqa: E402
from mamelesion.native import build_breast_canvas, read_source, save_image  # noqa: E402
from mamelesion.nativegen import evaluate_window, generate_native       # noqa: E402
from mamelesion.pipeline import load_view_table, resolve_view           # noqa: E402
from mamelesion.preprocessing import estimate_mm_per_px                 # noqa: E402
from mamelesion.window import anatomical_center                         # noqa: E402


def u8(img: np.ndarray) -> np.ndarray:
    """Para exibir: estica entre os percentis 0,5 e 99,5."""
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imagem", help="mamografia saudavel (PNG 16 bits ou DICOM)")
    ap.add_argument("--saida", default="outputs/exemplo_uma_lesao")
    ap.add_argument("--seed", type=int, default=1234, help="semente: outra semente = outra lesao")
    ap.add_argument("--tipo", default="birads5", choices=("birads5", "birads4"))
    ap.add_argument("--guidance", type=float, default=None, help="padrao: a do treino do LoRA (2,5)")
    ap.add_argument("--lora", default="treinos/birads_lora_v1")
    ap.add_argument("--incidencias", default="../mamas_com_lesao/lote_birads1_v1/_controle/incidencias.csv")
    a = ap.parse_args()
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ------------------------------------------------------------------ 0
    # Configuracao. O LoRA grava no treino.json a geometria da janela em que foi
    # treinado (contexto 2x, reducao minima 0,75, margem da mascara 10%...); ela
    # PREVALECE sobre o config.yaml, senao o modelo rodaria fora do que aprendeu.
    cfg = Config.load(None)
    run = load_run(REPO_ROOT / a.lora)
    run.apply_geometry(cfg.native_window)
    nw = cfg.native_window
    guidance = a.guidance if a.guidance is not None else float(run.config["melhor_guidance"])
    preset = cfg.lesion.preset(a.tipo)
    print("[0] LoRA {} | guidance {} | {} passos | prompt: {!r}".format(
        run.lora_dir.name, guidance, preset.num_inference_steps, preset.prompt))

    # ------------------------------------------------------------------ 1
    # Leitura. ``raw`` e a imagem como esta no arquivo (16 bits); ``gray_u8`` e
    # uma copia em 8 bits so para as contas de anatomia. A lesao sera somada a
    # ``raw``, na profundidade original.
    path = Path(a.imagem)
    src = read_source(path)
    print("[1] {} | {}x{} px | {} bits | PixelSpacing: {}".format(
        path.name, src.raw.shape[1], src.raw.shape[0], src.bit_depth, src.pixel_spacing_mm))

    # ------------------------------------------------------------------ 2
    # Enquadramento + mapa de plausibilidade. A mama e recortada e levada a um
    # canvas de 1024 px; sobre ele, cada pixel ganha um score em [0, 1]:
    # distancia da pele x fora do peitoral x densidade x vizinhanca glandular.
    canvas, framing, mama = build_breast_canvas(src.gray_u8, 1024,
                                                breast_threshold=cfg.preprocessing.breast_threshold)
    mm_canvas = estimate_mm_per_px(framing, cfg.preprocessing.detector_long_side_mm, src.pixel_spacing_mm)
    tabela = load_view_table(Path(a.incidencias)) if Path(a.incidencias).is_file() else None
    view = resolve_view(path.stem, tabela)            # a tabela corrige L_CC_1 -> MLO
    anat_cfg = AnatomyConfig(density_upper_percentile=60, min_glandular_fraction=0.2, min_box_score=0.0)
    pmap = plausibility_map(canvas, mama, float(mm_canvas), view.view, anat_cfg)
    print("[2] incidencia {} ({}) | peitoral: {} | {:.0%} da mama pode receber lesao".format(
        view.view, view.source, pmap.pectoral_method, pmap.valid_fraction()))
    mapa = cv.applyColorMap((pmap.score * 255).astype(np.uint8), cv.COLORMAP_VIRIDIS)
    mapa[pmap.pectoral > 0] = (0, 0, 200)
    cv.imwrite(str(saida / "2_mapa_plausibilidade.png"), np.hstack([cv.cvtColor(canvas, cv.COLOR_GRAY2BGR), mapa]))

    # ------------------------------------------------------------------ 3
    # Tamanho e posicao. O tamanho (px nativos) e sorteado do catalogo de lesoes
    # REAIS do treino; o centro, proporcional ao mapa, com a caixa inteira na
    # mama, longe da pele e fora do peitoral.
    tamanhos = run.size_catalog[a.tipo]
    wh = tuple(int(v) for v in tamanhos[int(rng.integers(len(tamanhos)))])
    centro, anat = anatomical_center(pmap, framing, float(mm_canvas), wh, rng, anat_cfg, allow_fallback=False)
    print("[3] lesao de {}x{} px, centro em ({:.0f}, {:.0f}) px nativos, nivel '{}'".format(
        wh[0], wh[1], centro[0], centro[1], anat["placement"]))

    # ------------------------------------------------------------------ 4
    # Difusao + composicao (nativegen.generate_native):
    #   a) recorta a janela (2x a lesao) da imagem original e leva a 512 px;
    #   b) mascara = retangulo da lesao + 10%, alinhado a grade de 8 px do latente;
    #   c) SD 1.5 inpainting + LoRA: VAE -> latente 64x64 -> 40 passos de remocao
    #      de ruido (DPM-Solver, guidance) -> VAE de volta;
    #   d) delta = saida - original, com textura nativa na margem, piso de
    #      densidade (a lesao nao escurece o tecido) e transicao suave;
    #   e) delta reamostrado a resolucao nativa e SOMADO a imagem original,
    #      zerado fora da mascara.
    inp = LesionInpainter(cfg, lora_path=str(run.lora_dir), lora_scale=nw.lora_scale)
    ref = src.scale_to_u8 * 255.0
    les = generate_native(inp.generate_window, src.raw, ref, centro, wh, nw, preset.prompt, a.seed,
                          preset.num_inference_steps, guidance, preset.lesion_gain,
                          None, cfg.preprocessing.model_resolution)
    j = les.window
    print("[4] janela de {} px nativos -> {} px no modelo (escala {:.2f}); mascara {}x{} px no modelo".format(
        j.side, j.model_res, j.scale, les.box.w, les.box.h))
    delta = les.composed_u8.astype(np.float32) - les.original_u8.astype(np.float32)
    dmax = max(1.0, float(np.abs(delta).max()))
    dcor = np.zeros(delta.shape + (3,), np.uint8)             # BGR: vermelho = somou, azul = tirou
    dcor[..., 2] = np.clip(255 * delta / dmax, 0, 255).astype(np.uint8)
    dcor[..., 0] = np.clip(-255 * delta / dmax, 0, 255).astype(np.uint8)
    tiras = [cv.cvtColor(x, cv.COLOR_GRAY2BGR) for x in
             (les.original_u8, les.mask_u8, les.generated_u8, les.composed_u8)] + [dcor]
    rot = ["a) janela original", "b) mascara", "c) saida do modelo", "d) composto", "e) delta (vermelho = +)"]
    for t, r in zip(tiras, rot):
        cv.putText(t, r, (6, 22), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv.imwrite(str(saida / "4_estagios_da_difusao.png"), np.hstack(tiras))

    # ------------------------------------------------------------------ 5
    # Portao de QC da geracao: contraste, CNR e saldo de densidade na janela.
    # Na producao, se reprova, gera de novo com outra semente (ate 3) e depois
    # em outra posicao (ate 4).
    mm_nativo = float(mm_canvas) * framing.scale
    ev = evaluate_window(les, mm_nativo, cfg.quality, a.tipo, cfg.preprocessing.breast_threshold, path.stem)
    m = ev["metrics"]
    print("[5] QC {} | Weber {:.2f} | CNR {:.2f} | saldo de densidade {:.3f} {}".format(
        "APROVADA" if ev["qc_ok"] else "REPROVADA", m.get("weber_contrast", float("nan")),
        m.get("cnr", float("nan")), m.get("roi_delta_ratio", float("nan")),
        "" if ev["qc_ok"] else "-> falhou em " + ", ".join(ev["qc_falhas"])))

    # ------------------------------------------------------------------ 6
    # Gravacao na resolucao e bits originais, e a garantia central conferida.
    save_image(saida / (path.stem + "_com_lesao.png"), les.raw_out)
    save_image(saida / (path.stem + "_mascara.png"), les.mask_native)
    fora = int(((les.raw_out != src.raw) & (les.mask_native == 0)).sum())
    dentro = int(((les.raw_out != src.raw) & (les.mask_native > 0)).sum())
    print("[6] gravado em {} | pixels alterados: {} dentro da mascara, {} fora (tem de ser 0)".format(
        saida, dentro, fora))
    x0, y0 = int(centro[0] - 1.5 * max(wh)), int(centro[1] - 1.5 * max(wh))
    x0, y0 = max(0, x0), max(0, y0)
    lado = 3 * max(wh)
    antes, depois = src.raw[y0:y0 + lado, x0:x0 + lado], les.raw_out[y0:y0 + lado, x0:x0 + lado]
    lo, hi = np.percentile(antes, [0.5, 99.5])                  # MESMO janelamento nos dois
    f = lambda z: np.clip((z.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    cv.imwrite(str(saida / "6_antes_depois.png"), np.hstack([f(antes), np.full((antes.shape[0], 8), 255, np.uint8), f(depois)]))
    return 0 if fora == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
