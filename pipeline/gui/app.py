"""Interface grafica para geracao interativa de lesoes sinteticas.

Inspirada na "MAM-E lesion drawing tool" do repositorio original (figura 5 do
artigo), com as extensoes pedidas neste projeto: canvas e saida em 1024x1024,
selecao da regiao por pincel ou caixa, validacao radiologica da regiao,
metricas de qualidade calculadas na hora e salvamento com metadados.

Execucao:
    python gui/app.py
    python gui/app.py --share          # link publico temporario
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2 as cv
import gradio as gr
import numpy as np

from mamelesion import Config, list_images, prepare_case, save_result
from mamelesion.generator import build_mask, random_box
from mamelesion.inpainting import LesionInpainter
from mamelesion.masks import LesionBox, binarize_mask, box_from_mask, validate_box
from mamelesion.quality import evaluate, passes_qc

CFG = Config.load()
INPAINTER = LesionInpainter(CFG)


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------


def _overlay_breast(canvas: np.ndarray, breast: np.ndarray) -> np.ndarray:
    """Fundo do editor: mamografia em RGB (o pincel desenha por cima)."""
    return np.stack([canvas] * 3, axis=-1)


def _mask_from_editor(editor: Optional[Dict[str, Any]], size: int) -> Optional[np.ndarray]:
    """Extrai a mascara pintada pelo usuario, tolerando variacoes da API."""
    if not editor:
        return None
    layers = editor.get("layers") or []
    accumulated = np.zeros((size, size), dtype=np.uint8)
    found = False
    for layer in layers:
        arr = np.asarray(layer)
        if arr.size == 0:
            continue
        if arr.ndim == 3 and arr.shape[2] == 4:
            painted = arr[..., 3]                       # canal alfa
        elif arr.ndim == 3:
            painted = arr.max(axis=2)
        else:
            painted = arr
        painted = cv.resize(painted.astype(np.uint8), (size, size),
                            interpolation=cv.INTER_NEAREST)
        accumulated = np.maximum(accumulated, painted)
        found = True
    if not found or accumulated.max() == 0:
        return None
    return binarize_mask(accumulated)


def _draw_box_preview(canvas: np.ndarray, box: LesionBox) -> np.ndarray:
    rgb = np.stack([canvas] * 3, axis=-1)
    cv.rectangle(rgb, (box.x1, box.y1), (box.x2, box.y2), (255, 176, 0), 3)
    return rgb


def _zoom(image: np.ndarray, box: LesionBox, pad_frac: float = 0.9) -> np.ndarray:
    pad = int(max(box.w, box.h) * pad_frac) + 24
    roi = box.expanded(pad, limit=image.shape[0])
    crop = image[roi.slices()]
    return cv.resize(crop, (512, 512), interpolation=cv.INTER_NEAREST)


def _metrics_markdown(ev: Dict[str, Any], params: Dict[str, Any]) -> str:
    m = ev["metrics"]
    ok, failed = passes_qc(ev, CFG.quality.qc_required)
    lines = [
        "### Metricas da regiao gerada",
        "",
        "| medida | valor | leitura |",
        "|---|---|---|",
        "| diametro equivalente | {:.1f} mm | tamanho da lesao segmentada |".format(
            m.get("equivalent_diameter_mm", float("nan"))),
        "| contraste de Weber | {:.3f} | densidade relativa ao tecido vizinho |".format(
            m.get("weber_contrast", float("nan"))),
        "| CNR | {:.2f} | contraste sobre ruido do tecido |".format(m.get("cnr", float("nan"))),
        "| circularidade | {:.2f} | 1.0 = circulo perfeito |".format(
            m.get("circularity", float("nan"))),
        "| ruido ROI / anel | {:.2f} | 1.0 = ruido igual ao tecido real |".format(
            m.get("noise_ratio_roi_ring", float("nan"))),
        "| nitidez ROI / anel | {:.2f} | 1.0 = nitidez igual ao tecido real |".format(
            m.get("sharpness_ratio", float("nan"))),
        "| excesso de emenda | {:+.3f} | 0 = sem degrau na borda |".format(
            m.get("seam_gradient_excess", float("nan"))),
        "| fora da regiao alterado | {:.1e} | 0 = restante da imagem intacto |".format(
            m.get("outside_changed_fraction", float("nan"))),
        "",
        "**Criterios de plausibilidade:** {}/{}".format(
            ev["plausibility"]["n_passed"], ev["plausibility"]["n_total"]),
        "",
        ("**Controle de qualidade: aprovado**" if ok else
         "**Controle de qualidade: reprovado** ({}) - gere novamente com outra semente"
         .format(", ".join(failed))),
        "",
        "_semente {} | guidance {} | passos {} | ganho {}_".format(
            params.get("seed"), params.get("guidance_scale"),
            params.get("num_inference_steps"), params.get("lesion_gain")),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def load_image(filename: str):
    """Carrega a mamografia escolhida e prepara o editor."""
    if not filename:
        return None, None, "Selecione uma imagem.", None, None
    case = prepare_case(Path(filename), CFG)
    background = _overlay_breast(case.canvas, case.breast)
    info = ("**{}** — canvas {}x{} px · escala estimada {:.4f} mm/px · lado {} · "
            "mama ocupa {:.0f}% do canvas").format(
        case.name, case.size, case.size, case.mm_per_px, case.side,
        100 * (case.breast > 0).mean())
    return case, {"background": background, "layers": [], "composite": background}, info, None, None


def suggest_region(case, lesion_type: str):
    """Sorteia uma regiao valida para o tipo de lesao (atalho para o usuario)."""
    if case is None:
        return None, "Carregue uma imagem primeiro.", None
    box = random_box(case, CFG, lesion_type)
    if box is None:
        return None, "Nao foi possivel encontrar regiao valida nesta mama.", None
    preview = _draw_box_preview(case.canvas, box)
    msg = "Regiao sugerida: {}x{} px (~{:.1f} mm). Use *Gerar* ou desenhe a sua.".format(
        box.w, box.h, np.sqrt(box.w * box.h) * case.mm_per_px)
    return box, msg, {"background": preview, "layers": [], "composite": preview}


def generate(case, editor, suggested_box, lesion_type, guidance, steps, seed_value,
             use_seed, detail_transfer, lesion_gain, blend, use_bbox, prompt):
    """Gera a lesao na regiao marcada e devolve imagem, zoom e metricas."""
    if case is None:
        return None, None, "Carregue uma imagem primeiro.", None

    painted = _mask_from_editor(editor, case.size)
    box: Optional[LesionBox] = None
    mask: Optional[np.ndarray] = None

    if painted is not None:
        box = box_from_mask(painted)
        if box is None:
            return None, None, "Nao consegui interpretar a regiao desenhada.", None
        if not use_bbox:
            mask = painted
    elif suggested_box is not None:
        box = suggested_box
    else:
        return None, None, ("Desenhe a regiao da lesao com o pincel ou use "
                            "*Sugerir regiao*."), None

    from dataclasses import replace
    cfg = Config.load()
    cfg.composition = replace(cfg.composition, blend=blend)
    mask_built, box = build_mask(case, box, cfg)
    if mask is None:
        mask = mask_built

    ok, reason = validate_box(box, case.breast, cfg.lesion.min_coverage, canvas=case.size)
    if not ok:
        return None, None, "Regiao invalida: {}".format(reason), None

    inpainter = INPAINTER
    inpainter.cfg.composition = cfg.composition
    try:
        result = inpainter.generate(
            case.canvas, mask, box, lesion_type=lesion_type,
            seed=int(seed_value) if use_seed else None,
            steps=int(steps), guidance=float(guidance), prompt=prompt or None,
            detail_transfer=float(detail_transfer), lesion_gain=float(lesion_gain))
    except Exception as exc:  # erro de VRAM, modelo ausente, etc.
        traceback.print_exc()
        return None, None, "Falha na geracao: {}".format(exc), None

    result.params.update({"case": case.name, "box_canvas": box.to_dict(),
                          "lesion_diameter_mm": round(float(np.sqrt(box.w * box.h) * case.mm_per_px), 2),
                          "mm_per_px_canvas": case.mm_per_px})
    ev = evaluate(result, case, cfg.quality, lesion_type)

    side_by_side = np.hstack([_zoom(result.original, box), _zoom(result.composed, box)])
    return (result.composed, side_by_side, _metrics_markdown(ev, result.params), result)


def save_current(case, result):
    if case is None or result is None:
        return "Nada para salvar: gere uma lesao primeiro."
    paths = save_result(result, case, CFG)
    return "Salvo em **{}** (1024x1024), com mascara, metadados e painel comparativo.".format(
        paths["image"])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    images = list_images(CFG.resolve_input_dir())
    choices = [(p.name, str(p)) for p in images]
    presets = CFG.lesion.types

    with gr.Blocks(title="MAM-E - lesoes sinteticas em mamografia") as demo:
        gr.Markdown(
            "# MAM-E · geracao de lesoes sinteticas em mamografias saudaveis\n"
            "Stable Diffusion v1-5 com DreamBooth para inpainting de lesoes "
            "(*Montoya-del-Angel et al., arXiv:2311.09822*). "
            "Desenhe a regiao com o pincel, gere a lesao e salve em 1024x1024."
        )

        case_state = gr.State(None)
        box_state = gr.State(None)
        result_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                image_picker = gr.Dropdown(choices=choices, label="Mamografia saudavel",
                                           value=choices[0][1] if choices else None)
                info = gr.Markdown("")
                editor = gr.ImageEditor(
                    label="Marque a regiao da lesao (pincel)",
                    type="numpy", image_mode="RGB", height=560,
                    brush=gr.Brush(colors=["#ff3860"], default_size=28),
                    layers=True, transforms=[],
                )
                with gr.Row():
                    suggest_btn = gr.Button("Sugerir regiao", variant="secondary")
                    generate_btn = gr.Button("Gerar lesao", variant="primary")
                    save_btn = gr.Button("Salvar 1024x1024")

            with gr.Column(scale=2):
                lesion_type = gr.Radio(
                    choices=list(presets), value=CFG.lesion.lesion_type,
                    label="Tipo de lesao",
                    info=" · ".join("{}: {}".format(k, v.description) for k, v in presets.items()))
                with gr.Accordion("Parametros de difusao", open=True):
                    guidance = gr.Slider(1.0, 10.0, value=CFG.lesion.preset().guidance_scale,
                                         step=0.5, label="Guidance scale",
                                         info="quanto o modelo se prende ao prompt")
                    steps = gr.Slider(10, 60, value=CFG.diffusion.num_inference_steps, step=1,
                                      label="Passos de difusao")
                    with gr.Row():
                        use_seed = gr.Checkbox(value=True, label="Usar semente")
                        seed_value = gr.Number(value=CFG.diffusion.seed, precision=0,
                                               label="Semente")
                with gr.Accordion("Composicao em 1024x1024", open=False):
                    lesion_gain = gr.Slider(0.2, 1.5, value=1.0, step=0.05,
                                            label="Ganho da lesao",
                                            info="<1 deixa a densidade mais sutil")
                    detail_transfer = gr.Slider(0.0, 1.0, value=CFG.composition.detail_transfer,
                                                step=0.05, label="Transferencia de textura",
                                                info="devolve a granulacao do detector na regiao")
                    blend = gr.Radio(["feather", "hard", "poisson"],
                                     value=CFG.composition.blend, label="Mistura da borda")
                    use_bbox = gr.Checkbox(
                        value=True, label="Usar caixa envolvente do desenho",
                        info="o modelo foi treinado com mascaras retangulares")
                    prompt = gr.Textbox(value=CFG.lesion.prompt, label="Prompt",
                                        info="o ajuste fino usou um unico prompt generico")
                status = gr.Markdown("")

        with gr.Row():
            output_image = gr.Image(label="Mamografia sintetica (1024x1024)",
                                    type="numpy", height=560)
            with gr.Column():
                zoom_view = gr.Image(label="Zoom: original (esq.) vs sintetica (dir.)",
                                     type="numpy", height=280)
                metrics_md = gr.Markdown("")

        image_picker.change(load_image, [image_picker],
                            [case_state, editor, info, output_image, zoom_view])
        demo.load(load_image, [image_picker],
                  [case_state, editor, info, output_image, zoom_view])
        suggest_btn.click(suggest_region, [case_state, lesion_type],
                          [box_state, status, editor])
        lesion_type.change(lambda t: gr.update(value=CFG.lesion.preset(t).guidance_scale),
                           [lesion_type], [guidance])
        generate_btn.click(
            generate,
            [case_state, editor, box_state, lesion_type, guidance, steps, seed_value,
             use_seed, detail_transfer, lesion_gain, blend, use_bbox, prompt],
            [output_image, zoom_view, metrics_md, result_state])
        save_btn.click(save_current, [case_state, result_state], [status])

    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=None,
                        help="pasta com as mamografias a carregar (PNG, JPEG ou DICOM)")
    parser.add_argument("--output-dir", default=None,
                        help="onde salvar as imagens geradas")
    parser.add_argument("--share", action="store_true", help="cria link publico temporario")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.input_dir:
        CFG.paths.input_dir = args.input_dir
    if args.output_dir:
        CFG.paths.output_dir = args.output_dir

    input_dir = CFG.resolve_input_dir()
    if not input_dir.is_dir():
        print("pasta de entrada nao encontrada: {}".format(input_dir))
        return 1
    found = list_images(input_dir)
    if not found:
        print("nenhuma imagem em {} (formatos aceitos: PNG, JPEG, DICOM, TIFF)"
              .format(input_dir))
        return 1
    print("{} imagens em {}".format(len(found), input_dir))
    print("saida: {}".format(CFG.resolve_output_dir()))

    demo = build_ui()
    demo.queue()
    demo.launch(server_port=args.port, share=args.share, inbrowser=not args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
