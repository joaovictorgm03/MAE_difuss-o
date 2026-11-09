#!/usr/bin/env python
"""Percorre UMA imagem por todas as etapas do pipeline, gravando cada estagio.

Serve para acompanhar manualmente o que acontece entre a mamografia saudavel e
o arquivo final: cada etapa vira um PNG numerado e uma linha na tabela impressa,
com o numero que importa naquele ponto.

As funcoes chamadas aqui sao **as mesmas** de ``mamelesion/pipeline.py``, na
mesma ordem -- este script nao reimplementa nada. Se um estagio parecer errado
aqui, ele esta errado no lote tambem.

Uso::

    python scripts/inspect_lesion.py ../sem_anomalia/6278_IM-0005_CCD_R.png
    python scripts/inspect_lesion.py <img> --type nodule --seed 42
    python scripts/inspect_lesion.py <img> --box 400,700,480,780   # posicao fixa
    python scripts/inspect_lesion.py <img> --etapas                # so a tabela

Saida em ``--out`` (padrao: ``outputs/inspecao/<nome_da_imagem>/``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamelesion import quality as qmod                                   # noqa: E402
from mamelesion.anatomy import (AnatomyConfig, plausibility_map,          # noqa: E402
                                sample_anatomical_box, describe_region)
from mamelesion.compose import compose, feather_alpha                     # noqa: E402
from mamelesion.config import Config                                      # noqa: E402
from mamelesion.generator import Case                                     # noqa: E402
from mamelesion.inpainting import LesionInpainter                         # noqa: E402
from mamelesion.masks import LesionBox, mask_from_box, snap_to_latent     # noqa: E402
from mamelesion.native import (build_breast_canvas, lesion_delta,         # noqa: E402
                               native_texture_metrics, read_source,
                               save_image, writeback_native)
from mamelesion.pipeline import parse_view                                # noqa: E402
from mamelesion.preprocessing import downscale, estimate_mm_per_px        # noqa: E402


# ---------------------------------------------------------------------------
# Escrita dos estagios
# ---------------------------------------------------------------------------


class Stages:
    """Numera, grava e descreve cada estagio."""

    def __init__(self, out_dir: Path, enabled: bool = True) -> None:
        self.out_dir = out_dir
        self.enabled = enabled
        self.n = 0
        self.rows: list[tuple[str, str, str]] = []
        if enabled:
            out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, image: np.ndarray | None, note: str) -> None:
        self.n += 1
        filename = "--"
        if image is not None and self.enabled:
            filename = "{:02d}_{}.png".format(self.n, name)
            save_image(self.out_dir / filename, image)
        self.rows.append(("{:02d}".format(self.n), name, note))
        print("  {:>2}. {:<26} {}".format(self.n, name, note))

    def table(self) -> None:
        print()
        print("  {} estagio(s) gravado(s) em {}".format(self.n, self.out_dir))


def heat(values: np.ndarray, mask: np.ndarray | None = None,
         colormap: int = cv.COLORMAP_VIRIDIS) -> np.ndarray:
    """Mapa float -> imagem colorida, normalizado pelo proprio intervalo."""
    v = values.astype(np.float32)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    norm = np.zeros_like(v) if hi <= lo else (v - lo) / (hi - lo)
    out = cv.applyColorMap((norm * 255).astype(np.uint8), colormap)
    if mask is not None:
        out[mask == 0] = 0
    return out


def signed_heat(values: np.ndarray, clip: float | None = None) -> np.ndarray:
    """Mapa divergente para um delta com sinal: azul negativo, vermelho positivo.

    Zero fica exatamente no cinza medio e a escala e simetrica, senao um delta
    inteiramente positivo e um inteiramente negativo sairiam com a mesma
    aparencia -- que e justamente a distincao que interessa olhar aqui.
    """
    v = values.astype(np.float32)
    limit = float(clip if clip else max(abs(float(v.min())), abs(float(v.max())), 1e-6))
    norm = np.clip(v / limit, -1.0, 1.0)
    out = np.full(v.shape + (3,), 128, dtype=np.float32)
    pos, neg = norm > 0, norm < 0
    out[pos] = np.stack([128 - 128 * norm[pos], 128 - 128 * norm[pos],
                         128 + 127 * norm[pos]], axis=-1)      # BGR: vermelho
    out[neg] = np.stack([128 - 127 * norm[neg], 128 + 128 * norm[neg],
                         128 + 128 * norm[neg]], axis=-1)      # BGR: azul
    return np.clip(out, 0, 255).astype(np.uint8)


def outline(gray: np.ndarray, box: LesionBox | None = None,
            extra: np.ndarray | None = None) -> np.ndarray:
    """Cinza -> BGR com a caixa desenhada e uma mascara opcional em vermelho."""
    rgb = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    if extra is not None:
        rgb[extra > 0] = (0.45 * rgb[extra > 0] + 0.55 * np.array([0, 0, 220])).astype(np.uint8)
    if box is not None:
        cv.rectangle(rgb, (box.x1, box.y1), (box.x2, box.y2), (0, 220, 255), 3)
    return rgb


def side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dois recortes lado a lado, com a MESMA janela de intensidade.

    Normalizar cada um pelo proprio intervalo faria a lesao sumir: ela desloca o
    maximo do recorte, e cada metade acabaria com um mapeamento diferente.
    """
    ref = a.astype(np.float32)
    lo, hi = np.percentile(ref[ref > 0], 1) if (ref > 0).any() else 0, np.percentile(ref, 99.5)
    scale = lambda z: np.clip((z.astype(np.float32) - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    sep = np.full((a.shape[0], 6), 255, np.uint8)
    return np.hstack([scale(a), sep, scale(b)])


# ---------------------------------------------------------------------------
# Percurso
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("imagem", help="mamografia saudavel de entrada")
    parser.add_argument("--out", default=None, help="pasta de saida")
    parser.add_argument("--type", dest="lesion_type", default="nodule",
                        help="tipo de lesao (padrao: nodule)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--box", default=None,
                        help="caixa fixa no canvas: x1,y1,x2,y2 (pula a amostragem)")
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--etapas", action="store_true",
                        help="so imprime a tabela, nao grava imagens")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    path = Path(args.imagem).expanduser().resolve()
    if not path.is_file():
        print("erro: {} nao encontrado".format(path), file=sys.stderr)
        return 2

    cfg = Config.load(args.config)
    anat_cfg = AnatomyConfig()
    out_dir = Path(args.out).expanduser() if args.out else \
        REPO_ROOT / "outputs" / "inspecao" / path.stem
    st = Stages(out_dir, enabled=not args.etapas)
    size = args.canvas_size

    print()
    print("Inspecao de {}".format(path.name))
    print("=" * 72)

    # -- leitura ------------------------------------------------------------
    src = read_source(path, cfg.preprocessing.max_value)
    st.add("origem", None, "{}x{}, {} bits, valores {}-{}".format(
        src.raw.shape[1], src.raw.shape[0], src.bit_depth,
        int(src.raw.min()), int(src.raw.max())))
    st.add("uint8", src.gray_u8, "escalonado por {:.1f}x para [0,255]".format(
        src.scale_to_u8 or 1.0))

    # -- enquadramento ------------------------------------------------------
    canvas, framing, breast = build_breast_canvas(
        src.gray_u8, size, breast_threshold=cfg.preprocessing.breast_threshold)
    st.add("canvas", canvas, "recorte pela mama -> {}x{}, escala {:.3f}".format(
        size, size, framing.scale))
    st.add("mascara_mama", breast, "tecido cobre {:.1%} do canvas".format(
        float((breast > 0).sum()) / (size * size)))

    mm_per_px = cfg.preprocessing.mm_per_px or estimate_mm_per_px(
        framing, cfg.preprocessing.detector_long_side_mm, src.pixel_spacing_mm)
    view = parse_view(path.stem)

    # -- plausibilidade anatomica -------------------------------------------
    pmap = plausibility_map(canvas, breast, float(mm_per_px), view.view, anat_cfg)
    st.add("distancia_pele", heat(pmap.skin_distance_mm, breast),
           "max {:.0f} mm; corte em {:.0f} mm".format(
               float(pmap.skin_distance_mm.max()), anat_cfg.min_skin_distance_mm))
    st.add("peitoral", outline(canvas, extra=pmap.pectoral),
           "{}, {:.1%} da mama".format(pmap.pectoral_method,
                                       pmap.stats["pectoral_area_frac"]))
    st.add("posto_densidade", heat(pmap.density_rank, breast),
           "gordura abaixo do percentil {:.0f}".format(anat_cfg.density_percentile))
    st.add("score_plausibilidade", heat(pmap.score, breast, cv.COLORMAP_INFERNO),
           "regiao valida = {:.1%} da mama".format(pmap.valid_fraction()))

    # -- posicao da lesao ---------------------------------------------------
    rng = np.random.default_rng(args.seed)
    preset = cfg.lesion.preset(args.lesion_type)
    if args.box:
        box = LesionBox(*[int(v) for v in args.box.split(",")])
        anat = {}
        origem_caixa = "caixa fixa pelo usuario"
    else:
        sampled = sample_anatomical_box(pmap, float(mm_per_px), preset.diameter_mm_range,
                                        preset.aspect_range, anat_cfg, rng, canvas=size)
        if sampled is None:
            print("\n  nenhuma caixa plausivel para '{}' nesta imagem".format(args.lesion_type))
            return 1
        box, anat = sampled
        origem_caixa = "amostrada ~ score, {:.0f}-{:.0f} mm".format(*preset.diameter_mm_range)

    if cfg.lesion.snap_to_latent:
        box = snap_to_latent(box, size, cfg.preprocessing.model_resolution)
    mask = mask_from_box(box, size, cfg.lesion.mask_shape)

    st.add("caixa", outline(canvas, box), "{}; {}x{} px".format(origem_caixa, box.w, box.h))
    st.add("mascara_lesao", mask, "retangulo alinhado a grade latente de 16 px")

    # -- o que o modelo ve --------------------------------------------------
    res = cfg.preprocessing.model_resolution
    entrada = downscale(canvas, res).copy()
    entrada[downscale(mask, res) > 127] = 0
    st.add("entrada_modelo", entrada, "{}x{} com a regiao apagada".format(res, res))

    # -- difusao ------------------------------------------------------------
    inpainter = LesionInpainter(cfg)
    result = inpainter.generate(canvas, mask, box, lesion_type=args.lesion_type, seed=args.seed)
    st.add("saida_modelo", result.generated_model_res,
           "prompt '{}', {} passos, guidance {}".format(
               result.params["prompt"], result.params["num_inference_steps"],
               result.params["guidance_scale"]))

    # -- composicao no canvas ----------------------------------------------
    from copy import deepcopy
    comp = deepcopy(cfg.composition)
    comp.detail_transfer = preset.detail_transfer
    comp.lesion_gain = preset.lesion_gain

    # O modelo devolve os 512x512 inteiros: o VAE altera a imagem toda, nao so a
    # regiao mascarada. Ver o delta bruto antes do alpha e o que torna obvio por
    # que o alpha existe.
    from mamelesion.compose import upscale
    bruto = upscale(result.generated_model_res, size).astype(np.float32) - canvas.astype(np.float32)
    fora = float((np.abs(bruto[mask <= 127]) > 0.5).mean())
    st.add("diferenca_bruta", signed_heat(bruto),
           "saida do modelo - original: mexe em {:.1%} dos pixels FORA da ROI".format(fora))

    alpha = feather_alpha(mask, comp.feather_px, box)
    largura = int(min(comp.feather_px, 0.25 * min(box.w, box.h)))
    st.add("alpha", (alpha * 255).astype(np.uint8),
           "peso da mistura: 1 no centro, 0 na borda; transicao de {} px".format(largura))

    diff = result.difference
    st.add("diferenca_aplicada", signed_heat(diff),
           "apos o alpha: media na ROI {:+.1f}, fora da ROI {:.1%} alterado".format(
               diff[mask > 127].mean(), float((np.abs(diff[mask <= 127]) > 0.5).mean())))

    result.composed = compose(canvas, result.generated_model_res, mask, comp, box)
    st.add("composto_canvas", result.composed,
           "hibrido; detail_transfer {:.2f}, ganho {:.2f}".format(
               comp.detail_transfer, comp.lesion_gain))

    # -- volta a resolucao original ----------------------------------------
    delta = lesion_delta(result, comp)
    st.add("delta_lesao", signed_heat(delta),
           "sem transferencia de detalhe; e o que vai para a resolucao original")

    native, geom, mask_native = writeback_native(src, framing, delta, box, result.mask)
    pad = max(40, (geom["y2"] - geom["y1"]) // 2)
    y0, y1 = max(0, geom["y1"] - pad), min(native.shape[0], geom["y2"] + pad)
    x0, x1 = max(0, geom["x1"] - pad), min(native.shape[1], geom["x2"] + pad)
    st.add("nativo_antes_depois", side_by_side(src.raw[y0:y1, x0:x1], native[y0:y1, x0:x1]),
           "recorte {}x{} na resolucao original".format(x1 - x0, y1 - y0))

    mudou = int((src.raw != native).sum())
    fora = int(((src.raw != native) & (mask_native <= 127)).sum())
    st.add("saida_nativa", None, "{} px alterados, {} fora da mascara".format(mudou, fora))

    # -- medidas ------------------------------------------------------------
    case = Case(path.stem, path, canvas, framing, breast, float(mm_per_px),
                pmap.side, src.pixel_spacing_mm)
    ev = qmod.evaluate(result, case, cfg.quality, args.lesion_type)
    passou, falhou = qmod.passes_qc(ev, cfg.quality.qc_required)
    m = ev["metrics"]
    nat = native_texture_metrics(native, src.raw, geom, mask_native,
                                 background_level=cfg.preprocessing.breast_threshold * src.scale_to_u8)

    print()
    print("Medidas")
    print("-" * 72)
    print("  regiao anatomica       {}".format(describe_region(box, pmap, view.view)))
    if anat:
        print("  distancia da pele      {:.1f} mm (minimo na caixa)".format(
            anat.get("skin_distance_mm_min", float("nan"))))
        print("  posto de densidade     {:.0f}/100".format(anat.get("density_rank_mean", 0)))
        print("  score anatomico        {:.3f}".format(anat.get("anat_score_mean", 0)))
    print("  diametro segmentado    {:.1f} mm".format(m.get("equivalent_diameter_mm", 0)))
    print("  contraste de Weber     {:+.3f} (canvas)   {:+.3f} (nativo)".format(
        m.get("weber_contrast", 0), nat.get("native_weber_contrast", float("nan"))))
    print("  CNR                    {:.2f} (canvas)    {:.2f} (nativo)".format(
        m.get("cnr", 0), nat.get("native_cnr", float("nan"))))
    print("  saldo de densidade     {:+.4f} (>0.01 exigido pelo QC)".format(
        m.get("roi_delta_ratio", 0)))
    print("  ruido ROI/tecido       {:.3f} (canvas)    {:.3f} (nativo)".format(
        m.get("noise_ratio_roi_ring", 0), nat.get("native_noise_ratio_roi_ring", float("nan"))))
    print("  emenda                 {:+.3f} (0 = borda indistinguivel)".format(
        m.get("seam_gradient_excess", 0)))
    print("  alterado fora da ROI   {:.2e} (canvas)".format(
        m.get("outside_changed_fraction", 0)))
    print()
    print("  QC: {}".format("APROVADO" if passou else "REPROVADO em " + ", ".join(falhou)))
    print("  criterios aprovados: {}/{}".format(
        ev["plausibility"]["n_passed"], ev["plausibility"]["n_total"]))

    reprovados = [c for c in ev["plausibility"]["checks"] if not c["passed"]]
    if reprovados:
        print()
        for c in reprovados:
            print("    x {:<28} {:.3f} fora de [{}, {}]".format(
                c["criterion"], c["value"], c["range"][0], c["range"][1]))

    st.table()
    inpainter.unload()
    return 0 if passou else 1


if __name__ == "__main__":
    raise SystemExit(main())
