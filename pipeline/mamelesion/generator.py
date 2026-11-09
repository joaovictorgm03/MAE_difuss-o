"""Orquestracao: imagem saudavel -> caso preparado -> lesao sintetica -> disco."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

from .config import Config
from .inpainting import GenerationResult, LesionInpainter
from .masks import LesionBox, mask_from_box, sample_box, snap_to_latent, validate_box
from .preprocessing import (breast_mask, breast_side, build_canvas, estimate_mm_per_px,
                            read_gray, Framing)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".tif", ".tiff"}


@dataclass
class Case:
    """Uma mamografia saudavel pronta para receber lesoes."""

    name: str
    path: Path
    canvas: np.ndarray          # uint8, canvas_size x canvas_size
    framing: Framing
    breast: np.ndarray          # mascara da mama no canvas (0/255)
    mm_per_px: float
    side: str
    dicom_spacing: Optional[float] = None

    @property
    def size(self) -> int:
        return int(self.canvas.shape[0])

    def meta(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": str(self.path),
            "framing": self.framing.to_dict(),
            "mm_per_px_canvas": self.mm_per_px,
            "breast_side": self.side,
            "breast_area_px": int((self.breast > 0).sum()),
            "dicom_pixel_spacing_mm": self.dicom_spacing,
        }


def list_images(directory: str | Path) -> List[Path]:
    directory = Path(directory)
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def prepare_case(path: str | Path, cfg: Config) -> Case:
    """Aplica o pre-processamento do artigo e devolve o caso pronto."""
    path = Path(path)
    gray, spacing = read_gray(path)
    pre = cfg.preprocessing
    canvas, framing = build_canvas(gray, pre.canvas_size, pre.framing, pre.max_value)
    breast = breast_mask(canvas, pre.breast_threshold)
    mm_per_px = pre.mm_per_px or estimate_mm_per_px(framing, pre.detector_long_side_mm, spacing)
    return Case(path.stem, path, canvas, framing, breast, float(mm_per_px),
                breast_side(breast), spacing)


def build_mask(case: Case, box: LesionBox, cfg: Config) -> Tuple[np.ndarray, LesionBox]:
    """Alinha a caixa a grade latente (opcional) e devolve (mascara, caixa)."""
    if cfg.lesion.snap_to_latent:
        box = snap_to_latent(box, cfg.preprocessing.canvas_size, cfg.preprocessing.model_resolution)
    mask = mask_from_box(box, case.size, cfg.lesion.mask_shape)
    return mask, box


def random_box(case: Case, cfg: Config, lesion_type: Optional[str] = None,
               rng: Optional[np.random.Generator] = None,
               tries: int = 40) -> Optional[LesionBox]:
    """Sorteia uma caixa plausivel para o tipo de lesao escolhido.

    O alinhamento a grade latente expande levemente a caixa, o que pode empurrar
    a borda para fora do tecido; por isso a validacao acontece *depois* do
    alinhamento, e nao antes.
    """
    preset = cfg.lesion.preset(lesion_type)
    for _ in range(tries):
        box = sample_box(case.breast, case.mm_per_px, preset.diameter_mm_range,
                         preset.aspect_range, rng=rng, canvas=case.size)
        if box is None:
            return None
        if cfg.lesion.snap_to_latent:
            box = snap_to_latent(box, cfg.preprocessing.canvas_size,
                                 cfg.preprocessing.model_resolution)
        ok, _ = validate_box(box, case.breast, cfg.lesion.min_coverage, canvas=case.size)
        if ok:
            return box
    return None


def synthesize(
    case: Case,
    inpainter: LesionInpainter,
    box: LesionBox,
    cfg: Config,
    lesion_type: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> GenerationResult:
    """Valida a regiao, monta a mascara e gera a lesao."""
    mask, box = build_mask(case, box, cfg)
    ok, reason = validate_box(box, case.breast, cfg.lesion.min_coverage, canvas=case.size)
    if not ok:
        raise ValueError(reason)
    result = inpainter.generate(case.canvas, mask, box, lesion_type=lesion_type,
                                seed=seed, **kwargs)
    result.params.update({
        "case": case.name,
        "box_canvas": box.to_dict(),
        "lesion_diameter_mm": round(float(np.sqrt(box.w * box.h) * case.mm_per_px), 2),
        "mm_per_px_canvas": case.mm_per_px,
    })
    return result


# ---------------------------------------------------------------------------
# Escrita em disco
# ---------------------------------------------------------------------------


def comparison_figure(result: GenerationResult, pad: int = 96) -> np.ndarray:
    """Painel original | sintetica | diferenca, com zoom na ROI."""
    box = result.box
    orig, synth = result.original, result.composed
    diff = np.clip(np.abs(result.difference) * 4, 0, 255).astype(np.uint8)

    def annotate(img: np.ndarray) -> np.ndarray:
        rgb = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        if box is not None:
            cv.rectangle(rgb, (box.x1, box.y1), (box.x2, box.y2), (0, 200, 255), 3)
        return rgb

    panels = [annotate(orig), annotate(synth), cv.cvtColor(diff, cv.COLOR_GRAY2BGR)]
    top = np.hstack(panels)

    if box is not None:
        roi = box.expanded(pad, limit=orig.shape[0])
        crops = [cv.resize(img[roi.slices()], (top.shape[1] // 3, top.shape[1] // 3),
                           interpolation=cv.INTER_NEAREST)
                 for img in (orig, synth, np.clip(np.abs(result.difference) * 4, 0, 255).astype(np.uint8))]
        bottom = np.hstack([cv.cvtColor(c, cv.COLOR_GRAY2BGR) for c in crops])
        h = min(top.shape[1], bottom.shape[1])
        top = top[:, :h]
        bottom = bottom[:, :h]
        return np.vstack([top, bottom])
    return top


def save_result(result: GenerationResult, case: Case, cfg: Config,
                out_dir: Optional[Path] = None, tag: Optional[str] = None) -> Dict[str, str]:
    """Salva imagem 1024x1024, mascara, metadados e painel comparativo."""
    out_dir = Path(out_dir or cfg.resolve_output_dir())
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (out_dir / "comparisons").mkdir(parents=True, exist_ok=True)

    stamp = tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    lesion_type = result.params.get("lesion_type", "lesion")
    stem = "{}_{}_{}".format(case.name, lesion_type, stamp)

    size = cfg.output.size
    image = result.composed
    if image.shape[0] != size:
        image = cv.resize(image, (size, size), interpolation=cv.INTER_LANCZOS4)
    if not cfg.output.save_grayscale:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    paths: Dict[str, str] = {}
    image_path = out_dir / "images" / (stem + ".png")
    cv.imwrite(str(image_path), image)
    paths["image"] = str(image_path)

    if cfg.output.save_mask:
        mask = result.mask
        if mask.shape[0] != size:
            mask = cv.resize(mask, (size, size), interpolation=cv.INTER_NEAREST)
        mask_path = out_dir / "masks" / (stem + "_mask.png")
        cv.imwrite(str(mask_path), mask)
        paths["mask"] = str(mask_path)

    if cfg.output.save_comparison:
        cmp_path = out_dir / "comparisons" / (stem + "_cmp.png")
        cv.imwrite(str(cmp_path), comparison_figure(result))
        paths["comparison"] = str(cmp_path)

    if cfg.output.save_metadata:
        meta = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                "case": case.meta(), "generation": result.params,
                "output_size": size, "files": paths}
        meta_path = out_dir / "metadata" / (stem + ".json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        paths["metadata"] = str(meta_path)

    return paths


__all__ = ["Case", "prepare_case", "list_images", "build_mask", "random_box",
           "synthesize", "save_result", "comparison_figure"]
