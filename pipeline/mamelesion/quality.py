"""Testes de qualidade de imagem sobre a regiao onde a lesao foi gerada.

Tres blocos de metricas, todos calculados na ROI da lesao e comparados contra um
anel de tecido real ao redor dela (controle pareado, mesma mama e mesma
densidade local):

1. Realismo do sinal da lesao - contraste, CNR, textura GLCM, forma e margem,
   confrontados com as faixas esperadas para massas/nodulos/densidades.
2. Qualidade da regiao reconstruida - artefatos de emenda, ruido, nitidez,
   consistencia de histograma.
3. Fidelidade fora da regiao - garante que nada alem da lesao foi alterado, o
   que e o requisito para usar o par (real, sintetico) como aumento de dataset.

Todas as funcoes recebem arrays uint8 no canvas e devolvem dicionarios planos,
faceis de agregar em CSV/relatorio.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
from scipy import stats as sps
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .config import QualityConfig
from .masks import LesionBox

EPS = 1e-8


# ---------------------------------------------------------------------------
# Regioes de analise
# ---------------------------------------------------------------------------


def ring_mask(mask: np.ndarray, breast: np.ndarray, ring_px: int = 40) -> np.ndarray:
    """Anel de tecido real imediatamente ao redor da regiao inpainted.

    E o controle pareado das metricas: mesma paciente, mesma vizinhanca
    anatomica, mas pixels 100% reais.
    """
    binary = (mask > 127).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ring_px * 2 + 1,) * 2)
    dilated = cv.dilate(binary, kernel)
    ring = (dilated > 0) & (binary == 0) & (breast > 0)
    return ring.astype(bool)


def segment_lesion(original: np.ndarray, composed: np.ndarray, mask: np.ndarray,
                   min_frac: float = 0.05) -> np.ndarray:
    """Segmenta a lesao gerada a partir do mapa de diferenca.

    A lesao e a parte da ROI que ficou mais densa que o tecido original. Usa
    limiar de Otsu sobre a diferenca positiva dentro da mascara e mantem a maior
    componente conexa.
    """
    binary = (mask > 127)
    diff = composed.astype(np.float32) - original.astype(np.float32)
    diff[~binary] = 0
    positive = np.clip(diff, 0, None)
    if positive.max() < 1.0:
        return np.zeros_like(binary)

    values = positive[binary]
    scaled = np.clip(positive / (positive.max() + EPS) * 255, 0, 255).astype(np.uint8)
    inside = scaled[binary]
    if inside.std() < 1:
        return np.zeros_like(binary)
    thresh, _ = cv.threshold(inside.reshape(-1, 1), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    lesion = ((scaled >= thresh) & binary).astype(np.uint8)

    # limpeza morfologica: fecha buracos de ruido antes de escolher a componente
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    lesion = cv.morphologyEx(lesion, cv.MORPH_CLOSE, kernel)
    lesion = cv.morphologyEx(lesion, cv.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv.connectedComponentsWithStats(lesion, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary)
    largest = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    out = (labels == largest).astype(np.uint8)
    contours, _ = cv.findContours(out, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(out, contours, -1, 1, thickness=cv.FILLED)   # preenche buracos
    out = out.astype(bool)

    if out.sum() < min_frac * binary.sum() and (values > 0).any():
        # lesao difusa: usa toda a area com diferenca acima da mediana positiva
        out = binary & (positive > np.median(values[values > 0]))
    return out.astype(bool)


# ---------------------------------------------------------------------------
# Bloco 1 - realismo do sinal da lesao
# ---------------------------------------------------------------------------


def intensity_metrics(composed: np.ndarray, original: np.ndarray, lesion: np.ndarray,
                      ring: np.ndarray, roi: np.ndarray) -> Dict[str, float]:
    """Contraste e relacao sinal-ruido da lesao contra o tecido vizinho real."""
    les = composed[lesion].astype(np.float32) if lesion.any() else composed[roi].astype(np.float32)
    bg = composed[ring].astype(np.float32)
    orig_roi = original[roi].astype(np.float32)

    mu_l, sd_l = float(les.mean()), float(les.std())
    mu_b, sd_b = float(bg.mean()), float(bg.std())

    michelson = (mu_l - mu_b) / (mu_l + mu_b + EPS)
    weber = (mu_l - mu_b) / (mu_b + EPS)
    cnr = (mu_l - mu_b) / (sd_b + EPS)
    snr = mu_l / (sd_l + EPS)

    # Saldo liquido de densidade em TODA a ROI, nao apenas na lesao segmentada.
    #
    # ``segment_lesion`` isola a parte da ROI que ficou mais clara, entao
    # ``weber_contrast`` acha contraste positivo mesmo quando o modelo, no
    # conjunto, *escureceu* a regiao -- a falha de reconstrucao em que nenhuma
    # lesao e inserida. Medido no lote de 66 imagens, isso acontecia em 10 a 15
    # geracoes por lote sem ser detectado. Este saldo, relativo ao nivel do
    # tecido original, distingue os dois casos e por isso entra no portao de QC.
    roi_delta = composed[roi].astype(np.float32) - original[roi].astype(np.float32)
    roi_original_level = float(original[roi].astype(np.float32).mean())

    return {
        "roi_delta_mean": float(roi_delta.mean()),
        "roi_delta_ratio": float(roi_delta.mean() / (roi_original_level + EPS)),
        "roi_positive_fraction": float((roi_delta > 0).mean()),
        "lesion_mean": mu_l,
        "lesion_std": sd_l,
        "ring_mean": mu_b,
        "ring_std": sd_b,
        "roi_original_mean": float(orig_roi.mean()),
        "delta_mean_vs_original": mu_l - float(original[lesion].mean() if lesion.any() else orig_roi.mean()),
        "michelson_contrast": float(michelson),
        "weber_contrast": float(weber),
        "cnr": float(cnr),
        "snr": float(snr),
        "rms_contrast": float(sd_l / (mu_l + EPS)),
    }


def _quantize(img: np.ndarray, levels: int) -> np.ndarray:
    return (img.astype(np.float32) / 256.0 * levels).astype(np.uint8).clip(0, levels - 1)


def glcm_features(img: np.ndarray, region: np.ndarray, distances: List[int],
                  levels: int, prefix: str = "") -> Dict[str, float]:
    """Descritores de Haralick medios na regiao (media sobre 4 angulos)."""
    ys, xs = np.nonzero(region)
    if len(xs) < 32:
        return {}
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = _quantize(img[y1:y2, x1:x2], levels)
    if min(patch.shape) <= max(distances):
        return {}
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(patch, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    out = {}
    for prop in ("contrast", "homogeneity", "energy", "correlation", "dissimilarity"):
        out[prefix + "glcm_" + prop] = float(np.mean(graycoprops(glcm, prop)))
    p = glcm / (glcm.sum(axis=(0, 1), keepdims=True) + EPS)
    out[prefix + "glcm_entropy"] = float(np.mean(-np.sum(p * np.log2(p + EPS), axis=(0, 1))))
    return out


def shape_metrics(lesion: np.ndarray, composed: np.ndarray, mm_per_px: float) -> Dict[str, float]:
    """Tamanho, circularidade, solidez e nitidez de margem da lesao segmentada."""
    if not lesion.any():
        return {"lesion_area_mm2": 0.0, "equivalent_diameter_mm": 0.0,
                "circularity": 0.0, "solidity": 0.0, "margin_gradient": 0.0,
                "margin_sharpness_ratio": 0.0, "spiculation_index": 0.0}

    binary = lesion.astype(np.uint8)
    area_px = int(binary.sum())
    area_mm2 = area_px * (mm_per_px ** 2)
    eq_diam_mm = 2.0 * np.sqrt(area_px / np.pi) * mm_per_px

    # O perimetro medido sobre a escada de pixels e superestimado em ~20-40%, o
    # que derruba artificialmente a circularidade. Suaviza-se o contorno antes de
    # medir (borramento + relimiarizacao), tecnica padrao em morfometria.
    smooth = cv.GaussianBlur(binary.astype(np.float32) * 255, (9, 9), 2.0)
    smooth = (smooth > 127).astype(np.uint8)
    if smooth.sum() < 8:
        smooth = binary
    contours, _ = cv.findContours(smooth, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv.contourArea)
    perimeter = cv.arcLength(contour, True)
    smooth_area = float(cv.contourArea(contour))
    circularity = float(4 * np.pi * smooth_area / (perimeter ** 2 + EPS))
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    # area e casco vindos do mesmo contorno suavizado, para que solidez <= 1
    solidity = float(min(1.0, smooth_area / (hull_area + EPS)))

    grad = cv.magnitude(cv.Sobel(composed.astype(np.float32), cv.CV_32F, 1, 0, ksize=3),
                        cv.Sobel(composed.astype(np.float32), cv.CV_32F, 0, 1, ksize=3))
    border = cv.dilate(binary, np.ones((3, 3), np.uint8)) - cv.erode(binary, np.ones((3, 3), np.uint8))
    border = border.astype(bool)
    inner = cv.erode(binary, np.ones((7, 7), np.uint8)).astype(bool)
    margin_gradient = float(grad[border].mean()) if border.any() else 0.0
    inner_gradient = float(grad[inner].mean()) if inner.any() else EPS
    sharpness_ratio = float(margin_gradient / (inner_gradient + EPS))

    # indice de espiculacao: variabilidade do raio ao longo do contorno
    pts = contour.reshape(-1, 2).astype(np.float32)
    center = pts.mean(axis=0)
    radii = np.linalg.norm(pts - center, axis=1)
    spiculation = float(radii.std() / (radii.mean() + EPS))

    return {
        "lesion_area_mm2": float(area_mm2),
        "equivalent_diameter_mm": float(eq_diam_mm),
        "circularity": circularity,
        "solidity": solidity,
        "margin_gradient": margin_gradient,
        "margin_sharpness_ratio": sharpness_ratio,
        "spiculation_index": spiculation,
    }


# ---------------------------------------------------------------------------
# Bloco 2 - qualidade da regiao reconstruida
# ---------------------------------------------------------------------------


def seam_metrics(composed: np.ndarray, original: np.ndarray, mask: np.ndarray,
                 band_px: int = 4) -> Dict[str, float]:
    """Deteccao de emenda no contorno da regiao inpainted.

    Compara a energia de gradiente na faixa que cruza a borda da mascara com a
    energia em faixas de controle deslocadas para dentro e para fora. Um degrau
    visivel eleva a razao bem acima de 1. Tambem mede o degrau de intensidade
    medio entre os dois lados imediatos da borda.
    """
    binary = (mask > 127).astype(np.uint8)
    kernel = lambda k: cv.getStructuringElement(cv.MORPH_ELLIPSE, (k * 2 + 1,) * 2)

    boundary = (cv.dilate(binary, kernel(band_px)) - cv.erode(binary, kernel(band_px))).astype(bool)
    outer_ctrl = (cv.dilate(binary, kernel(band_px * 4)) - cv.dilate(binary, kernel(band_px * 2))).astype(bool)
    inner_ctrl = (cv.erode(binary, kernel(band_px * 2)) - cv.erode(binary, kernel(band_px * 4))).astype(bool)

    def grad_energy(img: np.ndarray, region: np.ndarray) -> float:
        if not region.any():
            return float("nan")
        gx = cv.Sobel(img.astype(np.float32), cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(img.astype(np.float32), cv.CV_32F, 0, 1, ksize=3)
        return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)[region]))

    ctrl_regions = outer_ctrl | inner_ctrl
    g_boundary = grad_energy(composed, boundary)
    g_ctrl = grad_energy(composed, ctrl_regions)
    g_boundary_orig = grad_energy(original, boundary)
    g_ctrl_orig = grad_energy(original, ctrl_regions)

    just_in = (binary.astype(bool) & cv.dilate(binary, kernel(1)).astype(bool)
               & ~cv.erode(binary, kernel(band_px)).astype(bool))
    just_out = (cv.dilate(binary, kernel(band_px)).astype(bool) & (binary == 0))
    step = float(composed[just_in].mean() - composed[just_out].mean()) if just_in.any() and just_out.any() else 0.0
    step_orig = float(original[just_in].mean() - original[just_out].mean()) if just_in.any() and just_out.any() else 0.0

    return {
        "seam_gradient_ratio": float(g_boundary / (g_ctrl + EPS)),
        "seam_gradient_ratio_original": float(g_boundary_orig / (g_ctrl_orig + EPS)),
        "seam_gradient_excess": float(g_boundary / (g_ctrl + EPS) - g_boundary_orig / (g_ctrl_orig + EPS)),
        "border_step": step,
        "border_step_original": step_orig,
        "border_step_excess": float(abs(step) - abs(step_orig)),
    }


def _highpass(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    k = int(max(3, round(sigma * 6) | 1))
    return img.astype(np.float32) - cv.GaussianBlur(img.astype(np.float32), (k, k), sigma)


def noise_metrics(composed: np.ndarray, original: np.ndarray, roi: np.ndarray,
                  ring: np.ndarray) -> Dict[str, float]:
    """Consistencia de ruido e nitidez entre regiao gerada e tecido real."""
    hp_comp = _highpass(composed)
    hp_orig = _highpass(original)
    lap = cv.Laplacian(composed.astype(np.float32), cv.CV_32F)
    lap_orig = cv.Laplacian(original.astype(np.float32), cv.CV_32F)

    noise_roi = float(hp_comp[roi].std())
    noise_ring = float(hp_comp[ring].std())
    noise_roi_orig = float(hp_orig[roi].std())

    return {
        "noise_std_roi": noise_roi,
        "noise_std_ring": noise_ring,
        "noise_ratio_roi_ring": float(noise_roi / (noise_ring + EPS)),
        "noise_retention_vs_original": float(noise_roi / (noise_roi_orig + EPS)),
        "sharpness_roi": float(lap[roi].var()),
        "sharpness_ring": float(lap[ring].var()),
        "sharpness_ratio": float(lap[roi].var() / (lap[ring].var() + EPS)),
        "sharpness_retention_vs_original": float(lap[roi].var() / (lap_orig[roi].var() + EPS)),
    }


def radial_power_slope(img: np.ndarray, region_box: LesionBox) -> float:
    """Inclinacao log-log do espectro de potencia radial (proxy de NPS)."""
    patch = img[region_box.slices()].astype(np.float32)
    if min(patch.shape) < 16:
        return float("nan")
    n = min(patch.shape)
    patch = patch[:n, :n]
    patch = patch - patch.mean()
    window = np.outer(np.hanning(n), np.hanning(n))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(patch * window))) ** 2

    cy = cx = n // 2
    y, x = np.indices((n, n))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    radial = np.bincount(r.ravel(), spectrum.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    lo, hi = 2, min(len(radial) - 1, n // 2)
    if hi - lo < 4:
        return float("nan")
    freqs = np.arange(lo, hi)
    slope, _, _, _, _ = sps.linregress(np.log(freqs), np.log(radial[lo:hi] + EPS))
    return float(slope)


def histogram_metrics(composed: np.ndarray, original: np.ndarray, roi: np.ndarray,
                      ring: np.ndarray) -> Dict[str, float]:
    """Distancia entre a distribuicao de intensidades da ROI e do tecido real."""
    a = composed[roi].astype(np.float32)
    b = composed[ring].astype(np.float32)
    a0 = original[roi].astype(np.float32)
    return {
        "wasserstein_roi_ring": float(sps.wasserstein_distance(a, b)),
        "wasserstein_roi_ring_original": float(sps.wasserstein_distance(a0, b)),
        "ks_roi_ring": float(sps.ks_2samp(a, b).statistic),
        "skew_roi": float(sps.skew(a)),
        "skew_ring": float(sps.skew(b)),
        "kurtosis_roi": float(sps.kurtosis(a)),
        "kurtosis_ring": float(sps.kurtosis(b)),
    }


# ---------------------------------------------------------------------------
# Bloco 3 - fidelidade fora da regiao
# ---------------------------------------------------------------------------


def fidelity_metrics(composed: np.ndarray, original: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Verifica que apenas a regiao mascarada mudou."""
    outside = (mask <= 127)
    diff_outside = np.abs(composed.astype(np.float32) - original.astype(np.float32))[outside]
    changed = float((diff_outside > 0.5).mean())
    if changed == 0:
        psnr_out = float("inf")
    else:
        mse = float((diff_outside ** 2).mean())
        psnr_out = float(10 * np.log10(255.0 ** 2 / (mse + EPS)))
    ssim_full = float(structural_similarity(original, composed, data_range=255))
    return {
        "outside_changed_fraction": changed,
        "outside_max_abs_diff": float(diff_outside.max()),
        "outside_psnr_db": psnr_out,
        "ssim_full_image": ssim_full,
    }


# ---------------------------------------------------------------------------
# Regras de plausibilidade radiologica
# ---------------------------------------------------------------------------

#: faixas de referencia por tipo de lesao (diametro em mm segundo o preset,
#: contraste e margem segundo o comportamento esperado de massas em FFDM)
PLAUSIBILITY_RULES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "mass": {"equivalent_diameter_mm": (5.0, 40.0), "weber_contrast": (0.05, 1.50),
             "circularity": (0.35, 1.05), "cnr": (0.3, 12.0)},
    "nodule": {"equivalent_diameter_mm": (2.5, 18.0), "weber_contrast": (0.05, 1.80),
               "circularity": (0.45, 1.05), "cnr": (0.3, 15.0)},
    "density": {"equivalent_diameter_mm": (10.0, 70.0), "weber_contrast": (0.02, 1.00),
                "circularity": (0.20, 1.05), "cnr": (0.15, 8.0)},
}

#: limites de artefato validos para qualquer tipo
ARTIFACT_RULES: Dict[str, Tuple[float, float]] = {
    #: saldo liquido de densidade na ROI, relativo ao tecido original. Abaixo de
    #: 1% a geracao ou nao inseriu lesao alguma ou removeu densidade -- os dois
    #: casos sao inuteis para aumento de dataset.
    "roi_delta_ratio": (0.01, 1.00),
    "seam_gradient_excess": (-1.0, 0.60),
    "border_step_excess": (-15.0, 12.0),
    "noise_ratio_roi_ring": (0.55, 1.60),
    "sharpness_ratio": (0.25, 2.50),
    "outside_changed_fraction": (0.0, 1e-6),
}


def check_plausibility(metrics: Dict[str, float], lesion_type: str) -> Dict[str, Any]:
    """Aplica as regras e devolve o veredito por criterio."""
    checks: List[Dict[str, Any]] = []
    rules = dict(PLAUSIBILITY_RULES.get(lesion_type, PLAUSIBILITY_RULES["mass"]))
    for name, (lo, hi) in rules.items():
        value = metrics.get(name, float("nan"))
        checks.append({"criterion": name, "group": "realismo", "value": value,
                       "range": [lo, hi], "passed": bool(lo <= value <= hi)})
    for name, (lo, hi) in ARTIFACT_RULES.items():
        value = metrics.get(name, float("nan"))
        checks.append({"criterion": name, "group": "artefato", "value": value,
                       "range": [lo, hi], "passed": bool(lo <= value <= hi)})
    passed = sum(1 for c in checks if c["passed"])
    return {"checks": checks, "n_passed": passed, "n_total": len(checks),
            "pass_rate": passed / len(checks) if checks else 0.0}


# ---------------------------------------------------------------------------
# Avaliacao completa
# ---------------------------------------------------------------------------


def passes_qc(evaluation: Dict[str, Any], required: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """Portao de controle de qualidade para geracao em lote.

    O modelo ocasionalmente apenas reconstroi o tecido sem inserir lesao alguma
    (contraste nulo ou negativo) ou produz um blob exageradamente denso. Esses
    casos devem ser descartados e regerados com outra semente antes de entrarem
    em um dataset aumentado.
    """
    required = required or ["weber_contrast", "cnr", "outside_changed_fraction"]
    failed = [c["criterion"] for c in evaluation["plausibility"]["checks"]
              if c["criterion"] in required and not c["passed"]]
    return (len(failed) == 0), failed


def evaluate(result, case, cfg: QualityConfig, lesion_type: Optional[str] = None) -> Dict[str, Any]:
    """Roda toda a bateria de metricas de imagem sobre um resultado de geracao."""
    composed, original, mask = result.composed, result.original, result.mask
    box = result.box
    lesion_type = lesion_type or result.params.get("lesion_type", "mass")

    roi = (mask > 127)
    ring = ring_mask(mask, case.breast, cfg.ring_px)
    lesion = segment_lesion(original, composed, mask)

    metrics: Dict[str, float] = {}
    metrics.update(intensity_metrics(composed, original, lesion, ring, roi))
    metrics.update(shape_metrics(lesion, composed, case.mm_per_px))
    metrics.update(glcm_features(composed, lesion if lesion.any() else roi,
                                 cfg.glcm_distances, cfg.glcm_levels, "lesion_"))
    metrics.update(glcm_features(composed, ring, cfg.glcm_distances, cfg.glcm_levels, "ring_"))
    metrics.update(glcm_features(original, roi, cfg.glcm_distances, cfg.glcm_levels, "roi_original_"))
    metrics.update(seam_metrics(composed, original, mask))
    metrics.update(noise_metrics(composed, original, roi, ring))
    metrics.update(histogram_metrics(composed, original, roi, ring))
    metrics.update(fidelity_metrics(composed, original, mask))

    if box is not None:
        ctx = box.expanded(cfg.ring_px, limit=composed.shape[0])
        metrics["nps_slope_roi"] = radial_power_slope(composed, box)
        metrics["nps_slope_context"] = radial_power_slope(composed, ctx)
        metrics["nps_slope_original_roi"] = radial_power_slope(original, box)
        metrics["nps_slope_delta"] = float(metrics["nps_slope_roi"] - metrics["nps_slope_original_roi"])

    # divergencia de textura entre lesao e tecido real vizinho
    for prop in ("contrast", "homogeneity", "energy", "correlation", "dissimilarity", "entropy"):
        a = metrics.get("lesion_glcm_" + prop)
        b = metrics.get("ring_glcm_" + prop)
        if a is not None and b is not None:
            metrics["glcm_" + prop + "_ratio"] = float(a / (b + EPS))

    plausibility = check_plausibility(metrics, lesion_type)
    return {
        "case": case.name,
        "lesion_type": lesion_type,
        "lesion_area_px": int(lesion.sum()),
        "roi_area_px": int(roi.sum()),
        "metrics": metrics,
        "plausibility": plausibility,
        "lesion_mask": lesion,
    }


__all__ = ["evaluate", "passes_qc", "ring_mask", "segment_lesion", "intensity_metrics", "glcm_features",
           "shape_metrics", "seam_metrics", "noise_metrics", "histogram_metrics",
           "fidelity_metrics", "check_plausibility", "radial_power_slope",
           "PLAUSIBILITY_RULES", "ARTIFACT_RULES"]
