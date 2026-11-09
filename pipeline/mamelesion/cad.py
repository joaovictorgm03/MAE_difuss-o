"""Coerencia radiologica: a lesao sintetica se comporta como lesao para um CAD?

O artigo (secao 3.3) avalia o inpainting rodando um sistema CAD sobre a imagem
com lesao sintetica e inspecionando mapas de XAI (gradcam, saliency, occlusion):
a hipotese e que o algoritmo deve destacar a area da lesao sintetica, indicando
que ela tem distribuicao de pixels semelhante a de lesoes reais.

Nao existe aqui um CAD clinico treinado com laudos, entao a avaliacao usa dois
detectores independentes e explicitos, ambos construidos apenas com as imagens
saudaveis disponiveis:

* :class:`BlobCAD` - detector classico de massas por Laplaciano do Gaussiano
  multiescala, a familia de metodos usada em CAD de mamografia antes do deep
  learning. Mede *sensibilidade*: a lesao sintetica aparece entre os candidatos?
* :class:`Dinov2Anomaly` - detector de anomalia nao supervisionado no espaco de
  features do DINOv2 (distancia de Mahalanobis contra a distribuicao de tecido
  saudavel, no estilo PaDiM). Mede *saliencia semantica*: a lesao se destaca do
  tecido normal para um extrator de features moderno que nunca viu lesoes.

Os mapas de XAI (occlusion, saliency e Grad-CAM) sao calculados sobre o escore
de anomalia, reproduzindo a analise qualitativa do artigo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np

from .config import QualityConfig
from .masks import LesionBox

EPS = 1e-8
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# CAD classico: massas como blobs claros multiescala
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    x: int
    y: int
    sigma_px: float
    score: float

    def to_dict(self) -> Dict[str, float]:
        return {"x": int(self.x), "y": int(self.y),
                "sigma_px": float(self.sigma_px), "score": float(self.score)}


class BlobCAD:
    """Detector de massas por LoG normalizado em escala.

    Uma massa em FFDM e uma regiao aproximadamente elipsoidal mais densa (mais
    clara) que a vizinhanca: exatamente a assinatura de um blob claro. O LoG
    normalizado (sigma^2 * LoG) responde no centro do blob e a resposta e
    comparavel entre escalas, o que permite ranquear candidatos de tamanhos
    diferentes.
    """

    def __init__(self, scales_mm: Sequence[float], mm_per_px: float, top_k: int = 10):
        self.scales_mm = list(scales_mm)
        self.mm_per_px = mm_per_px
        self.top_k = top_k

    def response(self, image: np.ndarray, breast: np.ndarray) -> np.ndarray:
        """Mapa de resposta: maximo sobre escalas do LoG normalizado."""
        img = image.astype(np.float32)
        best = np.full(img.shape, -np.inf, dtype=np.float32)
        for scale_mm in self.scales_mm:
            # raio da lesao em px -> sigma que maximiza a resposta do LoG (r = sigma*sqrt(2))
            radius_px = (scale_mm / 2.0) / self.mm_per_px
            sigma = max(1.0, radius_px / np.sqrt(2))
            k = int(max(3, round(sigma * 6) | 1))
            blurred = cv.GaussianBlur(img, (k, k), sigma)
            log = cv.Laplacian(blurred, cv.CV_32F, ksize=3)
            normalized = -(sigma ** 2) * log     # blobs claros -> resposta positiva
            best = np.maximum(best, normalized)
        best[breast == 0] = 0.0
        # ignora a borda da mama, onde o gradiente pele/fundo domina a resposta
        interior = cv.erode(breast, cv.getStructuringElement(cv.MORPH_ELLIPSE, (41, 41)))
        best[interior == 0] = 0.0
        return best

    def detect(self, image: np.ndarray, breast: np.ndarray,
               min_distance_px: int = 40) -> Tuple[List[Candidate], np.ndarray]:
        """Devolve os ``top_k`` candidatos apos supressao de nao-maximos."""
        resp = self.response(image, breast)
        work = resp.copy()
        candidates: List[Candidate] = []
        for _ in range(self.top_k):
            idx = int(np.argmax(work))
            y, x = np.unravel_index(idx, work.shape)
            score = float(work[y, x])
            if not np.isfinite(score) or score <= 0:
                break
            candidates.append(Candidate(int(x), int(y), 0.0, score))
            cv.circle(work, (int(x), int(y)), min_distance_px, 0.0, thickness=cv.FILLED)
        return candidates, resp


def detection_report(cad: BlobCAD, original: np.ndarray, composed: np.ndarray,
                     breast: np.ndarray, box: LesionBox,
                     tolerance_px: Optional[int] = None) -> Dict[str, Any]:
    """Compara os candidatos do CAD antes e depois da insercao da lesao.

    Um resultado coerente tem: (a) a lesao sintetica entre os primeiros
    candidatos da imagem sintetica, (b) nenhum candidato equivalente na mesma
    posicao da imagem original saudavel, e (c) elevacao clara do escore local.
    """
    tolerance_px = tolerance_px or int(max(box.w, box.h) * 0.5 + 8)
    cx, cy = box.center

    cands_synth, resp_synth = cad.detect(composed, breast)
    cands_orig, resp_orig = cad.detect(original, breast)

    def rank_at_lesion(cands: List[Candidate]) -> Optional[int]:
        for i, c in enumerate(cands):
            if abs(c.x - cx) <= tolerance_px and abs(c.y - cy) <= tolerance_px:
                return i + 1
        return None

    roi = np.zeros(composed.shape, dtype=bool)
    roi[box.slices()] = True
    valid = (resp_synth != 0)
    local_synth = float(resp_synth[roi].max())
    local_orig = float(resp_orig[roi].max())
    tissue_scores = resp_synth[valid & ~roi]
    percentile = float((tissue_scores < local_synth).mean() * 100) if tissue_scores.size else float("nan")

    return {
        "rank_in_synthetic": rank_at_lesion(cands_synth),
        "rank_in_original": rank_at_lesion(cands_orig),
        "detected_top1": rank_at_lesion(cands_synth) == 1,
        "detected_topk": rank_at_lesion(cands_synth) is not None,
        "local_response_synthetic": local_synth,
        "local_response_original": local_orig,
        "response_gain": float(local_synth / (abs(local_orig) + EPS)),
        "response_percentile_vs_tissue": percentile,
        "top_scores_synthetic": [c.score for c in cands_synth[:5]],
        "tolerance_px": tolerance_px,
        "candidates_synthetic": [c.to_dict() for c in cands_synth[:5]],
    }


# ---------------------------------------------------------------------------
# Deteccao de anomalia no espaco de features do DINOv2
# ---------------------------------------------------------------------------


class Dinov2Anomaly:
    """Modelo de tecido saudavel em features do DINOv2 + distancia de Mahalanobis.

    Ajustado apenas com as mamografias saudaveis disponiveis (nenhuma lesao real
    e usada), no espirito do PaDiM. O escore de um token e a distancia de
    Mahalanobis ate a distribuicao de tecido normal; o mapa de anomalia e essa
    distancia reamostrada para o tamanho da imagem.
    """

    def __init__(self, cfg: QualityConfig, device: Optional[str] = None,
                 input_size: int = 518):
        self.cfg = cfg
        self.input_size = input_size
        self._device = device
        self._model = None
        self.mean_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.grid_: Optional[int] = None

    # -- infraestrutura ----------------------------------------------------
    @property
    def device(self) -> str:
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def load(self):
        if self._model is not None:
            return self._model
        import torch
        from transformers import AutoModel
        from .inpainting import resolve_model_path

        path = resolve_model_path(self.cfg.dinov2_id, self.cfg.dinov2_local_path)
        model = AutoModel.from_pretrained(path).to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        return model

    def _to_tensor(self, gray: np.ndarray, size: Optional[int] = None):
        """uint8 HxW -> tensor 1x3xSxS normalizado (sem depender de torchvision)."""
        import torch
        size = size or self.input_size
        img = cv.resize(gray, (size, size), interpolation=cv.INTER_AREA)
        rgb = np.stack([img] * 3, axis=-1).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def tokens(self, gray: np.ndarray) -> np.ndarray:
        """Features por token (grid x grid x dim) de uma imagem."""
        import torch
        model = self.load()
        with torch.no_grad():
            out = model(pixel_values=self._to_tensor(gray)).last_hidden_state[0, 1:]
        grid = int(np.sqrt(out.shape[0]))
        self.grid_ = grid
        return out.float().cpu().numpy().reshape(grid, grid, -1)

    def embed(self, patches: Sequence[np.ndarray], batch_size: int = 32,
              patch_input: int = 224) -> np.ndarray:
        """Embedding CLS de uma lista de recortes (uint8, qualquer tamanho)."""
        import torch
        model = self.load()
        feats = []
        for i in range(0, len(patches), batch_size):
            chunk = patches[i:i + batch_size]
            batch = torch.cat([self._to_tensor(p, patch_input) for p in chunk], dim=0)
            with torch.no_grad():
                out = model(pixel_values=batch).last_hidden_state[:, 0]
            feats.append(out.float().cpu().numpy())
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 768), np.float32)

    # -- ajuste e escore ---------------------------------------------------
    def fit(self, healthy_images: Sequence[np.ndarray], breast_masks: Sequence[np.ndarray]) -> "Dinov2Anomaly":
        """Estima media e covariancia (Ledoit-Wolf) dos tokens de tecido saudavel."""
        from sklearn.covariance import LedoitWolf

        collected: List[np.ndarray] = []
        for img, breast in zip(healthy_images, breast_masks):
            tok = self.tokens(img)
            grid = tok.shape[0]
            small = cv.resize(breast, (grid, grid), interpolation=cv.INTER_AREA)
            keep = small > 200          # tokens majoritariamente dentro da mama
            if keep.sum() < 10:
                continue
            collected.append(tok[keep])
        if not collected:
            raise RuntimeError("nenhum token de tecido saudavel coletado")
        features = np.concatenate(collected, axis=0)
        self.mean_ = features.mean(axis=0)
        lw = LedoitWolf().fit(features - self.mean_)
        self.precision_ = lw.precision_.astype(np.float32)
        return self

    def anomaly_map(self, gray: np.ndarray, breast: Optional[np.ndarray] = None) -> np.ndarray:
        """Mapa de distancia de Mahalanobis, no tamanho da imagem de entrada."""
        if self.mean_ is None:
            raise RuntimeError("chame fit() antes de anomaly_map()")
        tok = self.tokens(gray)
        grid, dim = tok.shape[0], tok.shape[2]
        flat = tok.reshape(-1, dim) - self.mean_
        dist = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", flat, self.precision_, flat), 0))
        heat = dist.reshape(grid, grid).astype(np.float32)
        heat = cv.resize(heat, (gray.shape[1], gray.shape[0]), interpolation=cv.INTER_CUBIC)
        if breast is not None:
            heat[breast == 0] = 0.0
        return heat


def anomaly_report(model: Dinov2Anomaly, original: np.ndarray, composed: np.ndarray,
                   breast: np.ndarray, box: LesionBox, interior_px: int = 40) -> Dict[str, Any]:
    """Compara o mapa de anomalia antes e depois da lesao sintetica.

    A comparacao usa o *interior* da mama: a linha da pele e o musculo peitoral
    sao naturalmente anomalos para um modelo de tecido e dominariam o pico global
    sem nenhuma relacao com a lesao.
    """
    heat_synth = model.anomaly_map(composed, breast)
    heat_orig = model.anomaly_map(original, breast)

    interior = cv.erode(breast, cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (interior_px * 2 + 1,) * 2))
    roi = np.zeros(composed.shape, dtype=bool)
    roi[box.slices()] = True
    tissue = (interior > 0) & ~roi

    peak = np.unravel_index(int(np.argmax(np.where(interior > 0, heat_synth, -np.inf))),
                            heat_synth.shape)
    roi_score = float(heat_synth[roi].mean())
    roi_score_orig = float(heat_orig[roi].mean())
    tissue_scores = heat_synth[tissue]
    percentile = float((tissue_scores < heat_synth[roi].max()).mean() * 100) if tissue_scores.size else float("nan")

    return {
        "roi_anomaly_synthetic": roi_score,
        "roi_anomaly_original": roi_score_orig,
        "anomaly_gain": float(roi_score / (roi_score_orig + EPS)),
        "tissue_anomaly_mean": float(tissue_scores.mean()) if tissue_scores.size else float("nan"),
        "roi_peak_percentile_vs_tissue": percentile,
        "global_peak_inside_roi": bool(roi[peak]),
        "global_peak_xy": [int(peak[1]), int(peak[0])],
        "heat_synthetic": heat_synth,
        "heat_original": heat_orig,
    }


# ---------------------------------------------------------------------------
# XAI: occlusion, saliency e Grad-CAM sobre o escore de anomalia
# ---------------------------------------------------------------------------


def _anomaly_score_tensor(model: Dinov2Anomaly, tensor, top_frac: float = 0.01):
    """Escore escalar diferenciavel: media das maiores distancias de Mahalanobis."""
    import torch
    net = model.load()
    out = net(pixel_values=tensor).last_hidden_state[:, 1:]
    mean = torch.as_tensor(model.mean_, device=tensor.device, dtype=out.dtype)
    prec = torch.as_tensor(model.precision_, device=tensor.device, dtype=out.dtype)
    centered = out - mean
    dist = torch.sqrt(torch.clamp(
        torch.einsum("bnd,de,bne->bn", centered, prec, centered), min=1e-8))
    k = max(1, int(dist.shape[1] * top_frac))
    return dist.topk(k, dim=1).values.mean(dim=1), out


def occlusion_map(model: Dinov2Anomaly, gray: np.ndarray, patch_px: int = 64,
                  stride_px: int = 32, batch_size: int = 16) -> np.ndarray:
    """Mapa de oclusao: queda do escore de anomalia ao ocultar cada regiao.

    Valores altos indicam regioes das quais o detector depende - a hipotese do
    artigo e que a lesao sintetica domine esse mapa.
    """
    import torch
    base_tensor = model._to_tensor(gray)
    with torch.no_grad():
        base_score = _anomaly_score_tensor(model, base_tensor)[0].item()

    h, w = gray.shape
    ys = list(range(0, h - patch_px + 1, stride_px))
    xs = list(range(0, w - patch_px + 1, stride_px))
    heat = np.zeros((len(ys), len(xs)), dtype=np.float32)
    fill = int(np.median(gray))

    coords = [(i, j) for i in range(len(ys)) for j in range(len(xs))]
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        tensors = []
        for i, j in chunk:
            occ = gray.copy()
            occ[ys[i]:ys[i] + patch_px, xs[j]:xs[j] + patch_px] = fill
            tensors.append(model._to_tensor(occ))
        batch = torch.cat(tensors, dim=0)
        with torch.no_grad():
            scores = _anomaly_score_tensor(model, batch)[0].cpu().numpy()
        for (i, j), score in zip(chunk, scores):
            heat[i, j] = base_score - float(score)

    return cv.resize(heat, (w, h), interpolation=cv.INTER_CUBIC)


def saliency_map(model: Dinov2Anomaly, gray: np.ndarray) -> np.ndarray:
    """Saliency por gradiente do escore de anomalia em relacao aos pixels."""
    import torch
    tensor = model._to_tensor(gray).requires_grad_(True)
    score, _ = _anomaly_score_tensor(model, tensor)
    score.sum().backward()
    grad = tensor.grad.detach().abs().amax(dim=1)[0].cpu().numpy()
    return cv.resize(grad, (gray.shape[1], gray.shape[0]), interpolation=cv.INTER_CUBIC)


def gradcam_map(model: Dinov2Anomaly, gray: np.ndarray, top_frac: float = 1.0) -> np.ndarray:
    """Grad-CAM sobre os tokens do ultimo bloco do DINOv2.

    Para um ViT, o "mapa de features" e a grade de tokens da ultima camada; os
    pesos vem da media dos gradientes por canal, como no Grad-CAM original.

    O escore usado aqui e a media da distancia de Mahalanobis sobre *todos* os
    tokens (``top_frac=1.0``), e nao o top-1% de :func:`saliency_map`: com um
    escore baseado em top-k o gradiente chega a apenas algumas dezenas de
    tokens, e o Grad-CAM resultante e quase todo zero.
    """
    import torch
    tensor = model._to_tensor(gray).requires_grad_(True)
    net = model.load()
    activations: Dict[str, Any] = {}

    def hook(_module, _inp, out):
        value = out[0] if isinstance(out, tuple) else out
        value.retain_grad()
        activations["value"] = value

    handle = net.encoder.layer[-1].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            score, _ = _anomaly_score_tensor(model, tensor, top_frac)
            score.sum().backward()

        act = activations["value"]
        grads = act.grad[0, 1:]
        feats = act.detach()[0, 1:]
        weights = grads.mean(dim=0)
        raw = (feats * weights).sum(dim=-1)
        cam = torch.relu(raw)
        if float(cam.max()) <= 0:      # todo o mapa negativo: usa o bruto centrado
            cam = raw - raw.min()
        cam = cam.float().cpu().numpy()
    finally:
        handle.remove()

    grid = int(np.sqrt(cam.shape[0]))
    cam = cam[:grid * grid].reshape(grid, grid)
    return cv.resize(cam, (gray.shape[1], gray.shape[0]), interpolation=cv.INTER_CUBIC)


def xai_hit_rate(heat: np.ndarray, box: LesionBox, breast: Optional[np.ndarray] = None,
                 interior_px: int = 40) -> Dict[str, float]:
    """Quantifica se o mapa de XAI se concentra na lesao.

    Restringe a analise ao interior da mama pelo mesmo motivo de
    :func:`anomaly_report`: a borda da pele domina qualquer mapa de saliencia.
    """
    roi = np.zeros(heat.shape, dtype=bool)
    roi[box.slices()] = True
    if breast is not None:
        interior = cv.erode(breast, cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (interior_px * 2 + 1,) * 2))
        valid = (interior > 0) | roi
    else:
        valid = np.ones_like(roi)
    outside = valid & ~roi
    peak = np.unravel_index(int(np.argmax(np.where(valid, heat, -np.inf))), heat.shape)
    inside_vals, outside_vals = heat[roi], heat[outside]
    return {
        "peak_inside_roi": bool(roi[peak]),
        "roi_mean": float(inside_vals.mean()),
        "outside_mean": float(outside_vals.mean()),
        "roi_over_tissue_ratio": float(inside_vals.mean() / (outside_vals.mean() + EPS)),
        "roi_percentile": float((outside_vals < inside_vals.max()).mean() * 100),
    }


# ---------------------------------------------------------------------------
# Indistinguibilidade: sonda linear real vs sintetico
# ---------------------------------------------------------------------------


def linear_probe_auc(features_a: np.ndarray, features_b: np.ndarray,
                     n_splits: int = 5, seed: int = 0) -> Dict[str, float]:
    """AUC de uma regressao logistica separando dois conjuntos de patches.

    Interpretacao depende do par testado:
    * lesao sintetica x tecido real  -> AUC alta = a lesao e detectavel (desejado)
    * fundo reconstruido x tecido real -> AUC ~0.5 = sem impressao digital do
      gerador (desejado); AUC alta indica artefato sistematico do modelo.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(features_a) < 3 or len(features_b) < 3:
        return {"auc": float("nan"), "n_a": len(features_a), "n_b": len(features_b)}

    X = np.concatenate([features_a, features_b], axis=0)
    y = np.concatenate([np.ones(len(features_a)), np.zeros(len(features_b))])
    n_splits = int(min(n_splits, len(features_a), len(features_b)))
    if n_splits < 2:
        return {"auc": float("nan"), "n_a": len(features_a), "n_b": len(features_b)}

    scores = np.zeros(len(y))
    cv_split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in cv_split.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
        clf.fit(X[train], y[train])
        scores[test] = clf.predict_proba(X[test])[:, 1]

    auc = float(roc_auc_score(y, scores))
    # intervalo por bootstrap
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(300):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boot.append(roc_auc_score(y[idx], scores[idx]))
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (float("nan"),) * 2
    return {"auc": auc, "auc_ci_low": lo, "auc_ci_high": hi,
            "n_a": int(len(features_a)), "n_b": int(len(features_b))}


def sample_coords_in_region(region: np.ndarray, size: int, n: int,
                            rng: Optional[np.random.Generator] = None) -> List[Tuple[int, int]]:
    """Centros cujo quadrado de lado ``size`` cabe inteiro dentro de ``region``.

    Usa a transformada de distancia de Chebyshev. E o que torna honesta a sonda
    de impressao digital: um recorte que vaze para fora da area reconstruida (ou
    que inclua a lesao) tornaria o teste trivial e sem significado.
    """
    rng = rng or np.random.default_rng(0)
    binary = region.astype(np.uint8)
    if binary.sum() == 0:
        return []
    dist = cv.distanceTransform(binary, cv.DIST_C, 3)
    ys, xs = np.nonzero(dist >= size / 2.0)
    if len(xs) == 0:
        return []
    # Selecao gulosa com espacamento minimo: recortes repetidos ou sobrepostos
    # inflariam o n das sondas e vazariam entre as particoes da validacao
    # cruzada, produzindo AUC otimista.
    order = rng.permutation(len(xs))
    chosen: List[Tuple[int, int]] = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if all(max(abs(y - cy), abs(x - cx)) >= size for cy, cx in chosen):
            chosen.append((y, x))
            if len(chosen) >= n:
                break
    return chosen


def crop_at(image: np.ndarray, center: Tuple[int, int], size: int) -> Optional[np.ndarray]:
    """Recorte quadrado centrado em ``center`` (y, x), ou None se nao couber."""
    y, x = center
    h = size // 2
    patch = image[y - h:y - h + size, x - h:x - h + size]
    return patch if patch.shape == (size, size) else None


def sample_patches_in_region(image: np.ndarray, region: np.ndarray, size: int, n: int,
                             rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    """Recortes de ``size`` px inteiramente contidos em ``region``."""
    patches = [crop_at(image, c, size)
               for c in sample_coords_in_region(region, size, n, rng)]
    return [p for p in patches if p is not None]


def sample_tissue_patches(image: np.ndarray, breast: np.ndarray, size: int, n: int,
                          exclude: Optional[np.ndarray] = None,
                          rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    """Amostra recortes quadrados de tecido real (controles negativos)."""
    rng = rng or np.random.default_rng(0)
    half = size // 2
    eroded = cv.erode(breast, cv.getStructuringElement(cv.MORPH_ELLIPSE, (size + 1, size + 1)))
    if exclude is not None:
        eroded = eroded.copy()
        eroded[cv.dilate(exclude.astype(np.uint8) * 255,
                         cv.getStructuringElement(cv.MORPH_ELLIPSE, (size + 1, size + 1))) > 0] = 0
    ys, xs = np.nonzero(eroded > 0)
    if len(xs) == 0:
        return []
    patches = []
    for _ in range(n):
        i = int(rng.integers(len(xs)))
        y, x = int(ys[i]), int(xs[i])
        patch = image[y - half:y + half, x - half:x + half]
        if patch.shape == (size, size):
            patches.append(patch)
    return patches


__all__ = ["BlobCAD", "Candidate", "detection_report", "Dinov2Anomaly", "anomaly_report",
           "occlusion_map", "saliency_map", "gradcam_map", "xai_hit_rate",
           "linear_probe_auc", "sample_tissue_patches", "sample_patches_in_region",
           "sample_coords_in_region", "crop_at"]
