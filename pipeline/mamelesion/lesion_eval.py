"""Quao parecidas com as lesoes BI-RADS reais sao as lesoes geradas.

Compara recortes reais (validacao ou teste, pacientes nunca vistos no treino) com
recortes sinteticos do **mesmo tamanho**, gerados em hospedeiros tambem nunca
vistos. Tudo que um lado passa, o outro passa igual: escala ``raw / 65535``,
redimensionamento, extrator de features.

Metricas
--------
``kid``
    Kernel Inception Distance sobre features DINOv3 (CLS + media dos patches).
    Estimador nao enviesado, adequado a amostras pequenas; 0 = distribuicoes
    indistinguiveis por esse kernel. Principal criterio de escolha de checkpoint.
``auc_classificador``
    Regressao logistica real x sintetico com validacao cruzada repetida. 0,5 =
    o classificador nao separa; 1,0 = separa perfeitamente.
``textura``
    Distancia de Wasserstein entre as distribuicoes real e sintetica de
    estatisticas interpretaveis (contraste nucleo/periferia, detalhe fino, GLCM,
    inclinacao espectral), normalizada pelo desvio padrao real.
``memorizacao``
    Similaridade de cada sintetico ao recorte de treino mais parecido, contra a
    mesma medida para reais nunca vistos. Se muitos sinteticos ficam mais perto
    do treino do que reais novos ficam, o modelo esta copiando.

Com ~15-33 recortes reais por particao, todas as metricas tem variancia alta:
servem para comparar bracos e checkpoints sobre os **mesmos** recortes, nao
como numero absoluto de qualidade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np

DINO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


# ---------------------------------------------------------------------------
# Geracao pareada com os recortes de referencia
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """Uma lesao sintetica a gerar: tamanho de um recorte real, num hospedeiro, com uma semente."""

    reference: str
    lesion_class: str
    wh: Tuple[int, int]
    host_index: int
    seed: int
    place_seed: int


def plan_jobs(references: Sequence[Tuple[str, str, Tuple[int, int]]], n_hosts: int,
              seeds_per_reference: int, base_seed: int) -> List[Job]:
    """Distribui recortes x sementes pelos hospedeiros de forma deterministica e balanceada."""
    jobs = []
    k = 0
    for i, (name, lesion_class, wh) in enumerate(references):
        for s in range(seeds_per_reference):
            jobs.append(Job(name, lesion_class, tuple(wh), k % n_hosts,
                            int(np.random.default_rng([base_seed, i, s]).integers(0, 2 ** 31 - 1)),
                            int(np.random.default_rng([base_seed, i, s, 1]).integers(0, 2 ** 31 - 1))))
            k += 1
    return jobs


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class DinoFeatures:
    """CLS + media dos patches do DINOv3 ViT-B/16 (lido do cache local, sem rede)."""

    def __init__(self, model_id: str = DINO_ID, device: str = "cuda", size: int = 224) -> None:
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.size = size
        self.model = AutoModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
        self.n_registers = int(getattr(self.model.config, "num_register_tokens", 0))
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _prep(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape
        interp = cv.INTER_AREA if max(h, w) > self.size else cv.INTER_CUBIC
        return np.clip(cv.resize(crop.astype(np.float32), (self.size, self.size), interpolation=interp), 0, 1)

    def __call__(self, crops: Sequence[np.ndarray], batch: int = 32) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(crops), batch):
                x = np.stack([self._prep(c) for c in crops[i:i + batch]])
                t = torch.from_numpy(x).to(self.device)[:, None].repeat(1, 3, 1, 1)
                t = (t - self.mean) / self.std
                hidden = self.model(pixel_values=t).last_hidden_state
                cls = hidden[:, 0]
                patches = hidden[:, 1 + self.n_registers:].mean(1)
                out.append(torch.cat([cls, patches], dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))


# ---------------------------------------------------------------------------
# Distancias entre distribuicoes
# ---------------------------------------------------------------------------


def _scaled(f: np.ndarray) -> np.ndarray:
    """Norma L2 = sqrt(d): cada dimensao com energia media 1, como o kernel do KID supoe."""
    d = f.shape[1]
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12) * np.sqrt(d)


def kid(real: np.ndarray, synth: np.ndarray, subsets: int = 200, seed: int = 0) -> Dict[str, float]:
    """KID (MMD^2 com kernel polinomial cubico), media e desvio sobre subconjuntos."""
    x, y = _scaled(real), _scaled(synth)
    d = x.shape[1]
    m = min(len(x), len(y))
    if m < 2:
        return {"media": float("nan"), "desvio": float("nan"), "n_real": len(x), "n_sint": len(y)}
    rng = np.random.default_rng(seed)
    k = lambda a, b: (a @ b.T / d + 1.0) ** 3                          # noqa: E731
    vals = []
    for _ in range(subsets):
        xi = x[rng.choice(len(x), m, replace=False)]
        yi = y[rng.choice(len(y), m, replace=False)]
        kxx, kyy, kxy = k(xi, xi), k(yi, yi), k(xi, yi)
        mmd = ((kxx.sum() - np.trace(kxx)) + (kyy.sum() - np.trace(kyy))) / (m * (m - 1)) - 2 * kxy.mean()
        vals.append(mmd)
    return {"media": float(np.mean(vals)), "desvio": float(np.std(vals)), "n_real": len(x), "n_sint": len(y)}


def centroid_cosine(real: np.ndarray, synth: np.ndarray) -> float:
    a, b = real.mean(0), synth.mean(0)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def classifier_auc(real: np.ndarray, synth: np.ndarray, repeats: int = 20, folds: int = 5,
                   pca_dims: int = 16, seed: int = 0) -> Dict[str, float]:
    """AUC fora da dobra de um classificador real x sintetico (0,5 = indistinguivel)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([real, synth])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synth))])
    folds = int(min(folds, len(real), len(synth)))
    if folds < 2:
        return {"media": float("nan"), "ic95": [float("nan"), float("nan")]}
    aucs = []
    for r in range(repeats):
        proba = np.zeros(len(y))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            dims = int(max(1, min(pca_dims, len(tr) - 2, X.shape[1])))
            clf = make_pipeline(StandardScaler(), PCA(dims, random_state=seed),
                                LogisticRegression(C=0.5, max_iter=5000))
            clf.fit(X[tr], y[tr])
            proba[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, proba))
    return {"media": float(np.mean(aucs)),
            "ic95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]}


def memorization(train: np.ndarray, heldout_real: np.ndarray, synth: np.ndarray,
                 quantile: float = 0.95) -> Dict[str, float]:
    """Sinteticos perto demais do treino, comparados a reais nunca vistos.

    ``fracao_acima`` e a fracao de sinteticos cuja similaridade ao vizinho de
    treino mais proximo supera o quantil ``quantile`` da mesma medida para reais
    nunca vistos. Sem copia, espera-se ~``1 - quantile``.
    """
    norm = lambda f: f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)   # noqa: E731
    t, h, s = norm(train), norm(heldout_real), norm(synth)
    nn_real = (h @ t.T).max(1)
    nn_synth = (s @ t.T).max(1)
    thr = float(np.quantile(nn_real, quantile))
    return {"nn_treino_real_mediana": float(np.median(nn_real)),
            "nn_treino_sint_mediana": float(np.median(nn_synth)),
            "limiar": thr, "fracao_acima": float(np.mean(nn_synth > thr)),
            "esperado_sem_copia": float(1 - quantile)}


# ---------------------------------------------------------------------------
# Estatisticas interpretaveis de textura
# ---------------------------------------------------------------------------


def _radial_slope(img: np.ndarray) -> float:
    h, w = img.shape
    a = img - img.mean()
    a = a * np.outer(np.hanning(h), np.hanning(w))
    psd = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    yy, xx = np.indices(psd.shape)
    r = np.hypot(yy - h / 2, xx - w / 2).astype(int)
    radial = np.bincount(r.ravel(), psd.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    lo, hi = 3, max(4, min(h, w) // 3)
    f = np.arange(lo, hi)
    if len(f) < 3:
        return float("nan")
    return float(np.polyfit(np.log(f), np.log(radial[lo:hi] + 1e-12), 1)[0])


def texture_stats(crop: np.ndarray) -> Dict[str, float]:
    """Estatisticas de um recorte em [0, 1], na resolucao original."""
    from skimage.feature import graycomatrix, graycoprops

    h, w = crop.shape
    by, bx = max(1, int(0.15 * h)), max(1, int(0.15 * w))
    border = np.ones_like(crop, dtype=bool)
    border[by:h - by, bx:w - bx] = False
    cy0, cy1, cx0, cx1 = int(0.3 * h), int(0.7 * h), int(0.3 * w), int(0.7 * w)
    core = crop[cy0:max(cy1, cy0 + 1), cx0:max(cx1, cx0 + 1)]
    dog = cv.GaussianBlur(crop, (0, 0), 1.0) - cv.GaussianBlur(crop, (0, 0), 3.0)
    q = np.clip((crop * 32).astype(np.uint8), 0, 31)
    glcm = graycomatrix(q, [2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    return {
        "nivel_medio": float(crop.mean()),
        "contraste_nucleo_periferia": float(core.mean() - crop[border].mean()),
        "amplitude_p5_p95": float(np.percentile(crop, 95) - np.percentile(crop, 5)),
        "detalhe_fino": float(dog.std()),
        "glcm_contraste": float(graycoprops(glcm, "contrast").mean()),
        "glcm_homogeneidade": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_energia": float(graycoprops(glcm, "energy").mean()),
        "glcm_correlacao": float(graycoprops(glcm, "correlation").mean()),
        "inclinacao_espectral": _radial_slope(crop),
    }


def texture_distances(real_crops: Sequence[np.ndarray], synth_crops: Sequence[np.ndarray]) -> Dict[str, object]:
    from scipy.stats import wasserstein_distance

    r = [texture_stats(c) for c in real_crops]
    s = [texture_stats(c) for c in synth_crops]
    out: Dict[str, object] = {}
    dists = []
    for key in r[0]:
        a = np.array([x[key] for x in r], float)
        b = np.array([x[key] for x in s], float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        # piso: com recortes reais quase identicos o desvio tende a zero e a distancia explodiria
        sd = max(float(a.std()), 1e-6 + 0.01 * abs(float(np.median(a))))
        dist = float(wasserstein_distance(a, b) / sd)
        dists.append(dist)
        out[key] = {"w1_normalizada": dist, "real_mediana": float(np.median(a)), "sint_mediana": float(np.median(b))}
    out["mediana_w1_normalizada"] = float(np.median(dists))
    return out


def vae_ceiling_crops(crops, hosts, cfg, seed: int = 0) -> List[np.ndarray]:
    """Os mesmos recortes reais depois de ida e volta pelo VAE congelado, na janela nativa.

    Referencia do efeito do VAE sozinho: quanto um classificador separa a lesao
    real da *mesma* lesao reconstruida. O recorte entra **inteiro** na janela --
    colagem retangular dura, sem transicao nem ajuste de nivel --, entao a pegada
    comparada e exatamente o recorte real antes do VAE.

    Nao e um limite rigido para os geradores: estes preservam pixels nativos na
    faixa de transicao e fora da mascara. A primeira versao desta funcao colava
    com superelipse e trocava os cantos do recorte por tecido do hospedeiro,
    misturando colagem e VAE -- e o LoRA saiu *menos* separavel que ela no teste.
    """
    import torch
    from diffusers import AutoencoderKL

    from .inpainting import resolve_model_path
    from .paste import PasteConfig, make_sample

    vae = AutoencoderKL.from_pretrained(resolve_model_path(cfg.paths.model_id, cfg.paths.model_local_path),
                                        subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
    pc = PasteConfig.from_native_window(cfg.native_window, scale_jitter=(1.0, 1.0), flip_horizontal=False,
                                        flip_vertical=False, allow_fallback=True)
    pc.alpha_shape, pc.paste_feather_frac, pc.level_match = "retangulo", 0.0, "nenhum"
    rng = np.random.default_rng(seed)
    out = []
    for i, c in enumerate(crops):
        s = make_sample(c, hosts[i % len(hosts)], rng, pc, keep_native=True)
        x = torch.from_numpy(s.image_u8.astype(np.float32) / 127.5 - 1)[None, None].repeat(1, 3, 1, 1).cuda()
        with torch.no_grad():
            rec = vae.decode(vae.encode(x).latent_dist.mode()).sample.clamp(-1, 1)[0].mean(0).cpu().numpy()
        rec = (rec + 1) / 2
        if s.window.side != 512:
            rec = cv.resize(rec, (s.window.side, s.window.side), interpolation=cv.INTER_CUBIC)
        fx0, fy0, fx1, fy1 = s.footprint
        ox, oy = fx0 - s.window.x0, fy0 - s.window.y0
        out.append(np.clip(rec[oy:oy + fy1 - fy0, ox:ox + fx1 - fx0], 0, 1).astype(np.float32))
    del vae
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Selecao de checkpoint
# ---------------------------------------------------------------------------

#: limites da guarda de presenca de lesao (contraste nucleo/periferia sint. / real)
CONTRAST_RATIO_RANGE = (0.5, 1.5)
MEMO_MAX = 0.15


def contrast_ratio(real_crops: Sequence[np.ndarray], synth_crops: Sequence[np.ndarray]) -> float:
    """Mediana do contraste nucleo/periferia sintetico dividida pela real.

    Guarda contra o erro medido na primeira validacao: o modelo base, na janela
    nativa, nao gera lesao nenhuma -- reconstroi tecido comum --, e ainda assim
    teve KID e distancia de textura **menores** que o LoRA, porque tecido
    fibroso se parece com lesao para features genericas. Sem lesao, esta razao
    cai para perto de zero.
    """
    r = np.median([texture_stats(c)["contraste_nucleo_periferia"] for c in real_crops])
    s = np.median([texture_stats(c)["contraste_nucleo_periferia"] for c in synth_crops])
    return float(s / r) if abs(r) > 1e-6 else float("nan")


def validation_summary(real_feats: np.ndarray, synth_feats: np.ndarray, train_feats: np.ndarray,
                       real_crops: Sequence[np.ndarray], synth_crops: Sequence[np.ndarray]) -> Dict[str, object]:
    """Todas as medidas usadas na escolha de checkpoint, num dicionario."""
    tex = texture_distances(real_crops, synth_crops)
    return {
        "auc_classificador": classifier_auc(real_feats, synth_feats),
        "kid": kid(real_feats, synth_feats),
        "cosseno_centroides": centroid_cosine(real_feats, synth_feats),
        "memorizacao": memorization(train_feats, real_feats, synth_feats),
        "razao_contraste": contrast_ratio(real_crops, synth_crops),
        "textura_mediana_w1": tex["mediana_w1_normalizada"],
        "textura": tex,
    }


def selection_key(summary: Dict[str, object]) -> Tuple[int, float, float]:
    """Chave simples (elegivel, AUC, KID); a escolha entre candidatos usa :func:`select_best`."""
    return (0 if is_eligible(summary) else 1, float(summary["auc_classificador"]["media"]),
            float(summary["kid"]["media"]))


def is_eligible(summary: Dict[str, object]) -> bool:
    """Lesao presente (razao de contraste em ``CONTRAST_RATIO_RANGE``) e sem memorizacao."""
    ratio = float(summary["razao_contraste"])
    lo, hi = CONTRAST_RATIO_RANGE
    return bool(np.isfinite(ratio) and lo <= ratio <= hi
                and float(summary["memorizacao"]["fracao_acima"]) <= MEMO_MAX)


def select_best(summaries: Sequence[Dict[str, object]]) -> Tuple[int, List[str]]:
    """Indice do melhor candidato por criterios em cascata, com empate estatistico.

    Um criterio so desempata quando a diferenca excede o ruido dele; se nao,
    todos os candidatos dentro do ruido do melhor passam para o proximo:

    1. elegivel (:func:`is_eligible`);
    2. AUC real x sintetico -- empate se dentro do IC95 do melhor. Revisao feita
       na validacao do primeiro treino: o AUC saturou em 0,98-1,00 para todos os
       candidatos, e comparar os valores exatos escolhia por ruido;
    3. KID -- empate se dentro de um desvio do melhor;
    4. distancia de textura (mediana W1), a medida mais interpretavel;
    5. razao de contraste mais proxima de 1.

    Devolve ``(indice, trilha)``; a trilha explica o que decidiu, para o relatorio.
    """
    idx = list(range(len(summaries)))
    trail: List[str] = []
    eligible = [i for i in idx if is_eligible(summaries[i])]
    if eligible:
        idx = eligible
    trail.append("elegiveis: {} de {}".format(len(eligible), len(summaries)))

    best = min(idx, key=lambda i: summaries[i]["auc_classificador"]["media"])
    hi = float(summaries[best]["auc_classificador"]["ic95"][1])
    tied = [i for i in idx if float(summaries[i]["auc_classificador"]["media"]) <= hi]
    trail.append("AUC: melhor {:.3f} (IC95 ate {:.3f}); {} empatado(s)".format(
        summaries[best]["auc_classificador"]["media"], hi, len(tied)))
    idx = tied

    best = min(idx, key=lambda i: summaries[i]["kid"]["media"])
    limit = float(summaries[best]["kid"]["media"]) + float(summaries[best]["kid"]["desvio"])
    tied = [i for i in idx if float(summaries[i]["kid"]["media"]) <= limit]
    trail.append("KID: melhor {:.3f} (+1 desvio = {:.3f}); {} empatado(s)".format(
        summaries[best]["kid"]["media"], limit, len(tied)))
    idx = tied

    chosen = min(idx, key=lambda i: (round(float(summaries[i]["textura_mediana_w1"]), 3),
                                     abs(np.log(max(float(summaries[i]["razao_contraste"]), 1e-6)))))
    trail.append("textura W1 decide: {:.3f}".format(float(summaries[chosen]["textura_mediana_w1"])))
    return chosen, trail


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------


def comparison_grid(rows: List[List[np.ndarray]], header: List[str], tile: int = 200) -> np.ndarray:
    """Uma linha por recorte real; cada coluna um braco. Janela de intensidade do real da linha."""
    cols = len(header)
    bar = 22
    sheet = np.full((bar + len(rows) * (tile + 4), cols * (tile + 4)), 255, np.uint8)
    for c, text in enumerate(header):
        cv.putText(sheet, text[:26], (c * (tile + 4) + 4, 15), cv.FONT_HERSHEY_SIMPLEX, 0.42, 0, 1, cv.LINE_AA)
    for r, row in enumerate(rows):
        lo, hi = np.percentile(row[0], 1), np.percentile(row[0], 99.5)
        for c, img in enumerate(row[:cols]):
            if img is None:
                continue
            g = (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            s = tile / max(g.shape)
            g = cv.resize(g, (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))), interpolation=cv.INTER_AREA)
            y0, x0 = bar + r * (tile + 4), c * (tile + 4)
            sheet[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = g
    return sheet


__all__ = ["DINO_ID", "Job", "plan_jobs", "DinoFeatures", "kid", "centroid_cosine", "classifier_auc",
           "memorization", "texture_stats", "texture_distances", "contrast_ratio", "validation_summary",
           "selection_key", "is_eligible", "select_best", "vae_ceiling_crops", "CONTRAST_RATIO_RANGE", "MEMO_MAX",
           "comparison_grid"]
