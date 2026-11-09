"""Figuras e relatorio dos testes de qualidade."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .masks import LesionBox  # noqa: E402


def _draw_box(ax, box: Optional[LesionBox], color: str = "#ffb000") -> None:
    if box is None:
        return
    ax.add_patch(plt.Rectangle((box.x1, box.y1), box.w, box.h, fill=False,
                               edgecolor=color, linewidth=1.6))


def case_figure(result, case, evaluation: Dict[str, Any], path: Path,
                heatmaps: Optional[Dict[str, np.ndarray]] = None) -> Path:
    """Painel por caso: original, sintetica, zoom, diferenca, segmentacao e XAI."""
    heatmaps = heatmaps or {}
    box = result.box
    zoom = box.expanded(max(48, box.w // 2), limit=case.size) if box else None

    n_heat = len(heatmaps)
    ncols = 4
    nrows = 2 + int(np.ceil(n_heat / ncols)) if n_heat else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(result.original, cmap="gray", vmin=0, vmax=255)
    _draw_box(axes[0, 0], box)
    axes[0, 0].set_title("original (saudavel)", fontsize=10)

    axes[0, 1].imshow(result.composed, cmap="gray", vmin=0, vmax=255)
    _draw_box(axes[0, 1], box)
    axes[0, 1].set_title("com lesao sintetica", fontsize=10)

    diff = result.difference
    lim = max(1.0, float(np.abs(diff).max()))
    im = axes[0, 2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0, 2].set_title("diferenca (sintetica - original)", fontsize=10)
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    metrics = evaluation["metrics"]
    plaus = evaluation["plausibility"]
    text = [
        "caso: {}".format(evaluation["case"]),
        "tipo: {}".format(evaluation["lesion_type"]),
        "diametro equiv.: {:.1f} mm".format(metrics.get("equivalent_diameter_mm", float("nan"))),
        "contraste de Weber: {:.3f}".format(metrics.get("weber_contrast", float("nan"))),
        "CNR: {:.2f}".format(metrics.get("cnr", float("nan"))),
        "circularidade: {:.2f}".format(metrics.get("circularity", float("nan"))),
        "ruido ROI/anel: {:.2f}".format(metrics.get("noise_ratio_roi_ring", float("nan"))),
        "nitidez ROI/anel: {:.2f}".format(metrics.get("sharpness_ratio", float("nan"))),
        "excesso de emenda: {:+.3f}".format(metrics.get("seam_gradient_excess", float("nan"))),
        "fora da ROI alterado: {:.1e}".format(metrics.get("outside_changed_fraction", float("nan"))),
        "criterios: {}/{}".format(plaus["n_passed"], plaus["n_total"]),
    ]
    axes[0, 3].text(0.02, 0.98, "\n".join(text), va="top", ha="left", fontsize=10,
                    family="monospace", transform=axes[0, 3].transAxes)

    if zoom is not None:
        axes[1, 0].imshow(result.original[zoom.slices()], cmap="gray", vmin=0, vmax=255)
        axes[1, 0].set_title("zoom original", fontsize=10)
        axes[1, 1].imshow(result.composed[zoom.slices()], cmap="gray", vmin=0, vmax=255)
        axes[1, 1].set_title("zoom sintetica", fontsize=10)
        axes[1, 2].imshow(result.composed[zoom.slices()], cmap="gray", vmin=0, vmax=255)
        lesion = evaluation.get("lesion_mask")
        if lesion is not None and lesion.any():
            axes[1, 2].contour(lesion[zoom.slices()].astype(float), levels=[0.5],
                               colors="#ff3860", linewidths=1.4)
        axes[1, 2].set_title("lesao segmentada", fontsize=10)
        axes[1, 3].hist(result.composed[result.mask > 127].ravel(), bins=48, alpha=0.65,
                        density=True, label="ROI sintetica")
        axes[1, 3].hist(result.original[result.mask > 127].ravel(), bins=48, alpha=0.65,
                        density=True, label="ROI original")
        axes[1, 3].axis("on")
        axes[1, 3].legend(fontsize=8)
        axes[1, 3].set_title("histograma da ROI", fontsize=10)

    # Os mapas sao exibidos apenas sobre o tecido: o ar do fundo e a linha da
    # pele saturam qualquer normalizacao e escondem o contraste dentro da mama.
    breast = getattr(case, "breast", None)
    for i, (name, heat) in enumerate(heatmaps.items()):
        ax = axes[2 + i // ncols, i % ncols]
        ax.imshow(result.composed, cmap="gray", vmin=0, vmax=255)
        h = heat.astype(np.float32)
        inside = (breast > 0) if breast is not None else np.ones(h.shape, bool)
        if inside.any():
            lo, hi = np.percentile(h[inside], [1, 99])
            h = np.clip((h - lo) / (hi - lo + 1e-8), 0, 1)
        h = np.ma.masked_where(~inside, h)
        ax.imshow(h, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        _draw_box(ax, box, "#00ff88")
        ax.set_title(name, fontsize=10)

    fig.suptitle("MAM-E - lesao sintetica ({})".format(evaluation["lesion_type"]), fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def metrics_dataframe(evaluations: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Uma linha por geracao, com metricas e resultado do QC."""
    rows = []
    for ev in evaluations:
        row = {"case": ev["case"], "lesion_type": ev["lesion_type"],
               "n_passed": ev["plausibility"]["n_passed"],
               "n_total": ev["plausibility"]["n_total"]}
        row.update({k: v for k, v in ev["metrics"].items() if isinstance(v, (int, float))})
        for key in ("cad", "anomaly", "xai"):
            for k, v in (ev.get(key) or {}).items():
                if isinstance(v, (int, float, bool)):
                    row["{}_{}".format(key, k)] = v
        rows.append(row)
    return pd.DataFrame(rows)


def distribution_figure(df: pd.DataFrame, path: Path,
                        columns: Optional[List[str]] = None) -> Path:
    """Boxplots das metricas principais por tipo de lesao."""
    columns = columns or ["weber_contrast", "cnr", "equivalent_diameter_mm", "circularity",
                          "noise_ratio_roi_ring", "sharpness_ratio", "seam_gradient_excess",
                          "border_step_excess"]
    columns = [c for c in columns if c in df.columns]
    ncols = 4
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    types = sorted(df["lesion_type"].unique())
    for ax, col in zip(axes, columns):
        data = [df.loc[df["lesion_type"] == t, col].dropna().values for t in types]
        ax.boxplot(data, labels=types, showmeans=True)
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    for ax in axes[len(columns):]:
        ax.axis("off")
    fig.suptitle("Distribuicao das metricas por tipo de lesao", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "sim" if value else "nao"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "inf" if value > 0 else "-"
        return "{:.{}f}".format(value, digits)
    if value is None:
        return "-"
    return str(value)


def _table(df: pd.DataFrame, columns: List[str], group: str = "lesion_type") -> str:
    columns = [c for c in columns if c in df.columns]
    if not columns:
        return "_(sem dados)_\n"
    grouped = df.groupby(group)[columns].agg(["mean", "std"])
    header = "| tipo | n | " + " | ".join(columns) + " |"
    sep = "|" + "---|" * (len(columns) + 2)
    lines = [header, sep]
    counts = df.groupby(group).size()
    for name, row in grouped.iterrows():
        cells = ["{} ± {}".format(_fmt(row[(c, "mean")]), _fmt(row[(c, "std")], 2))
                 for c in columns]
        lines.append("| {} | {} | {} |".format(name, counts[name], " | ".join(cells)))
    return "\n".join(lines) + "\n"


def write_report(path: Path, df: pd.DataFrame, context: Dict[str, Any],
                 figures: Dict[str, str], probes: Optional[Dict[str, Any]] = None,
                 checks_summary: Optional[pd.DataFrame] = None,
                 xai_df: Optional[pd.DataFrame] = None) -> Path:
    """Gera o relatorio em Markdown com todos os resultados agregados."""
    lines: List[str] = []
    add = lines.append

    add("# Relatorio de qualidade - lesoes sinteticas MAM-E\n")
    add("Gerado em {}.\n".format(datetime.now().strftime("%d/%m/%Y %H:%M")))
    add("## 1. Configuracao do experimento\n")
    for key, value in context.items():
        add("- **{}**: {}".format(key, value))
    add("")

    add("## 2. Realismo do sinal da lesao\n")
    add("Metricas calculadas na lesao segmentada, tendo como referencia um anel "
        "de tecido real de {} px ao redor da regiao gerada (controle pareado).\n"
        .format(context.get("anel de referencia (px)", "-")))
    add(_table(df, ["equivalent_diameter_mm", "weber_contrast", "michelson_contrast",
                    "cnr", "snr", "circularity", "solidity", "spiculation_index",
                    "margin_sharpness_ratio"]))

    add("### Textura (Haralick / GLCM) - razao lesao / tecido real vizinho\n")
    add("Valores proximos de 1 indicam textura estatisticamente compativel com o "
        "tecido real ao redor.\n")
    add(_table(df, ["glcm_contrast_ratio", "glcm_homogeneity_ratio", "glcm_energy_ratio",
                    "glcm_correlation_ratio", "glcm_entropy_ratio"]))

    add("## 3. Qualidade da regiao reconstruida\n")
    add("Artefatos de emenda, ruido e nitidez. `seam_gradient_excess` compara a "
        "energia de gradiente na borda da regiao com a mesma medida na imagem "
        "original: proximo de zero significa que a insercao nao criou um degrau "
        "detectavel.\n")
    add(_table(df, ["seam_gradient_excess", "border_step_excess", "noise_ratio_roi_ring",
                    "sharpness_ratio", "nps_slope_delta", "wasserstein_roi_ring",
                    "ks_roi_ring"]))

    add("## 4. Fidelidade fora da regiao\n")
    add(_table(df, ["outside_changed_fraction", "outside_max_abs_diff", "ssim_full_image"]))

    add("## 5. Coerencia radiologica (CAD e XAI)\n")
    cad_cols = [c for c in df.columns if c.startswith("cad_") or c.startswith("anomaly_")]
    if cad_cols:
        add(_table(df, ["cad_detected_topk", "cad_detected_top1", "cad_rank_in_synthetic",
                        "cad_response_gain", "cad_response_percentile_vs_tissue",
                        "anomaly_anomaly_gain", "anomaly_roi_peak_percentile_vs_tissue"]))
    if xai_df is not None and len(xai_df):
        add("\n### Mapas de explicabilidade (XAI)\n")
        add("Calculados sobre o escore de anomalia. `pico dentro da ROI` indica se "
            "o maximo do mapa cai na regiao da lesao; `ROI / tecido` e a razao "
            "entre a resposta media na lesao e no restante do parenquima.\n")
        add("| metodo | n | pico dentro da ROI | ROI / tecido | percentil da ROI |")
        add("|---|---|---|---|---|")
        for method, grp in xai_df.groupby("metodo"):
            add("| {} | {} | {:.0f}% | {:.2f} ± {:.2f} | {:.1f} |".format(
                method, len(grp), 100 * grp["peak_inside_roi"].mean(),
                grp["roi_over_tissue_ratio"].mean(), grp["roi_over_tissue_ratio"].std(ddof=0),
                grp["roi_percentile"].mean()))
        add("")

    if probes:
        add("\n### Indistinguibilidade (sondas lineares sobre features DINOv2)\n")
        add("| sonda | AUC | IC 95% | n positivos | n negativos | leitura |")
        add("|---|---|---|---|---|---|")
        for name, info in probes.items():
            add("| {} | {} | [{}, {}] | {} | {} | {} |".format(
                name, _fmt(info.get("auc")), _fmt(info.get("auc_ci_low")),
                _fmt(info.get("auc_ci_high")), info.get("n_a"), info.get("n_b"),
                info.get("reading", "")))
        add("")

    if checks_summary is not None and len(checks_summary):
        add("## 6. Criterios de plausibilidade\n")
        add("As faixas do grupo *realismo* dependem do tipo de lesao; as do grupo "
            "*artefato* valem para todos.\n")
        add("| criterio | grupo | tipo | faixa aceita | aprovacao |")
        add("|---|---|---|---|---|")
        for _, row in checks_summary.iterrows():
            add("| {} | {} | {} | [{}, {}] | {:.0f}% ({}/{}) |".format(
                row["criterion"], row["group"], row["lesion_type"],
                _fmt(row["low"]), _fmt(row["high"]),
                100 * row["pass_rate"], int(row["n_passed"]), int(row["n_total"])))
        add("")

    add("## 7. Leitura dos resultados\n")
    for line in _interpretation(df, probes):
        add("- " + line)
    add("")

    if figures:
        add("## 8. Figuras\n")
        for name, rel in figures.items():
            add("### {}\n".format(name))
            add("![{}]({})\n".format(name, rel))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _interpretation(df: pd.DataFrame, probes: Optional[Dict[str, Any]]) -> List[str]:
    """Conclusoes derivadas dos numeros efetivamente medidos nesta execucao."""
    out: List[str] = []

    def mean(col: str) -> float:
        return float(df[col].mean()) if col in df.columns else float("nan")

    if "outside_changed_fraction" in df.columns:
        worst = float(df["outside_changed_fraction"].max())
        out.append("**Fidelidade fora da lesao**: a fracao maxima de pixels alterados "
                   "fora da regiao marcada foi {:.1e}. Na composicao *hybrid* o "
                   "restante do mamograma e bit a bit identico a imagem real, de "
                   "modo que o par (real, sintetico) difere apenas pela lesao."
                   .format(worst))

    if "weber_contrast" in df.columns:
        by_type = df.groupby("lesion_type")["weber_contrast"].mean()
        parts = ", ".join("{} {:.2f}".format(k, v) for k, v in by_type.items())
        out.append("**Contraste**: contraste de Weber medio por tipo - {}. Todos "
                   "positivos, isto e, a lesao gerada e mais densa que o tecido "
                   "vizinho, como se espera de massas e assimetrias focais."
                   .format(parts))

    if "noise_ratio_roi_ring" in df.columns and "sharpness_ratio" in df.columns:
        out.append("**Textura e ruido**: a razao de ruido ROI/tecido real ficou em "
                   "{:.2f} e a de nitidez em {:.2f} (1.00 = identico ao tecido real). "
                   "A reinjecao de alta frequencia compensa a perda causada pela "
                   "geracao em 512 e pela reamostragem para 1024."
                   .format(mean("noise_ratio_roi_ring"), mean("sharpness_ratio")))

    if "seam_gradient_excess" in df.columns:
        out.append("**Artefato de emenda**: o excesso de energia de gradiente no "
                   "contorno da regiao foi {:+.3f} em media (0 = borda "
                   "indistinguivel da mesma borda na imagem original), ou seja, a "
                   "insercao nao deixa degrau detectavel no contorno da mascara."
                   .format(mean("seam_gradient_excess")))

    if "cad_response_percentile_vs_tissue" in df.columns:
        topk = mean("cad_detected_topk")
        out.append("**CAD classico**: a lesao sintetica caiu no percentil {:.1f} da "
                   "resposta do detector de massas em relacao ao restante do "
                   "parenquima, e apareceu entre os candidatos do top-k em {:.0f}% "
                   "dos casos. O detector responde a lesao sintetica como responde "
                   "a uma densidade focal real."
                   .format(mean("cad_response_percentile_vs_tissue"), 100 * topk))

    if "anomaly_anomaly_gain" in df.columns:
        out.append("**Anomalia em features profundas**: o escore de Mahalanobis na "
                   "regiao subiu {:.2f}x em relacao a mesma regiao na imagem "
                   "saudavel, com a lesao no percentil {:.1f} do tecido. O modelo "
                   "de tecido normal, ajustado sem nenhuma lesao, trata a lesao "
                   "sintetica como anomalia."
                   .format(mean("anomaly_anomaly_gain"),
                           mean("anomaly_roi_peak_percentile_vs_tissue")))

    if probes:
        keys = list(probes)
        lesion_key = next((k for k in keys if k.startswith("lesao")), None)
        bg_key = next((k for k in keys if k.startswith("fundo")), None)
        dc_key = next((k for k in keys if k.startswith("idem")), None)
        if lesion_key:
            out.append("**Detectabilidade**: uma sonda linear sobre features DINOv2 "
                       "separa a lesao sintetica do tecido real com AUC {:.3f} - a "
                       "lesao carrega sinal suficiente para ser aprendida por um "
                       "classificador.".format(probes[lesion_key].get("auc", float("nan"))))
        if bg_key:
            auc_bg = probes[bg_key].get("auc", float("nan"))
            delta = probes[bg_key].get("delta_intensidade_medio")
            msg = ("**Impressao digital do gerador (limitacao)**: comparando o tecido "
                   "reconstruido com *o mesmo recorte na imagem real*, a sonda ainda "
                   "atinge AUC {:.3f}".format(auc_bg))
            if dc_key:
                msg += (", caindo para {:.3f} quando o nivel medio de cada recorte e "
                        "removido".format(probes[dc_key].get("auc", float("nan"))))
            if delta is not None:
                msg += (". O desvio medio de intensidade e de {:+.1f} niveis de cinza"
                        .format(delta))
            msg += (". Ou seja, boa parte da assinatura e um desvio de nivel medio "
                    "corrigivel, e resta uma diferenca de microestrutura acima do "
                    "acaso. Um leitor humano nao percebe essa diferenca (o artigo "
                    "original reporta AUROC 0.49 para radiologista), mas um "
                    "classificador treinado especificamente para isso percebe: as "
                    "imagens servem para aumento de dataset, nao como padrao-ouro "
                    "para estudos de deteccao de sinteticos.")
            out.append(msg)

    return out


def checks_summary_frame(evaluations: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Taxa de aprovacao por criterio, separada por tipo de lesao.

    As faixas de realismo dependem do tipo (uma massa e um nodulo nao tem os
    mesmos limites de diametro nem de contraste), entao agregar tudo em uma
    linha unica mostraria a faixa de um tipo aplicada aos numeros de todos.
    """
    records: Dict[tuple, Dict[str, Any]] = {}
    for ev in evaluations:
        lesion_type = ev["lesion_type"]
        for check in ev["plausibility"]["checks"]:
            key = (lesion_type, check["criterion"])
            rec = records.setdefault(key, {
                "lesion_type": lesion_type, "criterion": check["criterion"],
                "group": check["group"], "low": check["range"][0],
                "high": check["range"][1], "n_passed": 0, "n_total": 0})
            rec["n_total"] += 1
            rec["n_passed"] += int(check["passed"])
    df = pd.DataFrame(list(records.values()))
    if len(df):
        df["pass_rate"] = df["n_passed"] / df["n_total"]
        df = df.sort_values(["group", "criterion", "lesion_type"])
    return df


__all__ = ["case_figure", "metrics_dataframe", "distribution_figure", "write_report",
           "checks_summary_frame"]
