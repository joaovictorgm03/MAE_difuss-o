"""Relatorio tecnico do dataset sintetico.

Resume o que o lote produziu e, principalmente, o que **foi medido**: as
metricas de imagem de :mod:`mamelesion.quality` agregadas por tipo de lesao, os
indicadores anatomicos de :mod:`mamelesion.anatomy` e a contabilidade de falhas.
Nao ha nenhuma afirmacao aqui que nao venha de um numero no ``manifest.csv``.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .pipeline import CaseReport, LesionRecord, PipelineOptions, environment_info


def _fmt(value: Optional[float], digits: int = 3, dash: str = "--") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    return "{:.{d}f}".format(float(value), d=digits)


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return {"n": 0, "media": None, "mediana": None, "min": None, "max": None, "dp": None}
    return {
        "n": len(clean),
        "media": statistics.fmean(clean),
        "mediana": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
        "dp": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
    }


def _metric_table(lesions: Sequence[LesionRecord], metric: str, digits: int = 3) -> List[str]:
    """Linhas de tabela markdown com a estatistica de um metrica por tipo."""
    by_type: Dict[str, List[float]] = defaultdict(list)
    for lesion in lesions:
        value = lesion.metrics.get(metric)
        if value is not None:
            by_type[lesion.lesion_type].append(value)
    rows = []
    for lesion_type in sorted(by_type):
        s = _stats(by_type[lesion_type])
        rows.append("| {} | {} | {} | {} | {} | {} |".format(
            lesion_type, s["n"], _fmt(s["media"], digits), _fmt(s["mediana"], digits),
            _fmt(s["min"], digits), _fmt(s["max"], digits)))
    return rows


def build_report(reports: Sequence[CaseReport], options: PipelineOptions,
                 elapsed_s: float, cfg_summary: Dict[str, Any]) -> str:
    """Monta o relatorio tecnico em Markdown."""
    lesions: List[LesionRecord] = [l for r in reports for l in r.lesions]
    status = Counter(r.status for r in reports)
    types = Counter(l.lesion_type for l in lesions)

    lines: List[str] = []
    add = lines.append

    add("# Relatorio tecnico -- dataset de mamografias com lesoes sinteticas")
    add("")
    add("Gerado em {} | {} imagem(ns) de entrada | {} lesao(oes) sintetizada(s)".format(
        datetime.now().strftime("%Y-%m-%d %H:%M"), len(reports), len(lesions)))
    add("")
    add("Modelo: `Likalto4/mammo_lesion-inpainting` (Stable Diffusion v1-5 inpainting "
        "ajustado por DreamBooth, checkpoint dos autores do MAM-E, "
        "[arXiv:2311.09822](https://arxiv.org/abs/2311.09822)). Nenhum treinamento "
        "foi feito aqui; apenas inferencia.")
    add("")

    # -- 1. resumo -----------------------------------------------------------
    add("## 1. Resumo da execucao")
    add("")
    add("| Situacao | Imagens |")
    add("|---|---|")
    for name, label in (("ok", "processadas com sucesso"),
                        ("parcial", "parcialmente processadas"),
                        ("falha", "com falha"),
                        ("ignorada", "ignoradas na validacao")):
        if status.get(name):
            add("| {} | {} |".format(label, status[name]))
    add("| **total** | **{}** |".format(len(reports)))
    add("")
    add("Tempo total: {:.1f} s ({:.1f} s por imagem em media).".format(
        elapsed_s, elapsed_s / max(1, len(reports))))
    add("")
    if types:
        add("Distribuicao por tipo de lesao: " + ", ".join(
            "{} ({})".format(k, v) for k, v in sorted(types.items())) + ".")
        add("")

    failures = [r for r in reports if r.status in {"falha", "ignorada"}]
    if failures:
        add("### Imagens nao processadas")
        add("")
        add("| Arquivo | Situacao | Motivo |")
        add("|---|---|---|")
        for r in failures:
            add("| `{}` | {} | {} |".format(Path(r.path).name, r.status, r.reason))
        add("")

    partial = [r for r in reports if r.status == "parcial"]
    if partial:
        add("Parcialmente processadas (menos lesoes que o pedido): " + ", ".join(
            "`{}` ({})".format(Path(r.path).name, r.reason) for r in partial) + ".")
        add("")

    # -- 2. plausibilidade anatomica ----------------------------------------
    add("## 2. Plausibilidade anatomica das localizacoes")
    add("")
    add("Cada lesao foi posicionada por amostragem proporcional ao mapa de "
        "plausibilidade (`mamelesion/anatomy.py`), que zera a probabilidade "
        "dentro do musculo peitoral, a menos de {:.0f} mm da linha da pele, e "
        "em tecido predominantemente adiposo. A caixa so e aceita se **todos** "
        "os seus pixels tiverem score acima do minimo -- nao apenas o centro."
        .format(cfg_summary.get("min_skin_distance_mm", 8.0)))
    add("")

    skin = _stats([l.anatomy.get("skin_distance_mm_min") for l in lesions])
    rank = _stats([l.anatomy.get("density_rank_mean") for l in lesions])
    score = _stats([l.anatomy.get("anat_score_mean") for l in lesions])
    pect = _stats([l.anatomy.get("pectoral_overlap_frac") for l in lesions])
    nipple = _stats([l.anatomy.get("nipple_distance_mm") for l in lesions])

    add("| Indicador | n | media | mediana | min | max |")
    add("|---|---|---|---|---|---|")
    add("| distancia minima ate a pele (mm) | {} | {} | {} | {} | {} |".format(
        skin["n"], _fmt(skin["media"], 1), _fmt(skin["mediana"], 1),
        _fmt(skin["min"], 1), _fmt(skin["max"], 1)))
    add("| posto de densidade do tecido (0-100) | {} | {} | {} | {} | {} |".format(
        rank["n"], _fmt(rank["media"], 1), _fmt(rank["mediana"], 1),
        _fmt(rank["min"], 1), _fmt(rank["max"], 1)))
    add("| score anatomico medio da regiao (0-1) | {} | {} | {} | {} | {} |".format(
        score["n"], _fmt(score["media"]), _fmt(score["mediana"]),
        _fmt(score["min"]), _fmt(score["max"])))
    add("| sobreposicao com o peitoral (fracao) | {} | {} | {} | {} | {} |".format(
        pect["n"], _fmt(pect["media"]), _fmt(pect["mediana"]),
        _fmt(pect["min"]), _fmt(pect["max"])))
    add("| distancia ao mamilo estimado (mm) | {} | {} | {} | {} | {} |".format(
        nipple["n"], _fmt(nipple["media"], 1), _fmt(nipple["mediana"], 1),
        _fmt(nipple["min"], 1), _fmt(nipple["max"], 1)))
    add("")

    if skin["min"] is not None:
        ok_skin = sum(1 for l in lesions
                      if (l.anatomy.get("skin_distance_mm_min") or 0) >=
                      cfg_summary.get("min_skin_distance_mm", 8.0))
        add("- **{}/{}** lesoes respeitam a distancia minima da pele.".format(ok_skin, len(lesions)))
    zero_pect = sum(1 for l in lesions if (l.anatomy.get("pectoral_overlap_frac") or 0) == 0)
    add("- **{}/{}** lesoes com sobreposicao nula ao musculo peitoral.".format(
        zero_pect, len(lesions)))
    add("")

    regions = Counter(l.region.split(" (")[0] for l in lesions)
    if regions:
        add("Distribuicao das localizacoes:")
        add("")
        add("| Regiao | Lesoes |")
        add("|---|---|")
        for region, count in regions.most_common():
            add("| {} | {} |".format(region, count))
        add("")

    pect_methods = Counter(r.anatomy.get("pectoral_method", "?") for r in reports if r.anatomy)
    if pect_methods:
        add("Deteccao do musculo peitoral por imagem: " + ", ".join(
            "{} ({})".format(k, v) for k, v in pect_methods.most_common()) + ".")
        add("")

    # -- 3. qualidade de imagem ---------------------------------------------
    add("## 3. Qualidade da sintese")
    add("")
    add("Metricas calculadas sobre a regiao gerada, comparada a um anel de "
        "tecido real ao redor dela (`mamelesion/quality.py`).")
    add("")

    for title, metric, digits, note in (
        ("Contraste de Weber (lesao x tecido vizinho)", "weber_contrast", 3,
         "positivo = lesao mais densa que o tecido, como se espera de uma massa"),
        ("Relacao contraste-ruido (CNR)", "cnr", 2,
         "acima de ~0.3 a lesao se destaca do ruido do tecido"),
        ("Excesso de gradiente na emenda", "seam_gradient_excess", 3,
         "0 = o contorno da regiao e indistinguivel do mesmo contorno na imagem original"),
        ("Razao de ruido ROI/tecido real", "noise_ratio_roi_ring", 3,
         "1.0 = mesma granulacao do tecido real"),
        ("Razao de nitidez ROI/tecido real", "sharpness_ratio", 3,
         "1.0 = mesma nitidez do tecido real"),
        ("Diametro equivalente da lesao (mm)", "equivalent_diameter_mm", 1,
         "tamanho da lesao efetivamente segmentada"),
    ):
        rows = _metric_table(lesions, metric, digits)
        if not rows:
            continue
        add("**{}** -- {}.".format(title, note))
        add("")
        add("| Tipo | n | media | mediana | min | max |")
        add("|---|---|---|---|---|---|")
        lines.extend(rows)
        add("")

    add("### Textura medida na saida em resolucao original")
    add("")
    add("As metricas acima avaliam a composicao no canvas de 1024, que e um "
        "**intermediario**: ela recebe transferencia de detalhe para compensar a "
        "ampliacao 2x da saida do modelo. O arquivo entregue em `original/` nao "
        "passa por isso -- a granulacao dele vem do proprio detector, porque a "
        "imagem de entrada entra inteira na soma. As duas texturas nao sao a "
        "mesma, entao os numeros abaixo, medidos no arquivo entregue, sao os que "
        "descrevem o dataset.")
    add("")
    for title, metric, digits, note in (
        ("Razao de ruido ROI/tecido real", "native_noise_ratio_roi_ring", 3,
         "1.0 = mesma granulacao do tecido vizinho"),
        ("Razao de nitidez ROI/tecido real", "native_sharpness_ratio", 3,
         "1.0 = mesma nitidez do tecido vizinho"),
        ("Contraste de Weber", "native_weber_contrast", 3,
         "lesao segmentada contra o anel de tecido real, mesma definicao da tabela acima"),
        ("Relacao contraste-ruido (CNR)", "native_cnr", 2,
         "mesma definicao da tabela acima, medida no arquivo entregue"),
    ):
        rows = _metric_table(lesions, metric, digits)
        if not rows:
            continue
        add("**{}** -- {}.".format(title, note))
        add("")
        add("| Tipo | n | media | mediana | min | max |")
        add("|---|---|---|---|---|---|")
        lines.extend(rows)
        add("")

    # -- 4. fidelidade -------------------------------------------------------
    add("## 4. Fidelidade fora da lesao")
    add("")
    changed = [l.metrics.get("outside_changed_fraction") for l in lesions]
    changed = [c for c in changed if c is not None]
    max_changed = max(changed) if changed else None
    add("A composicao e hibrida: apenas a regiao marcada e substituida. Na "
        "resolucao original a lesao entra como um *delta* somado a imagem de "
        "entrada, de modo que todo pixel fora da regiao permanece bit a bit "
        "identico ao arquivo original, inclusive nos PNGs de 16 bits.")
    add("")
    if max_changed is not None:
        add("- Fracao maxima de pixels alterados fora da ROI, medida no canvas: "
            "**{:.2e}** (limite do QC: 1e-6).".format(max_changed))
    ssim = _stats([l.metrics.get("ssim_full_image") for l in lesions])
    if ssim["n"]:
        add("- SSIM da imagem inteira contra a original: media **{}**, minimo **{}**."
            .format(_fmt(ssim["media"], 4), _fmt(ssim["min"], 4)))
    add("")

    # -- 5. controle de qualidade -------------------------------------------
    add("## 5. Controle de qualidade")
    add("")
    approved = sum(1 for l in lesions if l.qc_passed)
    attempts = _stats([l.attempts for l in lesions])
    add("- **{}/{}** lesoes entregues passaram no portao de QC.".format(approved, len(lesions)))
    add("- Tentativas por lesao entregue: media {}, maximo {}.".format(
        _fmt(attempts["media"], 2), _fmt(attempts["max"], 0)))
    rate = _stats([l.plausibility_rate for l in lesions])
    add("- Fracao de criterios de plausibilidade aprovados: media {}, minimo {}.".format(
        _fmt(rate["media"], 3), _fmt(rate["min"], 3)))
    conf = _stats([l.confidence for l in lesions])
    add("- Confianca da sintese (0-1): media {}, mediana {}, minimo {}.".format(
        _fmt(conf["media"], 3), _fmt(conf["mediana"], 3), _fmt(conf["min"], 3)))
    add("")

    failed_criteria: Counter = Counter()
    for lesion in lesions:
        failed_criteria.update(lesion.qc_failed_criteria)
    if failed_criteria:
        add("Criterios que ainda falham em lesoes entregues:")
        add("")
        for name, count in failed_criteria.most_common():
            add("- `{}`: {} lesao(oes)".format(name, count))
        add("")

    worst = sorted(lesions, key=lambda l: l.confidence)[:8]
    if worst:
        add("Lesoes de menor confianca (revisar antes de usar):")
        add("")
        add("| Arquivo | Tipo | Regiao | Weber | CNR | Confianca |")
        add("|---|---|---|---|---|---|")
        for lesion in worst:
            add("| `{}` | {} | {} | {} | {} | {} |".format(
                Path(lesion.files["original"]).name, lesion.lesion_type,
                lesion.region.split(" (")[0], _fmt(lesion.metrics.get("weber_contrast")),
                _fmt(lesion.metrics.get("cnr"), 2), _fmt(lesion.confidence)))
        add("")

    # -- 6. limitacoes -------------------------------------------------------
    add("## 6. Limitacoes")
    add("")
    add("1. **O gerador deixa assinatura.** O artigo reporta AUROC 0.49 para "
        "leitor humano, mas um classificador treinado para isso distingue o "
        "tecido reconstruido do real. As imagens servem para aumento de "
        "dataset, nao como padrao-ouro em estudos de deteccao de imagem sintetica.")
    add("2. **Escala em milimetros estimada.** Sem `PixelSpacing` no arquivo, "
        "`mm/px` vem da hipotese de um detector de 24x30 cm; os diametros em "
        "mm e as distancias a pele sao aproximados na mesma proporcao.")
    add("3. **Resolucao efetiva da lesao.** A lesao nasce em 512x512 e e "
        "ampliada ate a resolucao original. A textura nativa e preservada "
        "porque a imagem original entra inteira na composicao, mas a "
        "microestrutura da propria lesao nasce na escala do modelo. "
        "Microcalcificacoes nao sao geradas por este pipeline.")
    add("4. **Nomenclatura de quadrantes.** As localizacoes sao descritas em "
        "termos geometricos verificaveis na imagem (terco posterior/medio/"
        "anterior, faixa superior/media/inferior). Mapear isso para os "
        "quadrantes do BI-RADS exigiria a lateralidade e a orientacao de "
        "aquisicao registradas no arquivo, que os PNGs nao trazem.")
    add("5. **Deteccao do peitoral.** E baseada em intensidade e conectividade, "
        "nao num modelo treinado. Quando a deteccao falha em uma incidencia "
        "obliqua, um triangulo conservador e excluido no lugar; o metodo usado "
        "por imagem esta no `manifest.csv` e nos metadados.")
    add("")

    # -- 7. reprodutibilidade ------------------------------------------------
    add("## 7. Reprodutibilidade")
    add("")
    env = environment_info()
    add("| Item | Valor |")
    add("|---|---|")
    for key in ("python", "torch", "cuda", "gpu", "diffusers", "opencv", "numpy", "plataforma"):
        if key in env:
            add("| {} | `{}` |".format(key, env[key]))
    add("| semente do lote | `{}` |".format(options.seed))
    add("")
    add("Cada lesao grava em `metadata/` a semente, a caixa nos dois sistemas de "
        "coordenadas, a escala mm/px e todos os hiperparametros de difusao e "
        "composicao usados. Reexecutar com a mesma semente reproduz o lote.")
    add("")
    add("Rastreamento completo por imagem: `processing.log` (legivel) e "
        "`processing.jsonl` (um evento por linha, para auditoria).")
    add("")

    add("---")
    add("")
    add("> Imagens sinteticas destinadas a pesquisa e aumento de dataset. Nao "
        "devem ser usadas para diagnostico nem misturadas a arquivos clinicos "
        "reais sem marcacao explicita de origem sintetica.")
    return "\n".join(lines) + "\n"


__all__ = ["build_report"]
