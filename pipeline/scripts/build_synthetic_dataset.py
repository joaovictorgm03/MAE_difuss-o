#!/usr/bin/env python
"""Gera um dataset de mamografias com lesoes sinteticas, pronto para uso.

Le todas as imagens de uma pasta de mamografias saudaveis, insere em cada uma
uma (ou mais) lesao sintetica em posicao anatomicamente plausivel e grava o
resultado em duas resolucoes, com mascaras, metadados, manifesto, log de
processamento e relatorio tecnico.

Exemplos
--------
    # lote padrao: uma lesao por imagem, tipos alternados
    python scripts/build_synthetic_dataset.py \\
        --input-dir ../sem_anomalia --output-dir ../mamas_com_lesao/lote_novo

    # tres lesoes por imagem, apenas massas, semente fixa
    python scripts/build_synthetic_dataset.py --per-image 3 --types mass --seed 42

    # so validar a entrada e os mapas anatomicos, sem rodar o modelo
    python scripts/build_synthetic_dataset.py --dry-run

    # sobrescrever qualquer campo do config.yaml
    python scripts/build_synthetic_dataset.py --set diffusion.guidance_scale=5.0

    # modo nativo com o LoRA ajustado em lesoes BI-RADS 4/5 (docs/FINETUNING.md)
    python scripts/build_synthetic_dataset.py --lora treinos/birads_lora_v1 \\
        --output-dir ../mamas_com_lesao/lote_birads

Codigos de saida: 0 = tudo processado; 1 = concluido com falhas parciais;
2 = nenhuma imagem produzida (erro de configuracao ou ambiente).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamelesion.config import Config                                   # noqa: E402
from mamelesion.dataset_report import build_report                     # noqa: E402
from mamelesion.pipeline import (PipelineOptions, SyntheticLesionPipeline,   # noqa: E402
                                 ValidationError, setup_logging, write_manifest)

LOGGER = logging.getLogger("mamelesion.cli")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="config.yaml alternativo (padrao: o da raiz do projeto)")
    parser.add_argument("--input-dir", default=None,
                        help="pasta com as mamografias saudaveis (padrao: paths.input_dir)")
    parser.add_argument("--output-dir", default=None,
                        help="pasta de saida (padrao: ../mamas_com_lesao/lote_<AAAAMMDD_HHMM>)")
    parser.add_argument("--per-image", type=int, default=1,
                        help="lesoes por imagem (padrao: 1)")
    parser.add_argument("--types", default=None,
                        help="tipos de lesao, separados por virgula, com peso opcional TIPO:PESO "
                             "(padrao: mass,nodule,density alternados; com --lora: birads5:0.83,birads4:0.17, "
                             "a proporcao dos recortes reais)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="semente do lote; fixa reproduz a mesma saida")
    parser.add_argument("--canvas-size", type=int, default=1024,
                        help="lado do canvas de trabalho (padrao: 1024)")
    parser.add_argument("--limit", type=int, default=None,
                        help="processa apenas as N primeiras imagens (teste)")
    parser.add_argument("--placement-attempts", type=int, default=4,
                        help="quantas posicoes tentar antes de desistir de uma lesao")
    parser.add_argument("--no-canvas", action="store_true",
                        help="nao gravar a saida em 1024 (so a resolucao original)")
    parser.add_argument("--no-overlay", action="store_true",
                        help="nao gravar os paineis de comparacao")
    parser.add_argument("--no-anatomy-map", action="store_true",
                        help="nao gravar as figuras de auditoria anatomica")
    parser.add_argument("--no-qc", action="store_true",
                        help="desliga o portao de qualidade (grava a melhor tentativa)")
    parser.add_argument("--dry-run", action="store_true",
                        help="valida a entrada e os mapas anatomicos sem rodar o modelo")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="log de nivel DEBUG tambem no console")
    parser.add_argument("--nativo", action="store_true",
                        help="gera a lesao numa janela em resolucao original (native_window.enabled)")
    parser.add_argument("--lora", default=None, metavar="PASTA",
                        help="treino LoRA (usa o melhor checkpoint); implica --nativo")
    parser.add_argument("--lora-escala", type=float, default=None,
                        help="intensidade do adaptador (padrao: native_window.lora_scale)")
    parser.add_argument("--incidencias", default=None, metavar="CSV",
                        help="tabela arquivo,view,lateralidade que prevalece sobre o nome do arquivo "
                             "(o nome pode mentir: no birads_1, L_CC_1 e a MLO esquerda)")
    parser.add_argument("--lista", default=None, metavar="TXT",
                        help="processa so as imagens listadas (um nome por linha) de --input-dir; "
                             "e assim que um lote grande e dividido em partes retomaveis")
    parser.add_argument("--rotulo-saida", default="", metavar="SUFIXO",
                        help="acrescentado ao nome de cada saida (ex.: _L03), para lotes que reusam mamografias")
    parser.add_argument("--anatomia", action="append", default=[], metavar="CHAVE=VALOR",
                        help="sobrescreve um campo de AnatomyConfig (repetivel), ex.: "
                             "density_upper_percentile=75")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="CHAVE=VALOR",
                        help="sobrescreve um campo do config.yaml (repetivel)")
    return parser.parse_args(argv)


def parse_anatomia(itens) -> dict:
    """``["density_upper_percentile=75", ...]`` -> dict com os tipos de AnatomyConfig."""
    from mamelesion.anatomy import AnatomyConfig
    campos = AnatomyConfig.__dataclass_fields__
    out = {}
    for item in itens:
        chave, _, valor = item.partition("=")
        chave = chave.strip()
        if chave not in campos:
            raise SystemExit("erro: --anatomia {}: campo desconhecido (validos: {})".format(
                chave, ", ".join(campos)))
        v = valor.strip()
        out[chave] = None if v.lower() in ("none", "null") else (
            v.lower() == "true" if v.lower() in ("true", "false") else float(v))
    return out


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = Config.load(args.config)
    cfg.apply_overrides(args.overrides)
    if args.no_qc:
        cfg.quality.qc_enabled = False
    if args.lora:
        cfg.native_window.lora_path = str(Path(args.lora).expanduser().resolve())
        cfg.native_window.enabled = True
    if args.nativo:
        cfg.native_window.enabled = True
    if args.lora_escala is not None:
        cfg.native_window.lora_scale = args.lora_escala
    if args.types is None:
        args.types = "birads5:0.83,birads4:0.17" if cfg.native_window.lora_path and cfg.native_window.enabled \
            else "mass,nodule,density"

    input_dir = Path(args.input_dir).expanduser() if args.input_dir else \
        (REPO_ROOT / cfg.paths.input_dir).resolve()
    # Sem --output-dir, cria um lote novo datado dentro de mamas_com_lesao/ em vez
    # de sobrescrever um lote existente.
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else \
        (input_dir.parent / "mamas_com_lesao" /
         time.strftime("lote_%Y%m%d_%H%M"))
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()

    types, weights = [], []
    for item in (t.strip() for t in args.types.split(",") if t.strip()):
        name, _, weight = item.partition(":")
        types.append(name.strip())
        weights.append(float(weight) if weight else None)
    if any(w is not None for w in weights):
        if any(w is None or w <= 0 for w in weights):
            print("erro: com pesos, todo tipo precisa de TIPO:PESO > 0 ({})".format(args.types), file=sys.stderr)
            return 2
    else:
        weights = None
    try:
        for lesion_type in types:
            cfg.lesion.preset(lesion_type)
    except KeyError as exc:
        print("erro: {}".format(exc), file=sys.stderr)
        return 2

    options = PipelineOptions(
        input_dir=input_dir, output_dir=output_dir,
        lesions_per_image=max(1, args.per_image), lesion_types=types, lesion_type_weights=weights,
        seed=args.seed,
        canvas_size=args.canvas_size, save_canvas=not args.no_canvas,
        save_overlay=not args.no_overlay, save_anatomy_map=not args.no_anatomy_map,
        max_placement_attempts=max(1, args.placement_attempts),
        limit=args.limit, dry_run=args.dry_run,
        view_table_path=Path(args.incidencias).expanduser().resolve() if args.incidencias else None,
        file_list_path=Path(args.lista).expanduser().resolve() if args.lista else None,
        anatomy_overrides=parse_anatomia(args.anatomia), output_tag=args.rotulo_saida)

    setup_logging(output_dir, args.verbose)
    started = time.time()

    try:
        pipeline = SyntheticLesionPipeline(cfg, options)
        reports = pipeline.run()
    except ValidationError as exc:
        LOGGER.error("configuracao invalida: %s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("interrompido pelo usuario")
        return 1
    except Exception as exc:                                   # falha global (ambiente)
        LOGGER.error("lote abortado: %s", exc, exc_info=True)
        return 2

    elapsed = time.time() - started
    rows = write_manifest(reports, output_dir / "manifest.csv")

    report_text = build_report(reports, options, elapsed, pipeline.anatomy_cfg.to_dict())
    (output_dir / "RELATORIO_TECNICO.md").write_text(report_text, encoding="utf-8")

    summary = {
        "imagens": len(reports),
        "lesoes": sum(len(r.lesions) for r in reports),
        "ok": sum(1 for r in reports if r.status == "ok"),
        "parcial": sum(1 for r in reports if r.status == "parcial"),
        "falha": sum(1 for r in reports if r.status in {"falha", "ignorada"}),
        "linhas_manifesto": rows,
        "duracao_s": round(elapsed, 1),
        "saida": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info("manifesto, relatorio e resumo gravados em %s", output_dir,
                extra={"resumo": summary})

    print()
    print("dataset em: {}".format(output_dir))
    print("  {} imagem(ns), {} lesao(oes), {} falha(s)".format(
        summary["imagens"], summary["lesoes"], summary["falha"]))
    print("  relatorio: {}".format(output_dir / "RELATORIO_TECNICO.md"))
    print("  log:       {}".format(output_dir / "processing.log"))

    if summary["lesoes"] == 0 and not args.dry_run:
        return 2
    return 1 if summary["falha"] or summary["parcial"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
