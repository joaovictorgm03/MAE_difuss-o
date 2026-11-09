#!/usr/bin/env python
"""Verifica um dataset sintetico ja gerado, sem depender do pipeline.

Confere de forma independente as garantias que o relatorio tecnico afirma:

1. cada imagem de saida existe, abre e tem **exatamente** as dimensoes e a
   profundidade de bits do arquivo de origem;
2. **nenhum pixel fora da mascara da lesao** difere do original;
3. dentro da mascara ha alteracao de fato (a lesao nao e vazia);
4. o ``manifest.csv`` e os metadados batem com os arquivos em disco.

Uso::

    python scripts/verify_dataset.py ../mamas_com_lesao/lote_misto \
        --input-dir ../sem_anomalia

Codigo de saida 0 quando tudo passa, 1 quando alguma checagem falha.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def read_gray(path: Path) -> np.ndarray:
    im = Image.open(path)
    im.load()
    if im.mode in {"RGB", "RGBA", "P"}:
        im = im.convert("L")
    arr = np.asarray(im)
    return arr[..., 0] if arr.ndim == 3 else arr


def verify_pair(src: Path, out: Path, mask: Path) -> Tuple[bool, List[str]]:
    """Compara uma saida com sua origem. Devolve (ok, mensagens)."""
    problems: List[str] = []
    a = read_gray(src)
    b = read_gray(out)
    m = read_gray(mask)

    if a.shape != b.shape:
        problems.append("dimensoes diferem: origem {} x saida {}".format(a.shape, b.shape))
        return False, problems
    if a.dtype != b.dtype:
        problems.append("dtype difere: origem {} x saida {}".format(a.dtype, b.dtype))
    if m.shape != a.shape:
        problems.append("mascara com dimensoes {} (esperado {})".format(m.shape, a.shape))
        return False, problems

    changed = a.astype(np.int64) != b.astype(np.int64)
    inside = m > 127
    outside_changed = int((changed & ~inside).sum())
    inside_changed = int((changed & inside).sum())

    if outside_changed:
        problems.append("{} pixel(s) alterado(s) FORA da mascara".format(outside_changed))
    if inside_changed == 0:
        problems.append("nenhum pixel alterado dentro da mascara (lesao vazia)")
    if inside.sum() == 0:
        problems.append("mascara vazia")

    return (not problems), problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="pasta gerada por build_synthetic_dataset.py")
    parser.add_argument("--input-dir", required=True, help="pasta das mamografias de origem")
    parser.add_argument("--quiet", "-q", action="store_true", help="so o resumo final")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset).expanduser().resolve()
    inputs = Path(args.input_dir).expanduser().resolve()
    manifest_path = dataset / "manifest.csv"

    if not manifest_path.is_file():
        print("erro: {} nao encontrado".format(manifest_path), file=sys.stderr)
        return 1

    with open(manifest_path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("arquivo_saida")]

    if not rows:
        print("erro: manifesto sem nenhuma lesao registrada", file=sys.stderr)
        return 1

    failures: Dict[str, List[str]] = {}
    checked = 0
    for row in rows:
        src = inputs / row["arquivo_origem"]
        out = dataset / "original" / row["arquivo_saida"]
        mask = dataset / "masks" / "original" / (Path(row["arquivo_saida"]).stem + "_mask.png")

        missing = [str(p) for p in (src, out, mask) if not p.is_file()]
        if missing:
            failures[row["arquivo_saida"]] = ["arquivo ausente: " + ", ".join(missing)]
            continue

        ok, problems = verify_pair(src, out, mask)
        checked += 1
        if not ok:
            failures[row["arquivo_saida"]] = problems
        elif not args.quiet:
            print("ok   {}".format(row["arquivo_saida"]))

    # coerencia entre manifesto e metadados
    meta_dir = dataset / "metadata"
    n_meta_lesions = 0
    if meta_dir.is_dir():
        for meta_file in meta_dir.glob("*.json"):
            with open(meta_file, encoding="utf-8") as fh:
                n_meta_lesions += len(json.load(fh).get("lesoes", []))
    if n_meta_lesions != len(rows):
        failures["<metadados>"] = [
            "manifesto lista {} lesao(oes), metadados somam {}".format(len(rows), n_meta_lesions)]

    print()
    print("verificadas: {} lesao(oes) de {} no manifesto".format(checked, len(rows)))
    if failures:
        print("FALHAS: {}".format(len(failures)))
        for name, problems in failures.items():
            print("  {}".format(name))
            for problem in problems:
                print("    - {}".format(problem))
        return 1
    print("todas as checagens passaram:")
    print("  - dimensoes e profundidade de bits identicas a origem")
    print("  - zero pixels alterados fora da mascara da lesao")
    print("  - lesao presente dentro da mascara em todas as saidas")
    print("  - manifesto e metadados coerentes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
