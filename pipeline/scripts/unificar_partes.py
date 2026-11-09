#!/usr/bin/env python
"""Junta as partes de um lote num lote unico, sem copiar bytes.

Cada parte de ``lote_em_partes.py`` e um lote completo (``original/``,
``masks/``, ``metadata/``, ``auditoria/``, ``fidelidade/``...). A entrega e
montada por ``montar_dataset_final.py``, que le UM lote. Este programa cria esse
lote na raiz da pasta:

* imagens, mascaras, paineis e mapas anatomicos por **hardlink** (mesmo disco:
  custo zero de espaco, e apagar a parte nao apaga o arquivo);
* metadados reescritos com os caminhos do lote unico;
* ``manifest.csv``, ``auditoria/auditoria.csv``, ``fidelidade/fidelidade.csv`` e
  ``calibracao_contraste.csv`` concatenados, com a coluna ``parte``;
* ``auditoria/auditoria.json`` com as lesoes de todas as partes.

As auditorias continuam validas depois da juncao: todo criterio e por lesao,
e a deduplicacao compara variantes da MESMA mama, que nunca fica em duas partes.

    .venv/bin/python scripts/unificar_partes.py ../mamas_com_lesao/lote_birads1_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

SUBPASTAS = ("original", "masks/original", "overlays", "anatomy")


def _vincular(origem: Path, destino: Path) -> None:
    if destino.exists():
        if os.path.samefile(origem, destino):
            return
        destino.unlink()
    os.link(origem, destino)


def _concatenar(partes: List[Path], rel: str, destino: Path) -> int:
    linhas: List[Dict[str, str]] = []
    campos: List[str] = []
    for p in partes:
        f = p / rel
        if not f.is_file():
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            leitor = csv.DictReader(fh)
            for c in leitor.fieldnames or []:
                if c not in campos:
                    campos.append(c)
            for r in leitor:
                r["parte"] = p.name
                linhas.append(r)
    if not linhas:
        return 0
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=campos + ["parte"])
        wr.writeheader()
        wr.writerows(linhas)
    return len(linhas)


def unificar(lote: Path, partes: List[Path]) -> Dict[str, int]:
    lote = Path(lote).resolve()
    nomes_vistos: Dict[str, str] = {}
    for sub in SUBPASTAS + ("metadata",):
        (lote / sub).mkdir(parents=True, exist_ok=True)
    n_arquivos = 0
    for parte in partes:
        parte = parte.resolve()
        for sub in SUBPASTAS:
            for f in sorted((parte / sub).glob("*")):
                if f.name in nomes_vistos and nomes_vistos[f.name] != parte.name:
                    raise RuntimeError("{} aparece em {} e {}".format(f.name, nomes_vistos[f.name], parte.name))
                nomes_vistos[f.name] = parte.name
                _vincular(f, lote / sub / f.name)
                n_arquivos += 1
        for m in sorted((parte / "metadata").glob("*.json")):
            meta = json.loads(m.read_text(encoding="utf-8"))
            meta["parte"] = parte.name
            for les in meta.get("lesoes", []):
                files = les.get("files") or {}
                for k, v in list(files.items()):
                    if isinstance(v, str) and v.startswith(str(parte) + "/"):
                        files[k] = str(lote) + v[len(str(parte)):]
            (lote / "metadata" / m.name).write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")

    contagem = {
        "arquivos_vinculados": n_arquivos,
        "manifesto": _concatenar(partes, "manifest.csv", lote / "manifest.csv"),
        "auditoria": _concatenar(partes, "auditoria/auditoria.csv", lote / "auditoria" / "auditoria.csv"),
        "fidelidade": _concatenar(partes, "fidelidade/fidelidade.csv", lote / "fidelidade" / "fidelidade.csv"),
        "calibracao": _concatenar(partes, "calibracao_contraste.csv", lote / "calibracao_contraste.csv"),
    }

    junto: Dict[str, object] = {"lote": lote.name, "partes": [p.name for p in partes], "lesoes": []}
    for p in partes:
        f = p / "auditoria" / "auditoria.json"
        if f.is_file():
            d = json.loads(f.read_text(encoding="utf-8"))
            junto.setdefault("parametros", d.get("parametros"))
            junto.setdefault("criterios", d.get("criterios"))
            for l in d.get("lesoes", []):
                l["parte"] = p.name
                junto["lesoes"].append(l)
    (lote / "auditoria" / "auditoria.json").write_text(json.dumps(junto, ensure_ascii=False), encoding="utf-8")

    resumos = [json.loads((p / "summary.json").read_text(encoding="utf-8"))
               for p in partes if (p / "summary.json").is_file()]
    soma = {k: sum(int(r.get(k, 0)) for r in resumos)
            for k in ("imagens", "lesoes", "ok", "parcial", "falha", "linhas_manifesto")}
    soma["duracao_s"] = round(sum(float(r.get("duracao_s", 0)) for r in resumos), 1)
    soma["partes"] = len(partes)
    soma["saida"] = str(lote)
    (lote / "summary.json").write_text(json.dumps(soma, indent=2, ensure_ascii=False), encoding="utf-8")
    contagem.update({"partes": len(partes)})
    print("lote unificado em {}: {}".format(lote, json.dumps(contagem)))
    return contagem


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lote", help="pasta do lote, com partes/parte_NNN/")
    ap.add_argument("--partes", nargs="*", default=None, help="so estas partes (padrao: todas com fidelidade)")
    a = ap.parse_args(argv)
    lote = Path(a.lote).resolve()
    if a.partes:
        partes = [lote / "partes" / n for n in a.partes]
    else:
        partes = sorted(p for p in (lote / "partes").glob("parte_*")
                        if (p / "fidelidade" / "fidelidade.csv").is_file())
    if not partes:
        print("erro: nenhuma parte pronta em {}".format(lote / "partes"), file=sys.stderr)
        return 2
    unificar(lote, partes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
