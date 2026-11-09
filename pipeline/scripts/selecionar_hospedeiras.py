#!/usr/bin/env python
"""Seleciona as mamografias que podem receber lesao e decide a incidencia de cada uma.

O nome do arquivo nao basta. No ``birads_1`` (2000 PNG de 16 bits):

* a MLO esquerda foi exportada como ``L_CC_1`` (e ``L_CC_2``/``_3`` nas
  repeticoes): o rotulo gravado na imagem diz ``MLOE`` e o peitoral esta
  visivel. Pelo nome o pipeline trataria como CC e nao excluiria o peitoral;
* ha texto clinico gravado em algumas imagens -- ``SEGMENTECTOMIA``,
  ``QUADRANTECTOMIA``, ``MASTECTOMIA TOTAL COM RECONSTRUCAO``, ``CIRURGIA
  MAMA D``, ``RADIO E QUIMIO``, ``CICATRIZ`` -- e ate um phantom de
  calibracao (``PHANTON AUT.``). Mama operada tem distorcao cicatricial e nao
  e hospedeira plausivel para uma lesao sintetica; phantom nao e mama.

A decisao usa o **rotulo de incidencia gravado na propria imagem**: o texto
claro fora da mama. ``CCE``/``CCD`` tem 3 letras, ``MLOE``/``MLOD`` tem 4; a
razao largura/altura do rotulo separa as duas sem sobreposicao (2,7-2,9 contra
4,07-4,29 nas 2000). Uma imagem entra so quando:

1. o rotulo e exatamente 3 ou 4 letras no tamanho padrao (texto extra, rotulo
   ilegivel ou fonte de outro equipamento -> ``rotulo_nao_padrao``);
2. a incidencia do rotulo concorda com a do nome corrigido (``L_CC_<n>`` = MLO)
   (``incidencia_divergente``, excluida por conservadorismo);
3. o conteudo nao repete outro arquivo byte a byte (``duplicata_exata``);
4. a imagem nao foi hospedeira de uma entrega anterior (``--excluir-de``), para
   o dataset novo nao repetir mamas ja entregues (``ja_usada_em_entrega``).

    .venv/bin/python scripts/selecionar_hospedeiras.py ../birads_1 \\
        --saida ../mamas_com_lesao/lote_birads1_v1/_controle --excluir-de ../sem_anomalia

Grava ``hospedeiras.csv`` (todas as imagens, com medida e motivo),
``incidencias.csv`` (a tabela lida por ``build_synthetic_dataset.py
--incidencias``), ``incluidas.txt`` e ``excluidas.png`` (miniaturas das
excluidas, para conferencia visual).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import cv2 as cv
import numpy as np

#: faixas da razao largura/altura do rotulo, medidas nas 2000 do birads_1
RAZAO_CC = (2.5, 3.1)
RAZAO_MLO = (3.9, 4.5)
NOME = re.compile(r"^(?P<pac>\d+)_(?P<estudo>\d+)_(?P<lado>[LR])_(?P<inc>CC|MLO)(?:_(?P<rep>\d+))?$")


def incidencia_pelo_nome(stem: str) -> Optional[str]:
    """Incidencia do nome, com a correcao da fonte: ``L_CC_<n>`` e a MLO esquerda."""
    m = NOME.match(stem)
    if not m:
        return None
    if m.group("inc") == "MLO":
        return "MLO"
    if m.group("lado") == "L" and m.group("rep"):
        return "MLO"
    return "CC"


def medir_rotulo(path: Path) -> Dict[str, object]:
    """Mede o texto claro fora da mama: letras na maior linha e razao largura/altura."""
    raw = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if raw is None:
        return {"n_letras": 0, "razao_rotulo": float("nan"), "md5": ""}
    md5 = hashlib.md5(raw.tobytes()).hexdigest()
    if raw.ndim == 3:
        raw = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
    h, w = raw.shape
    s = 4
    sm = cv.resize(raw, (w // s, h // s), interpolation=cv.INTER_AREA).astype(np.float32)
    lo, hi = np.percentile(sm, [5, 99.9])
    fg = (sm > lo + 0.05 * (hi - lo)).astype(np.uint8)
    n, lab, st, _ = cv.connectedComponentsWithStats(fg, 8)
    if n < 2:
        return {"n_letras": 0, "razao_rotulo": float("nan"), "md5": md5}
    mama = (lab == 1 + int(np.argmax(st[1:, cv.CC_STAT_AREA]))).astype(np.uint8)
    mama = cv.resize(cv.dilate(mama, np.ones((15, 15), np.uint8)), (w, h), interpolation=cv.INTER_NEAREST)
    texto = ((raw >= 0.6 * np.percentile(raw, 99.5)) & (mama == 0)).astype(np.uint8)
    n2, _, st2, _ = cv.connectedComponentsWithStats(texto, 8)
    letras = [st2[i] for i in range(1, n2)
              if 20 <= st2[i, cv.CC_STAT_AREA] <= 20000 and 15 <= st2[i, cv.CC_STAT_HEIGHT] <= 200]
    if not letras:
        return {"n_letras": 0, "razao_rotulo": float("nan"), "md5": md5}
    letras.sort(key=lambda c: c[0])
    melhor = None
    for i, c in enumerate(letras):              # agrupa letras da mesma linha
        x0, y0, cw, ch = (int(v) for v in c[:4])
        x1, y1, k = x0 + cw, y0 + ch, 1
        for d in letras[i + 1:]:
            if abs(d[1] - y0) < 0.5 * ch and abs(d[3] - ch) < 0.4 * ch and d[0] - x1 < ch:
                x1, y1, k = max(x1, int(d[0] + d[2])), max(y1, int(d[1] + d[3])), k + 1
        if melhor is None or k > melhor[4]:
            melhor = (x0, y0, x1, y1, k)
    x0, y0, x1, y1, k = melhor
    return {"n_letras": k, "razao_rotulo": round((x1 - x0) / max(1, y1 - y0), 3), "md5": md5}


def _medir(path: Path) -> Dict[str, object]:
    return {"arquivo": path.name, **medir_rotulo(path)}


def incidencia_pelo_rotulo(n_letras: int, razao: float) -> Optional[str]:
    if n_letras == 3 and RAZAO_CC[0] <= razao <= RAZAO_CC[1]:
        return "CC"
    if n_letras == 4 and RAZAO_MLO[0] <= razao <= RAZAO_MLO[1]:
        return "MLO"
    return None


def miniaturas(pasta: Path, linhas: List[Dict[str, object]], destino: Path, por_linha: int = 10) -> None:
    tiles = []
    for r in linhas:
        raw = cv.imread(str(pasta / str(r["arquivo"])), cv.IMREAD_UNCHANGED).astype(np.float32)
        lo, hi = np.percentile(raw, [1, 99.5])
        u = np.clip((raw - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        u = cv.resize(u, (200, 300), interpolation=cv.INTER_AREA)
        cv.putText(u, str(r["arquivo"])[:22], (2, 12), cv.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
        cv.putText(u, str(r["motivo_exclusao"])[:24], (2, 294), cv.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
        tiles.append(u)
    if not tiles:
        return
    while len(tiles) % por_linha:
        tiles.append(np.zeros_like(tiles[0]))
    grade = np.vstack([np.hstack(tiles[i:i + por_linha]) for i in range(0, len(tiles), por_linha)])
    cv.imwrite(str(destino), grade)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", help="pasta com as mamografias candidatas")
    ap.add_argument("--saida", required=True, help="pasta onde gravar as tabelas")
    ap.add_argument("--excluir-de", action="append", default=[], metavar="PASTA",
                    help="mamografias com o mesmo nome nesta pasta ja foram hospedeiras de uma entrega "
                         "(repetivel)")
    ap.add_argument("--processos", type=int, default=16)
    args = ap.parse_args(argv)

    pasta = Path(args.entrada).expanduser().resolve()
    saida = Path(args.saida).expanduser().resolve()
    saida.mkdir(parents=True, exist_ok=True)
    arquivos = sorted(p for p in pasta.iterdir() if p.suffix.lower() == ".png")
    if not arquivos:
        print("erro: nenhum PNG em {}".format(pasta), file=sys.stderr)
        return 2
    ja_usadas = {p.name for d in args.excluir_de for p in Path(d).expanduser().iterdir()}

    with Pool(args.processos) as pool:
        medidas = pool.map(_medir, arquivos, chunksize=8)

    vistos: Dict[str, str] = {}
    linhas: List[Dict[str, object]] = []
    for r in medidas:
        stem = Path(str(r["arquivo"])).stem
        m = NOME.match(stem)
        pelo_nome = incidencia_pelo_nome(stem)
        pelo_rotulo = incidencia_pelo_rotulo(int(r["n_letras"]), float(r["razao_rotulo"]))
        motivo = ""
        if pelo_nome is None:
            motivo = "nome_fora_do_padrao"
        elif pelo_rotulo is None:
            motivo = "rotulo_nao_padrao"
        elif pelo_rotulo != pelo_nome:
            motivo = "incidencia_divergente"
        elif r["md5"] in vistos:
            motivo = "duplicata_exata"
        elif r["arquivo"] in ja_usadas:
            motivo = "ja_usada_em_entrega"
        vistos.setdefault(str(r["md5"]), str(r["arquivo"]))
        linhas.append({
            "arquivo": r["arquivo"], "paciente": m.group("pac") if m else "",
            "lado": m.group("lado") if m else "",
            "view_nome": pelo_nome or "", "view_rotulo": pelo_rotulo or "",
            "view": pelo_rotulo or pelo_nome or "UNKNOWN",
            "n_letras": r["n_letras"], "razao_rotulo": r["razao_rotulo"],
            "duplicata_de": vistos[str(r["md5"])] if motivo == "duplicata_exata" else "",
            "incluida": int(not motivo), "motivo_exclusao": motivo,
        })

    with open(saida / "hospedeiras.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        wr.writeheader()
        wr.writerows(linhas)
    incl = [r for r in linhas if r["incluida"]]
    with open(saida / "incidencias.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["arquivo", "view", "lateralidade"])
        wr.writeheader()
        for r in incl:
            wr.writerow({"arquivo": r["arquivo"], "view": r["view"], "lateralidade": r["lado"]})
    (saida / "incluidas.txt").write_text("".join("{}\n".format(r["arquivo"]) for r in incl), encoding="utf-8")
    miniaturas(pasta, [r for r in linhas if not r["incluida"]], saida / "excluidas.png")

    corrigidas = sum(1 for r in incl if r["view"] == "MLO" and "_CC" in str(r["arquivo"]))
    print("{} imagens: {} incluidas ({} CC, {} MLO; {} MLO que o nome chamava de CC)".format(
        len(linhas), len(incl), sum(r["view"] == "CC" for r in incl),
        sum(r["view"] == "MLO" for r in incl), corrigidas))
    for motivo, n in Counter(r["motivo_exclusao"] for r in linhas if not r["incluida"]).most_common():
        print("  excluida: {:<24} {}".format(motivo, n))
    print("pacientes incluidos: {}".format(len({r["paciente"] for r in incl})))
    print("tabelas em {}".format(saida))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
