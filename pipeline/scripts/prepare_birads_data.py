#!/usr/bin/env python
"""Fase 0 do ajuste fino: audita os recortes BI-RADS, divide por paciente e cataloga tamanhos.

Nao usa GPU nem o modelo. Grava em ``--out``:

    split.json              particoes treino/validacao/teste dos recortes e dos hospedeiros
    catalogo_tamanhos.json  [largura, altura] reais dos recortes de TREINO, por classe
    auditoria.json          contagens, rejeitados e estatisticas
    AUDITORIA.md            o mesmo, legivel
    folha_<particao>.png    todos os recortes de cada particao, para conferencia visual

Exemplo
-------
    .venv/bin/python scripts/prepare_birads_data.py \\
        --classe birads5=../_download_drive --classe birads4=../birads_4 \\
        --hospedeiros ../sem_anomalia --out treinos/dados_birads

A semente fixa a divisao. Trocar a semente ou os dados muda o conjunto de teste:
toda avaliacao comparativa precisa usar o mesmo ``split.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2 as cv                                                        # noqa: E402
import numpy as np                                                      # noqa: E402
from PIL import Image                                                   # noqa: E402

from mamelesion.birads_data import (CLASS_PROMPTS, discover_crops, find_hosts,  # noqa: E402
                                    parse_ids, save_json, size_catalog, split_by_patient)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classe", action="append", required=True, metavar="CLASSE=PASTA",
                    help="pasta de recortes de uma classe (repetivel; mesma classe pode ter varias pastas)")
    ap.add_argument("--hospedeiros", required=True, help="pasta das mamografias saudaveis")
    ap.add_argument("--out", required=True, help="pasta de saida (ex.: treinos/dados_birads)")
    ap.add_argument("--seed", type=int, default=20260916, help="semente da divisao")
    ap.add_argument("--fracoes", default="0.70,0.15,0.15", help="treino,validacao,teste (por paciente)")
    ap.add_argument("--fracoes-hospedeiros", default="0.60,0.20,0.20",
                    help="treino,validacao,teste dos hospedeiros (por paciente)")
    return ap.parse_args(argv)


def contact_sheet(crops, path: Path, tile: int = 160, cols: int = 10) -> None:
    rows = max(1, (len(crops) + cols - 1) // cols)
    sheet = np.zeros((rows * tile, cols * tile), np.uint8)
    for i, c in enumerate(crops):
        img = c.image
        lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
        g = (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        s = tile / max(g.shape)
        g = cv.resize(g, (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))), interpolation=cv.INTER_AREA)
        r, k = divmod(i, cols)
        sheet[r * tile:r * tile + g.shape[0], k * tile:k * tile + g.shape[1]] = g
        # marca BI-RADS 4 com um canto claro, para conferir a estratificacao a olho
        if c.lesion_class == "birads4":
            sheet[r * tile:r * tile + 10, k * tile:k * tile + 10] = 255
    Image.fromarray(sheet).save(path)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sources = {}
    for item in args.classe:
        name, _, folder = item.partition("=")
        if not folder:
            print("erro: use CLASSE=PASTA, recebido {!r}".format(item), file=sys.stderr)
            return 2
        sources.setdefault(name.strip(), []).append(Path(folder).expanduser())
    for name, dirs in sources.items():
        for d in dirs:
            if not d.is_dir():
                print("erro: pasta inexistente para {}: {}".format(name, d), file=sys.stderr)
                return 2

    fr = tuple(float(x) for x in args.fracoes.split(","))
    fr_h = tuple(float(x) for x in args.fracoes_hospedeiros.split(","))

    started = time.time()
    crops, rejected = discover_crops(sources)
    if not crops:
        print("erro: nenhum recorte valido", file=sys.stderr)
        return 2

    split = split_by_patient(((c.name, c.patient, c.lesion_class) for c in crops), fr, args.seed)
    by_name = {c.name: c for c in crops}

    hosts = find_hosts(Path(args.hospedeiros))
    if len(hosts) < 3:
        print("erro: {} hospedeiro(s) da mesma fonte; minimo 3".format(len(hosts)), file=sys.stderr)
        return 2
    host_split = split_by_patient(((h.name, parse_ids(h.name)[0], "host") for h in hosts), fr_h, args.seed)

    catalog = size_catalog(by_name[n] for n in split["train"])
    save_json({"seed": args.seed, "fracoes": fr, "fracoes_hospedeiros": fr_h,
               "fontes": {k: [str(p.resolve()) for p in v] for k, v in sources.items()},
               "hospedeiros_pasta": str(Path(args.hospedeiros).resolve()),
               "recortes": split, "hospedeiros": host_split,
               "classe_de": {c.name: c.lesion_class for c in crops},
               "caminho_de": {c.name: str(c.path.resolve()) for c in crops},
               "paciente_de": {c.name: c.patient for c in crops}},
              out / "split.json")
    save_json(catalog, out / "catalogo_tamanhos.json")

    # --- auditoria ---------------------------------------------------------
    stats = {"recortes_validos": len(crops), "rejeitados": rejected, "particoes": {}}
    for part, names in split.items():
        cs = [by_name[n] for n in names]
        per_class = Counter(c.lesion_class for c in cs)
        pats = {c.lesion_class: len({x.patient for x in cs if x.lesion_class == c.lesion_class}) for c in cs}
        sides = np.array([max(c.wh) for c in cs]) if cs else np.array([0])
        stats["particoes"][part] = {
            "recortes": len(cs), "por_classe": dict(per_class), "pacientes_por_classe": pats,
            "lado_maior_px": {"min": int(sides.min()), "mediana": int(np.median(sides)), "max": int(sides.max())},
            "hospedeiros": host_split[part],
        }
        contact_sheet(cs, out / "folha_{}.png".format(part))
    save_json(stats, out / "auditoria.json")

    lines = ["# Auditoria dos recortes BI-RADS", "",
             "Gerado em {} | semente {} | {:.1f} s".format(time.strftime("%Y-%m-%d %H:%M"), args.seed,
                                                          time.time() - started), "",
             "Fontes: " + "; ".join("`{}` <- {}".format(k, ", ".join(str(p) for p in v))
                                    for k, v in sources.items()), "",
             "## Divisao por paciente", "",
             "| Particao | Recortes | BI-RADS 4 (pacientes) | BI-RADS 5 (pacientes) | Lado maior px (min/med/max) | Hospedeiros |",
             "|---|---|---|---|---|---|"]
    for part in ("train", "val", "test"):
        s = stats["particoes"][part]
        pc, pp = s["por_classe"], s["pacientes_por_classe"]
        lines.append("| {} | {} | {} ({}) | {} ({}) | {}/{}/{} | {} |".format(
            part, s["recortes"], pc.get("birads4", 0), pp.get("birads4", 0),
            pc.get("birads5", 0), pp.get("birads5", 0), s["lado_maior_px"]["min"],
            s["lado_maior_px"]["mediana"], s["lado_maior_px"]["max"], len(s["hospedeiros"])))
    lines += ["", "Nenhum paciente aparece em mais de uma particao (garantido por "
              "`split_by_patient` e verificado abaixo).", "",
              "## Rejeitados", ""]
    lines += ["- `{}` ({}): {}".format(Path(r["arquivo"]).name, r["classe"], r["motivo"]) for r in rejected] \
        or ["Nenhum."]
    lines += ["", "## Hospedeiros (mesma fonte, 16 bits)", ""]
    for part in ("train", "val", "test"):
        lines.append("- **{}**: {}".format(part, ", ".join("`{}`".format(h) for h in host_split[part])))
    lines += ["", "## Catalogo de tamanhos (treino)", "",
              "Usado pela inferencia para sortear o tamanho de cada lesao a partir dos recortes reais.", ""]
    for k, v in catalog.items():
        lados = np.array([max(x) for x in v])
        lines.append("- `{}`: {} tamanhos, lado maior {}-{} px (mediana {})".format(
            k, len(v), lados.min(), lados.max(), int(np.median(lados))))
    (out / "AUDITORIA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- verificacao independente de vazamento ----------------------------------
    pats = {part: {by_name[n].patient for n in names} for part, names in split.items()}
    leak = (pats["train"] & pats["val"]) | (pats["train"] & pats["test"]) | (pats["val"] & pats["test"])
    hpats = {part: {parse_ids(n)[0] for n in names} for part, names in host_split.items()}
    hleak = (hpats["train"] & hpats["val"]) | (hpats["train"] & hpats["test"]) | (hpats["val"] & hpats["test"])
    if leak or hleak:
        print("ERRO: vazamento de paciente entre particoes: {} {}".format(sorted(leak), sorted(hleak)),
              file=sys.stderr)
        return 1

    print("\n".join(lines[8:13]))
    print("\nrejeitados: {} | hospedeiros: {} | saida: {}".format(len(rejected), len(hosts), out.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
