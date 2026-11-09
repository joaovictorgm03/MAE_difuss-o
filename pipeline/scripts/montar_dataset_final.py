#!/usr/bin/env python
"""Monta o dataset final a partir de um lote auditado.

Le ``<lote>/auditoria/auditoria.csv`` (produzido por ``auditar_dataset.py``),
seleciona as lesoes **aprovadas** e escreve duas entregas irmas:

    dataset_escolhido/
    ├── dataset_bbox/         conferencia e treino de detector
    │   ├── imagens/            a mamografia com a caixa desenhada (8 bits RGB)
    │   ├── anotacoes/          caixas.csv, coco.json, yolo/*.txt, classes.txt
    │   └── LEIA-ME.md
    └── datasetfinal_MAE/     a entrega principal
        ├── imagens/            PNG na resolucao e profundidade de bits originais
        ├── mascaras/           regiao redesenhada (branco), mesma resolucao
        ├── manifest.csv        uma linha por imagem, com metricas e particao
        ├── qualidade/          auditoria completa do que entrou
        └── LEIA-ME.md

As caixas sao a **pegada da lesao** em pixels da resolucao nativa (x1,y1
inclusivos; x2,y2 exclusivos), nao a mascara -- a mascara inclui a margem que o
modelo redesenha em volta.

    .venv/bin/python scripts/montar_dataset_final.py \\
        ../mamas_com_lesao/lote_variacoes_v1 --saida ../dataset_escolhido
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from PIL import Image

from desenhar_bbox import CORES, to_rgb8                      # noqa: E402
import cv2 as cv                                              # noqa: E402

from mamelesion.pipeline import load_view_table, resolve_view  # noqa: E402

CLASSES = ["birads4", "birads5", "mass", "nodule", "density"]


def paciente(stem: str) -> str:
    return stem.split("_", 1)[0]


def particoes(hosts: List[str], split_path: Optional[Path], seed: int = 1337,
              fracoes=(0.7, 0.15, 0.15), estavel: bool = False) -> Dict[str, str]:
    """Particao por paciente, honrando a divisao ja usada no treino do LoRA.

    Duas imagens do mesmo paciente nunca caem em particoes diferentes, e um
    hospedeiro que ja era de validacao ou teste continua sendo.

    ``estavel``: a particao de cada paciente vem de um hash do proprio ID, nao
    de um sorteio sobre os pacientes presentes. Necessario quando a entrega
    cresce em etapas -- com o sorteio, acrescentar pacientes mudaria a particao
    dos que ja tinham sido entregues.
    """
    fixo: Dict[str, str] = {}
    if split_path and split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        for parte, arquivos in (split.get("hospedeiros") or {}).items():
            for a in arquivos:
                fixo[paciente(Path(a).stem)] = parte
        # paciente com recorte de lesao no treino do LoRA fica na mesma particao
        # do recorte: senao a mama saudavel dele iria para o teste do detector e a
        # lesao real dele para o treino do gerador (ou o contrario)
        pac_de = split.get("paciente_de", {})
        for parte, recortes in (split.get("recortes") or {}).items():
            for c in recortes:
                if c in pac_de:
                    fixo.setdefault(pac_de[c], parte)

    pacientes = sorted({paciente(h) for h in hosts})
    livres = [p for p in pacientes if p not in fixo]
    if estavel:
        destino = dict(fixo)
        for p in livres:
            u = zlib.crc32("particao:{}".format(p).encode("utf-8")) / 2.0 ** 32
            destino[p] = "train" if u < fracoes[0] else ("val" if u < fracoes[0] + fracoes[1] else "test")
        return {h: destino[paciente(h)] for h in hosts}
    rng = np.random.default_rng(seed)
    rng.shuffle(livres)
    n = len(livres)
    corte1 = int(round(fracoes[0] * n))
    corte2 = corte1 + int(round(fracoes[1] * n))
    destino = dict(fixo)
    for i, p in enumerate(livres):
        destino[p] = "train" if i < corte1 else ("val" if i < corte2 else "test")
    return {h: destino[paciente(h)] for h in hosts}


def desenhar(img_path: Path, caixa, tipo: str, destino: Path) -> None:
    img = to_rgb8(np.asarray(Image.open(img_path)))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = caixa
    cor = CORES.get(tipo, (255, 255, 0))
    thick = max(3, int(round(0.0025 * max(h, w))))
    font = max(1.0, 0.0006 * max(h, w))
    cv.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), cor, thick, cv.LINE_AA)
    rotulo = "{} {}x{} px".format(tipo, x2 - x1, y2 - y1)
    (tw, th), base = cv.getTextSize(rotulo, cv.FONT_HERSHEY_SIMPLEX, font, max(2, thick // 2))
    ty = y1 - thick - 6 if y1 - thick - 6 - th > 0 else y2 + th + thick + 6
    tx = min(max(0, x1), w - tw - 2)
    cv.rectangle(img, (tx - 2, ty - th - 4), (tx + tw + 2, ty + base), (0, 0, 0), -1)
    cv.putText(img, rotulo, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, font, cor, max(2, thick // 2), cv.LINE_AA)
    Image.fromarray(img).save(destino, optimize=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lote", help="pasta do lote auditado")
    ap.add_argument("--saida", default="../dataset_escolhido", help="pasta do dataset final")
    ap.add_argument("--max-por-mama", type=int, default=None,
                    help="limita variantes por mamografia (maior confianca primeiro)")
    ap.add_argument("--split", default="treinos/birads_lora_v1/split.json",
                    help="divisao de hospedeiros a honrar (json do treino)")
    ap.add_argument("--sem-bbox", action="store_true", help="nao gerar dataset_bbox/")
    ap.add_argument("--incidencias", default=None, metavar="CSV",
                    help="tabela arquivo,view,lateralidade usada na geracao (coluna view do manifesto)")
    ap.add_argument("--particao-estavel", action="store_true",
                    help="particao por hash do ID do paciente: nao muda quando a entrega cresce em etapas")
    ap.add_argument("--entregas", default=None, metavar="JSON",
                    help="{parte: numero da entrega} de lote_em_partes.py (coluna entrega do manifesto)")
    ap.add_argument("--politicas", default=None, metavar="JSON",
                    help="{parte: rotulo da politica de geracao} (coluna politica do manifesto)")
    ap.add_argument("--reusar-bbox", default=None, metavar="PASTA",
                    help="dataset_bbox/imagens de uma entrega anterior: a imagem com caixa de mesmo nome e "
                         "reaproveitada (hardlink) em vez de redesenhada")
    ap.add_argument("--selecionar", type=int, default=None, metavar="N",
                    help="entrega exatamente N das aprovadas, estratificando por contraste (5 faixas de "
                         "crel_afastado) x tamanho (3 faixas de diametro), na proporcao de cada celula")
    ap.add_argument("--processos", type=int, default=6,
                    help="processos para desenhar as caixas (2-3 s por imagem de 16 bits em resolucao cheia)")
    ap.add_argument("--vincular", action="store_true",
                    help="hardlink em vez de copia para imagens e mascaras (mesmo disco): a entrega "
                         "nao duplica o espaco dos PNG de 16 bits; apagar o lote nao a afeta")
    ap.add_argument("--fidelidade", default=None,
                    help="fidelidade.csv de scripts/fidelidade_real.py; entra no dataset so a "
                         "lesao aprovada nos DOIS criterios (auditoria e fidelidade as reais). "
                         "'auto' = <lote>/fidelidade/fidelidade.csv")
    ap.add_argument("--calibracao", default=None,
                    help="calibracao_contraste.csv de scripts/calibrar_contraste.py, para gravar "
                         "no manifesto o ganho aplicado. 'auto' = <lote>/calibracao_contraste.csv")
    args = ap.parse_args(argv)

    lote = Path(args.lote).expanduser().resolve()
    aud_csv = lote / "auditoria" / "auditoria.csv"
    if not aud_csv.is_file():
        print("erro: rode antes scripts/auditar_dataset.py {}".format(lote), file=sys.stderr)
        return 2
    with open(aud_csv, encoding="utf-8") as fh:
        linhas = list(csv.DictReader(fh))
    aprovadas = [l for l in linhas if l["veredito"] == "aprovada"]
    if not aprovadas:
        print("erro: nenhuma lesao aprovada na auditoria", file=sys.stderr)
        return 2

    fid_por_saida: Dict[str, Dict[str, str]] = {}
    if args.fidelidade:
        fid_csv = (lote / "fidelidade" / "fidelidade.csv") if args.fidelidade == "auto" \
            else Path(args.fidelidade).expanduser().resolve()
        if not fid_csv.is_file():
            print("erro: {} nao existe; rode scripts/fidelidade_real.py".format(fid_csv), file=sys.stderr)
            return 2
        with open(fid_csv, encoding="utf-8") as fh:
            fid_por_saida = {r["saida"]: r for r in csv.DictReader(fh)}
        antes = len(aprovadas)
        aprovadas = [l for l in aprovadas
                     if fid_por_saida.get(l["saida"], {}).get("veredito") == "aprovada"]
        print("fidelidade as reais: {} de {} aprovadas na auditoria seguem no dataset".format(
            len(aprovadas), antes))
        if not aprovadas:
            print("erro: nenhuma lesao passou nos dois criterios", file=sys.stderr)
            return 2

    cal_por_saida: Dict[str, Dict[str, str]] = {}
    if args.calibracao:
        cal_csv = (lote / "calibracao_contraste.csv") if args.calibracao == "auto" \
            else Path(args.calibracao).expanduser().resolve()
        if cal_csv.is_file():
            with open(cal_csv, encoding="utf-8") as fh:
                cal_por_saida = {r["saida"]: r for r in csv.DictReader(fh)}

    meta_por_host: Dict[str, Dict[str, Any]] = {}
    for m in (lote / "metadata").glob("*.json"):
        meta_por_host[m.stem] = json.loads(m.read_text(encoding="utf-8"))
    lesao_por_saida: Dict[str, Dict[str, Any]] = {}
    for meta in meta_por_host.values():
        for l in meta.get("lesoes", []):
            lesao_por_saida[Path(l["files"]["original"]).name] = {"lesao": l, "meta": meta}

    if args.selecionar and len(aprovadas) > args.selecionar:
        aprovadas = selecionar_estratificado(aprovadas, fid_por_saida, args.selecionar)
        print("selecionadas {} (estratificadas por contraste x tamanho)".format(len(aprovadas)))

    if args.max_por_mama:
        por_host: Dict[str, List[Dict[str, str]]] = {}
        for l in aprovadas:
            por_host.setdefault(l["origem"], []).append(l)
        aprovadas = [x for grupo in por_host.values()
                     for x in sorted(grupo, key=lambda r: -float(r["confianca"]))[:args.max_por_mama]]
        aprovadas.sort(key=lambda r: r["saida"])

    saida = Path(args.saida).expanduser().resolve()
    final = saida / "datasetfinal_MAE"
    bbox = saida / "dataset_bbox"
    for d in (final / "imagens", final / "mascaras", final / "qualidade"):
        d.mkdir(parents=True, exist_ok=True)
    if not args.sem_bbox:
        for d in (bbox / "imagens", bbox / "anotacoes" / "yolo"):
            d.mkdir(parents=True, exist_ok=True)

    hosts = sorted({l["origem"] for l in aprovadas})
    split_path = (REPO_ROOT / args.split) if args.split else None
    parte_de = particoes(hosts, split_path, estavel=args.particao_estavel)
    entrega_de = json.loads(Path(args.entregas).read_text(encoding="utf-8")) if args.entregas else {}
    politica_de = json.loads(Path(args.politicas).read_text(encoding="utf-8")) if args.politicas else {}
    reusar = Path(args.reusar_bbox).resolve() if args.reusar_bbox else None
    # so reaproveita a imagem desenhada se a caixa e o tipo forem os mesmos da
    # entrega anterior: o nome sozinho nao garante (uma parte refeita com outro
    # codigo teria o mesmo nome e outra lesao)
    caixa_anterior: Dict[str, tuple] = {}
    if reusar and (reusar.parent / "anotacoes" / "caixas.csv").is_file():
        with open(reusar.parent / "anotacoes" / "caixas.csv", newline="", encoding="utf-8") as fh:
            caixa_anterior = {r["arquivo"]: (r["x1"], r["y1"], r["x2"], r["y2"], r["tipo_lesao"])
                              for r in csv.DictReader(fh)}
    treino_lora, recortes_lora = set(), set()
    if split_path and split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        pac_de = split.get("paciente_de", {})
        treino_lora = {pac_de[c] for c in (split.get("recortes", {}).get("train") or []) if c in pac_de}
        recortes_lora = {pac_de[c] for lst in (split.get("recortes") or {}).values() for c in lst
                         if c in pac_de}
    incidencias = load_view_table(Path(args.incidencias)) if args.incidencias else None

    linhas_manifesto, linhas_caixas, coco_imgs, coco_anns = [], [], [], []
    a_desenhar = []
    for i, l in enumerate(sorted(aprovadas, key=lambda r: r["saida"]), 1):
        registro = lesao_por_saida.get(l["saida"])
        if registro is None:
            print("aviso: {} sem metadado; ignorada".format(l["saida"]), file=sys.stderr)
            continue
        lesao, meta = registro["lesao"], registro["meta"]
        origem_path = Path(meta["arquivo"])
        src_img = Path(lesao["files"]["original"])
        src_mask = Path(lesao["files"]["mask_original"])
        if not src_img.is_file():
            src_img = lote / "original" / src_img.name
        if not src_mask.is_file():
            src_mask = lote / "masks" / "original" / src_mask.name

        entregar(src_img, final / "imagens" / src_img.name, args.vincular)
        entregar(src_mask, final / "mascaras" / src_mask.name, args.vincular)

        g = lesao["box_original"]
        x1, y1, x2, y2 = [int(v) for v in g["pegada_lesao"]]
        w_img, h_img = int(meta["origem"]["width"]), int(meta["origem"]["height"])
        view = resolve_view(meta_nome(origem_path), incidencias)
        parte = parte_de.get(l["origem"], "train")
        pac = paciente(l["origem"])
        linhas_manifesto.append({
            "arquivo": src_img.name, "mascara": src_mask.name,
            "mamografia_origem": origem_path.name, "paciente": pac, "particao": parte,
            "variante": int(l["indice"]) + 1, "tipo_lesao": l["tipo"],
            "parte": l.get("parte", ""), "entrega": entrega_de.get(l.get("parte", ""), ""),
            "politica": politica_de.get(l.get("parte", ""), ""),
            "view": view.view, "lateralidade": view.laterality, "view_fonte": view.source,
            "largura_img": w_img, "altura_img": h_img,
            "bit_depth": int(meta["origem"]["bit_depth"]), "dtype": meta["origem"]["dtype"],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "largura_px": x2 - x1, "altura_px": y2 - y1,
            "diametro_mm": l.get("diametro_mm"), "mm_por_px": round(295.0 / max(w_img, h_img), 5),
            "regiao_anatomica": lesao.get("region"),
            "weber": l.get("weber_nativo"), "cnr": l.get("cnr_nativo"),
            "roi_delta_ratio": l.get("roi_delta_ratio"),
            "distancia_pele_mm": l.get("distancia_pele_mm"),
            "ruido_roi_anel": l.get("ruido_roi_anel"), "nitidez_roi_anel": l.get("nitidez_roi_anel"),
            "grade_z": l.get("grade_z"), "confianca": l.get("confianca"),
            "seed": l.get("seed"), "guidance": lesao["generation"].get("guidance"),
            "lora": Path(lesao["generation"].get("lora") or "").name or None,
            "fonte_igual_ao_treino": lesao["generation"].get("fonte_igual_ao_treino"),
            "paciente_em_treino_lora": pac in treino_lora,
            "paciente_nos_recortes_lora": pac in recortes_lora,
            "avisos_auditoria": l.get("avisos", ""),
            "ganho_contraste": (cal_por_saida.get(src_img.name, {}) or {}).get("ganho", ""),
            "crel": (fid_por_saida.get(src_img.name, {}) or {}).get("crel", ""),
            "conspic": (fid_por_saida.get(src_img.name, {}) or {}).get("conspic", ""),
            "radialidade": (fid_por_saida.get(src_img.name, {}) or {}).get("radialidade", ""),
            "sigma_ratio": (fid_por_saida.get(src_img.name, {}) or {}).get("sigma_ratio", ""),
            "avisos_fidelidade": (fid_por_saida.get(src_img.name, {}) or {}).get("avisos", ""),
        })
        linhas_caixas.append({
            "arquivo": src_img.name, "tipo_lesao": l["tipo"], "particao": parte,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "largura_px": x2 - x1, "altura_px": y2 - y1,
            "largura_img": w_img, "altura_img": h_img})
        coco_imgs.append({"id": i, "file_name": src_img.name, "width": w_img, "height": h_img})
        coco_anns.append({"id": i, "image_id": i, "category_id": CLASSES.index(l["tipo"]) + 1,
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                          "iscrowd": 0, "segmentation": []})

        if not args.sem_bbox:
            alvo = bbox / "imagens" / (src_img.stem + "_bbox.png")
            anterior = reusar / alvo.name if reusar else None
            mesma = caixa_anterior.get(src_img.name) == (str(x1), str(y1), str(x2), str(y2), l["tipo"])
            if anterior is not None and anterior.is_file() and mesma:
                entregar(anterior, alvo, True)
            else:
                a_desenhar.append((src_img, (x1, y1, x2, y2), l["tipo"], alvo))
            cx, cy = (x1 + x2) / 2.0 / w_img, (y1 + y2) / 2.0 / h_img
            bw, bh = (x2 - x1) / w_img, (y2 - y1) / h_img
            (bbox / "anotacoes" / "yolo" / (src_img.stem + ".txt")).write_text(
                "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(CLASSES.index(l["tipo"]), cx, cy, bw, bh),
                encoding="utf-8")
        if i % 25 == 0:
            print("  {} / {}".format(i, len(aprovadas)), flush=True)

    if a_desenhar:
        print("desenhando {} caixa(s) em {} processo(s)".format(len(a_desenhar), args.processos), flush=True)
        with ProcessPoolExecutor(max(1, args.processos)) as ex:
            list(ex.map(_desenhar, a_desenhar, chunksize=4))

    with open(final / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas_manifesto[0].keys()))
        w.writeheader()
        w.writerows(linhas_manifesto)

    for nome in ("auditoria.csv", "auditoria.json", "RELATORIO_QUALIDADE.md"):
        origem = lote / "auditoria" / nome
        if origem.is_file():
            shutil.copy2(origem, final / "qualidade" / nome)
    for nome in ("fidelidade.csv", "FIDELIDADE.md"):
        origem = lote / "fidelidade" / nome
        if origem.is_file():
            shutil.copy2(origem, final / "qualidade" / nome)
    origem = lote / "calibracao_contraste.csv"
    if origem.is_file():
        shutil.copy2(origem, final / "qualidade" / origem.name)
    for nome in ("RELATORIO_TECNICO.md", "manifest.csv", "summary.json", "processing.log"):
        origem = lote / nome
        if origem.is_file():
            shutil.copy2(origem, final / "qualidade" / ("lote_" + nome))

    if not args.sem_bbox:
        with open(bbox / "anotacoes" / "caixas.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(linhas_caixas[0].keys()))
            w.writeheader()
            w.writerows(linhas_caixas)
        (bbox / "anotacoes" / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
        usados = sorted({l["tipo_lesao"] for l in linhas_caixas})
        (bbox / "anotacoes" / "coco.json").write_text(json.dumps({
            "info": {"description": "lesoes sinteticas MAM-E + LoRA BI-RADS", "lote": lote.name},
            "images": coco_imgs, "annotations": coco_anns,
            "categories": [{"id": CLASSES.index(c) + 1, "name": c} for c in CLASSES if c in usados],
        }, indent=1, ensure_ascii=False), encoding="utf-8")

    contagem: Dict[str, int] = {}
    for l in linhas_manifesto:
        contagem[l["particao"]] = contagem.get(l["particao"], 0) + 1
    resumo = {"lote": lote.name, "lesoes": len(linhas_manifesto),
              "mamografias": len({l["mamografia_origem"] for l in linhas_manifesto}),
              "pacientes": len({l["paciente"] for l in linhas_manifesto}),
              "por_tipo": {t: sum(1 for l in linhas_manifesto if l["tipo_lesao"] == t)
                           for t in sorted({l["tipo_lesao"] for l in linhas_manifesto})},
              "por_particao": contagem}
    (saida / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("dataset final em: {}".format(saida))
    print("  {}".format(json.dumps(resumo, ensure_ascii=False)))
    return 0


def selecionar_estratificado(linhas: List[Dict[str, str]], fid: Dict[str, Dict[str, str]], n: int,
                             seed: int = 20260925) -> List[Dict[str, str]]:
    """N linhas com a mesma distribuicao conjunta de contraste x tamanho das aprovadas.

    Celulas = quintis de ``crel_afastado`` (da fidelidade) x tercis de diametro;
    cada celula recebe N x (fracao dela), arredondado pelo maior resto. Sem a
    medida de contraste, a lesao cai numa celula propria (nao some).
    """
    def num(v):
        try:
            x = float(v)
            return x if np.isfinite(x) else None
        except (TypeError, ValueError):
            return None
    cr = [num((fid.get(l["saida"]) or {}).get("crel_afastado")) for l in linhas]
    dm = [num(l.get("diametro_mm")) for l in linhas]
    qc = np.nanpercentile([c for c in cr if c is not None], [20, 40, 60, 80]) if any(c is not None for c in cr) else []
    qd = np.nanpercentile([d for d in dm if d is not None], [33.3, 66.7]) if any(d is not None for d in dm) else []
    celulas: Dict[tuple, List[Dict[str, str]]] = {}
    for l, c, d in zip(linhas, cr, dm):
        chave = (int(np.searchsorted(qc, c)) if c is not None else -1,
                 int(np.searchsorted(qd, d)) if d is not None else -1)
        celulas.setdefault(chave, []).append(l)
    total = len(linhas)
    cotas = {k: n * len(v) / total for k, v in celulas.items()}
    alvo = {k: int(np.floor(q)) for k, q in cotas.items()}
    for k in sorted(cotas, key=lambda k: -(cotas[k] - alvo[k]))[: n - sum(alvo.values())]:
        alvo[k] += 1
    rng = np.random.default_rng(seed)
    out: List[Dict[str, str]] = []
    for k in sorted(celulas):
        grupo = sorted(celulas[k], key=lambda r: r["saida"])
        out += [grupo[i] for i in sorted(rng.permutation(len(grupo))[: alvo[k]])]
    return sorted(out, key=lambda r: r["saida"])


def _desenhar(tarefa) -> None:
    desenhar(*tarefa)


def entregar(origem: Path, destino: Path, vincular: bool) -> None:
    """Copia, ou cria hardlink quando pedido (cai para copia entre discos)."""
    if destino.exists():
        destino.unlink()
    if vincular:
        try:
            os.link(origem, destino)
            return
        except OSError:
            pass
    shutil.copy2(origem, destino)


def meta_nome(path: Path) -> str:
    return path.stem


if __name__ == "__main__":
    raise SystemExit(main())
