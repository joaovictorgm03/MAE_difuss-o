#!/usr/bin/env python
"""Gera um lote grande em partes retomaveis e monta o dataset de entrega no fim.

Um lote de milhares de lesoes leva horas. Este programa divide as hospedeiras
em partes, e cada parte passa pela mesma cadeia que ja validou os lotes
anteriores, com um marcador por etapa:

    gerar (GPU) -> verificar -> calibrar -> verificar -> auditar -> fidelidade

A GPU gera uma parte enquanto a CPU pos-processa a anterior. Se o processo
morrer (queda de energia, ``kill``), rodar o MESMO comando retoma de onde parou:
etapas com marcador sao puladas; uma parte cuja geracao nao terminou e apagada e
refeita inteira (a semente e por parte, entao o resultado e o mesmo).

Entregas incrementais: cada vez que o total de lesoes aprovadas passa de um
multiplo de ``--entrega-a-cada`` (500), e de novo no fim, a entrega e remontada
com todas as partes prontas: ``unificar_partes`` junta as partes num lote unico
(hardlinks, sem copiar bytes), ``montar_dataset_final.py`` monta em
``<saida>.novo``, ``verificar_entrega.py`` confere cada arquivo contra a origem,
``relatorio_qualidade_lote.py`` escreve o relatorio, e so entao a pasta nova
TROCA a antiga (duas renomeacoes: a pasta some por milissegundos, mas nunca
aparece pela metade). A particao
e por hash do paciente, entao nao muda entre entregas, e o manifesto diz em que
``entrega`` cada imagem entrou. Uma entrega que falha deixa a anterior no lugar.

Para rodar desacoplado da sessao (sobrevive a queda do SSH)::

    setsid nohup .venv/bin/python scripts/lote_em_partes.py ... > saida.log 2>&1 < /dev/null &

Estado a qualquer momento: ``<lote>/_controle/estado.json`` e
``<lote>/_controle/orquestrador.log``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = str(REPO_ROOT / ".venv" / "bin" / "python")
SCRIPTS = REPO_ROOT / "scripts"
LOG = logging.getLogger("lote_em_partes")

ETAPAS_POS = ("verificar", "calibrar", "verificar_calibrado", "auditar", "fidelidade")
#: o que cada etapa tem de deixar gravado. O codigo de saida nao basta: 1 quer
#: dizer "alguma lesao reprovada" (esperado), mas uma excecao nao tratada do
#: Python TAMBEM sai com 1 -- e uma calibracao que caiu no meio deixaria a parte
#: com lesoes metade calibradas, marcada como ok.
SAIDA_DA_ETAPA = {"calibrar": "calibracao_contraste.csv", "auditar": "auditoria/auditoria.csv",
                  "fidelidade": "fidelidade/fidelidade.csv"}


class Estado:
    """``estado.json``: o que ja foi feito em cada parte. Gravado a cada mudanca."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.d: Dict[str, Dict[str, object]] = {}
        if path.is_file():
            self.d = json.loads(path.read_text(encoding="utf-8"))

    def marcar(self, parte: str, etapa: str, valor: object) -> None:
        with self._lock:
            self.d.setdefault(parte, {})[etapa] = valor
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.d, indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)


def rodar(cmd: List[str], log_path: Path, env_extra: Optional[Dict[str, str]] = None) -> int:
    """Roda um passo com a saida num arquivo proprio; devolve o codigo de saida."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", **(env_extra or {}))
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("\n$ {}\n".format(" ".join(cmd)))
        fh.flush()
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT), env=env,
                               stdin=subprocess.DEVNULL)


def contar_linhas(path: Path, campo: Optional[str] = None) -> int:
    with open(path, newline="", encoding="utf-8") as fh:
        return sum(1 for r in csv.DictReader(fh) if not campo or r.get(campo))


def livre_gb(path: Path) -> float:
    return shutil.disk_usage(str(path)).free / 1e9


def planejar(hospedeiras: Path, controle: Path, tamanho: int, semente: int) -> List[Path]:
    """Divide as hospedeiras em partes embaralhadas; reusa o plano se ja existir.

    Embaralhar faz qualquer prefixo das partes ser uma amostra representativa --
    importa se a guarda de disco interromper a geracao antes do fim.
    """
    pasta = controle / "partes"
    existentes = sorted(pasta.glob("parte_*.txt"))
    if existentes:
        return existentes
    nomes = [l.strip() for l in hospedeiras.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = np.random.default_rng(semente)
    ordem = list(rng.permutation(len(nomes)))
    pasta.mkdir(parents=True, exist_ok=True)
    saida = []
    for k in range(0, len(nomes), tamanho):
        p = pasta / "parte_{:03d}.txt".format(k // tamanho)
        p.write_text("".join(nomes[i] + "\n" for i in sorted(ordem[k:k + tamanho])), encoding="utf-8")
        saida.append(p)
    return saida


class Orquestrador:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.entrada = Path(a.entrada).resolve()
        self.lote = Path(a.lote).resolve()
        self.controle = self.lote / "_controle"
        self.partes_dir = self.lote / "partes"
        self.incidencias = Path(a.incidencias).resolve()
        self.estado = Estado(self.controle / "estado.json")
        self.falhou_integridade: List[str] = []
        # limita threads do pos-processamento: maquina compartilhada, e a GPU
        # tambem precisa de CPU para preparar cada imagem
        self.env_cpu = {"OMP_NUM_THREADS": str(a.threads_cpu), "OPENBLAS_NUM_THREADS": str(a.threads_cpu),
                        "MKL_NUM_THREADS": str(a.threads_cpu), "OPENCV_FOR_THREADS_NUM": str(a.threads_cpu)}

    # -- geracao ---------------------------------------------------------

    def gerar(self, idx: int, lista: Path) -> bool:
        nome = lista.stem
        destino = self.partes_dir / nome
        if self.estado.d.get(nome, {}).get("gerar") == "ok" and (destino / "summary.json").is_file():
            return True
        if destino.exists():                   # geracao interrompida: refaz a parte inteira
            LOG.warning("%s: geracao incompleta anterior, refazendo", nome)
            shutil.rmtree(destino)
        destino.mkdir(parents=True)
        cmd = [PY, str(SCRIPTS / "build_synthetic_dataset.py"),
               "--input-dir", str(self.entrada), "--lista", str(lista),
               "--incidencias", str(self.incidencias), "--output-dir", str(destino),
               "--per-image", str(self.a.por_imagem), "--seed", str(self.a.semente + idx),
               "--lora", self.a.lora, "--no-canvas"]
        for s in self.a.set:
            cmd += ["--set", s]
        for s in self.a.anatomia:
            cmd += ["--anatomia", s]
        if self.a.rotulo_saida:
            cmd += ["--rotulo-saida", self.a.rotulo_saida]
        t0 = time.time()
        codigo = rodar(cmd, self.controle / "logs" / (nome + "_gerar.log"))
        if codigo >= 2 or not (destino / "summary.json").is_file():
            self.estado.marcar(nome, "gerar", "erro_{}".format(codigo))
            return False
        resumo = json.loads((destino / "summary.json").read_text(encoding="utf-8"))
        self.estado.marcar(nome, "gerar", "ok")
        self.estado.marcar(nome, "politica", self.a.politica)
        self.estado.marcar(nome, "gerar_resumo", dict(resumo, duracao_total_s=round(time.time() - t0, 1)))
        LOG.info("%s gerada: %s lesoes em %.0f s (livre %.1f GB)", nome, resumo.get("lesoes"),
                 time.time() - t0, livre_gb(self.lote))
        return True

    # -- pos-processamento ------------------------------------------------

    def conferir_saida(self, etapa: str, destino: Path, t0: float) -> Optional[str]:
        """Motivo de falha se a etapa nao deixou a saida completa; None se deixou."""
        rel = SAIDA_DA_ETAPA.get(etapa)
        if rel is None:
            return None
        f = destino / rel
        if not f.is_file() or f.stat().st_mtime < t0:
            return "{} nao foi gravado".format(rel)
        n = contar_linhas(f)
        esperado = contar_linhas(destino / "manifest.csv", "arquivo_saida")
        # a fidelidade pula lesao que nao da para medir (janela cortada pela borda):
        # ela so precisa existir; calibracao e auditoria cobrem TODA lesao gerada
        if etapa == "fidelidade":
            return None if n > 0 or esperado == 0 else "{} vazio".format(rel)
        return None if n == esperado else "{}: {} linhas, esperado {}".format(rel, n, esperado)

    def pos(self, nome: str) -> str:
        try:
            return self._pos(nome)
        except Exception:                       # disco cheio, arquivo sumido...: a parte sai, o lote segue
            LOG.exception("%s: pos-processamento abortado -- parte fora da entrega", nome)
            self.estado.marcar(nome, "pos", "excecao")
            return "erro"

    def _pos(self, nome: str) -> str:
        destino = self.partes_dir / nome
        log = self.controle / "logs" / (nome + "_pos.log")
        feito = self.estado.d.get(nome, {})
        passos = {
            "verificar": [PY, str(SCRIPTS / "verify_dataset.py"), str(destino),
                          "--input-dir", str(self.entrada), "--quiet"],
            "calibrar": [PY, str(SCRIPTS / "calibrar_contraste.py"), str(destino),
                         "--entrada", str(self.entrada), "--reais", self.a.reais]
                        + (["--ganho-fixo", str(self.a.ganho_fixo)] if self.a.calibracao == "fixo" else [])
                        + (["--ganho-max", str(self.a.ganho_max)] if self.a.ganho_max else []),
            "verificar_calibrado": [PY, str(SCRIPTS / "verify_dataset.py"), str(destino),
                                    "--input-dir", str(self.entrada), "--quiet"],
            "auditar": [PY, str(SCRIPTS / "auditar_dataset.py"), str(destino),
                        "--incidencias", str(self.incidencias)],
            "fidelidade": [PY, str(SCRIPTS / "fidelidade_real.py"), str(destino),
                           "--reais", self.a.reais, "--entrada", str(self.entrada)]
                          + (["--contraste-afastado"] if self.a.contraste_afastado else []),
        }
        for etapa in ETAPAS_POS:
            if feito.get(etapa) == "ok":
                continue
            if self.a.calibracao == "nenhuma" and etapa in ("calibrar", "verificar_calibrado"):
                self.estado.marcar(nome, etapa, "ok")
                self.estado.marcar(nome, etapa + "_s", "pulada: --calibracao nenhuma")
                continue
            t0 = time.time()
            if etapa in SAIDA_DA_ETAPA and (destino / SAIDA_DA_ETAPA[etapa]).is_file():
                (destino / SAIDA_DA_ETAPA[etapa]).unlink()      # saida velha nao pode passar por nova
            codigo = rodar(passos[etapa], log, self.env_cpu)
            # verificar: qualquer falha e bug de composicao -> a parte nao entra.
            # demais: 0 = tudo aprovado, 1 = alguma lesao reprovada (esperado), 2 = erro
            limite = 0 if etapa.startswith("verificar") else 1
            motivo = "codigo {}".format(codigo) if codigo > limite else self.conferir_saida(etapa, destino, t0)
            if motivo:
                self.estado.marcar(nome, etapa, "erro: " + motivo)
                LOG.error("%s: etapa %s falhou (%s) -- parte fora da entrega; ver %s",
                          nome, etapa, motivo, log)
                if etapa.startswith("verificar"):
                    self.falhou_integridade.append(nome)
                return "erro"
            self.estado.marcar(nome, etapa, "ok")
            self.estado.marcar(nome, etapa + "_s", round(time.time() - t0, 1))
            if etapa == "calibrar":
                # o cru so serve para recalibrar; com ganho fixo o CSV ja registra
                # o ganho, e ele custa 5 MB por lesao de 16 bits
                cru = destino / "original_sem_calibracao"
                if cru.is_dir() and not self.a.manter_cru:
                    shutil.rmtree(cru)
        LOG.info("%s pos-processada (livre %.1f GB)", nome, livre_gb(self.lote))
        return "ok"

    # -- entregas incrementais ----------------------------------------------

    def aprovadas(self, nome: str) -> int:
        """Lesoes da parte aprovadas na auditoria E na fidelidade (as que entram na entrega)."""
        d = self.partes_dir / nome
        with open(d / "auditoria" / "auditoria.csv", newline="", encoding="utf-8") as fh:
            aud = {r["saida"]: r["veredito"] for r in csv.DictReader(fh)}
        with open(d / "fidelidade" / "fidelidade.csv", newline="", encoding="utf-8") as fh:
            fid = {r["saida"]: r["veredito"] for r in csv.DictReader(fh)}
        return sum(1 for s, v in aud.items() if v == "aprovada" and fid.get(s) == "aprovada")

    def _gravar_entregas(self) -> None:
        # a thread principal agenda e a de entrega atualiza o mesmo dicionario:
        # sem o lock, um json.dumps no meio de um update levanta RuntimeError
        with self._lock_entregas:
            tmp = self.arq_entregas.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.entregas, indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.arq_entregas)

    def _cobertas(self) -> set:
        """Partes da ultima entrega que deu certo ou ainda esta na fila."""
        for e in reversed(self.entregas["entregas"]):
            if e.get("status") in ("ok", "agendada"):
                return set(e["partes"])
        return set()

    def agendar_entrega(self, prontas: List[str], final: bool = False) -> None:
        """Agenda uma entrega com as partes prontas, se houver motivo para uma nova."""
        if self.a.entregar_politicas:           # pasta de entrega so com estas politicas
            quer = {x.strip() for x in self.a.entregar_politicas.split(",") if x.strip()}
            prontas = [n for n in prontas if self.estado.d.get(n, {}).get("politica") in quer]
            if not prontas:
                return
        with self._lock_entregas:
            novas = [n for n in prontas if n not in self._cobertas()]
            if not novas:
                return
            total = sum(self.aprovadas(n) for n in prontas)
            feitas = sum(1 for e in self.entregas["entregas"] if e.get("status") in ("ok", "agendada"))
            if not final and total < (feitas + 1) * self.a.entrega_a_cada:
                return
            k = len(self.entregas["entregas"]) + 1
            for n in novas:                       # entrega em que a imagem aparece pela 1a vez
                self.entregas["partes"][n] = k
            self.entregas["entregas"].append({"entrega": k, "partes": sorted(prontas), "aprovadas": total,
                                              "final": final, "status": "agendada"})
            self._gravar_entregas()
        LOG.info("entrega %d agendada: %d aprovadas em %d partes", k, total, len(prontas))
        self.futuros_entrega.append(self.pool_entrega.submit(self.entregar, k, sorted(prontas)))

    def entregar(self, k: int, partes: List[str]) -> str:
        """unificar -> montar em <saida>.novo -> verificar -> relatorio -> troca da pasta."""
        from unificar_partes import unificar          # noqa: E402 (scripts/ no path)
        with self._lock_entregas:
            registro = self.entregas["entregas"][k - 1]
            registro["inicio"] = time.strftime("%Y-%m-%d %H:%M:%S")
            mapa_partes = dict(self.entregas["partes"])
        try:
            unificar(self.lote, [self.partes_dir / n for n in partes])
            self.estado.marcar("_lote", "partes_prontas", partes)
            saida = Path(self.a.saida_dataset).resolve()
            novo = saida.with_name(saida.name + ".novo")
            if novo.exists():
                shutil.rmtree(novo)
            mapa = self.controle / (self.arq_entregas.stem + "_entrega_de_parte.json")
            mapa.write_text(json.dumps(mapa_partes), encoding="utf-8")
            pol = self.controle / (self.arq_entregas.stem + "_politica_de_parte.json")
            pol.write_text(json.dumps({n: self.estado.d.get(n, {}).get("politica", "") for n in partes}),
                           encoding="utf-8")
            montar = [PY, str(SCRIPTS / "montar_dataset_final.py"), str(self.lote), "--saida", str(novo),
                      "--fidelidade", "auto", "--calibracao", "auto", "--incidencias", str(self.incidencias),
                      "--vincular", "--particao-estavel", "--entregas", str(mapa), "--politicas", str(pol)]
            if self.a.selecionar:
                montar += ["--selecionar", str(self.a.selecionar)]
            if (saida / "dataset_bbox" / "imagens").is_dir():
                montar += ["--reusar-bbox", str(saida / "dataset_bbox" / "imagens")]
            passos = [
                ("montar", montar),
                ("verificar_entrega", [PY, str(SCRIPTS / "verificar_entrega.py"), str(novo / "datasetfinal_MAE"),
                                       "--entrada", str(self.entrada)]),
                ("relatorio", [PY, str(SCRIPTS / "relatorio_qualidade_lote.py"), str(self.lote),
                               "--dataset", str(novo), "--entrada", str(self.entrada), "--nome", saida.name,
                               "--reais", self.a.reais, "--entregas-arquivo", self.arq_entregas.name]),
            ]
            log = self.controle / "logs" / "{}_{:02d}.log".format(self.arq_entregas.stem, k)
            for etapa, cmd in passos:
                codigo = rodar(cmd, log, self.env_cpu)
                if codigo != 0:
                    raise RuntimeError("{} falhou (codigo {}); ver {}".format(etapa, codigo, log))
            (novo / ".origem_lote").write_text("{}\nentrega {}\n".format(self.lote, k), encoding="utf-8")
            # troca: quem esta usando a entrega nunca ve a pasta pela metade
            antigo = saida.with_name(saida.name + ".antigo")
            if saida.exists():
                if not (saida / ".origem_lote").is_file():
                    raise RuntimeError("{} existe e nao foi criada por este lote; nada foi trocado "
                                       "(a entrega nova ficou em {})".format(saida, novo))
                if antigo.exists():
                    if not (antigo / ".origem_lote").is_file():
                        raise RuntimeError("{} existe e nao foi criada por este lote".format(antigo))
                    shutil.rmtree(antigo)
                saida.rename(antigo)            # entre as duas renomeacoes a pasta some por ms
            novo.rename(saida)
            if antigo.exists():
                shutil.rmtree(antigo)
            n_img = contar_linhas(saida / "datasetfinal_MAE" / "manifest.csv")
            with self._lock_entregas:
                registro.update(status="ok", fim=time.strftime("%Y-%m-%d %H:%M:%S"), imagens_entregues=n_img)
            LOG.info("ENTREGA %d pronta em %s: %d imagens (%d partes)", k, saida,
                     registro["imagens_entregues"], len(partes))
            return "ok"
        except Exception as exc:
            with self._lock_entregas:
                registro.update(status="erro", erro=str(exc), fim=time.strftime("%Y-%m-%d %H:%M:%S"))
            LOG.exception("entrega %d falhou; a entrega anterior continua no lugar", k)
            return "erro"
        finally:
            self._gravar_entregas()

    # -- execucao ---------------------------------------------------------

    def executar(self) -> int:
        (self.controle / "logs").mkdir(parents=True, exist_ok=True)
        listas = planejar(Path(self.a.hospedeiras), self.controle, self.a.tamanho_parte, self.a.semente)
        if self.a.max_partes:
            listas = listas[: self.a.max_partes]
        LOG.info("%d partes de ate %d imagens, %d lesoes por imagem; entrega a cada %d aprovadas; livre %.1f GB",
                 len(listas), self.a.tamanho_parte, self.a.por_imagem, self.a.entrega_a_cada,
                 livre_gb(self.lote))
        self.arq_entregas = self.controle / self.a.entregas_arquivo
        self.entregas = json.loads(self.arq_entregas.read_text(encoding="utf-8")) \
            if self.arq_entregas.is_file() else {"partes": {}, "entregas": []}
        self._lock_entregas = threading.RLock()
        for e in self.entregas["entregas"]:       # entrega que morreu com o processo anterior
            if e.get("status") == "agendada":
                e["status"] = "interrompida"
        self.pool_entrega = ThreadPoolExecutor(max_workers=1)
        self.futuros_entrega: List[Future] = []

        pool = ThreadPoolExecutor(max_workers=self.a.pos_paralelo)
        futuros: Dict[str, Future] = {}
        ordem: List[str] = []
        erros_seguidos = 0

        def prontas() -> List[str]:
            return [n for n in ordem if futuros[n].done() and futuros[n].exception() is None
                    and futuros[n].result() == "ok"]

        for idx, lista in enumerate(listas):
            nome = lista.stem
            if not (self.estado.d.get(nome, {}).get("gerar") == "ok"):
                if livre_gb(self.lote) < self.a.disco_min_gb:
                    LOG.error("disco livre %.1f GB < %.1f GB: geracao interrompida antes de %s; "
                              "a entrega final sera montada com as partes ja geradas",
                              livre_gb(self.lote), self.a.disco_min_gb, nome)
                    self.estado.marcar("_lote", "interrompido_por_disco", nome)
                    break
            ok = self.gerar(idx, lista)
            if not ok:
                erros_seguidos += 1
                LOG.error("%s: geracao falhou (%d seguida(s)); ver logs/%s_gerar.log",
                          nome, erros_seguidos, nome)
                if erros_seguidos >= 2:       # erro de ambiente: nao queimar as partes restantes
                    LOG.error("duas partes seguidas falharam: abortando a geracao")
                    break
                continue
            erros_seguidos = 0
            ordem.append(nome)
            futuros[nome] = pool.submit(self.pos, nome)
            if not self.a.sem_entrega:
                self.agendar_entrega(prontas())
        pool.shutdown(wait=True)

        finais = prontas()
        if self.a.entregar_politicas:
            quer = {x.strip() for x in self.a.entregar_politicas.split(",") if x.strip()}
            finais_entrega = [n for n in finais if self.estado.d.get(n, {}).get("politica") in quer]
        else:
            finais_entrega = finais
        LOG.info("partes prontas: %d de %d planejadas", len(finais), len(listas))
        if not finais:
            return 2
        if self.a.sem_entrega:
            self.pool_entrega.shutdown(wait=True)
            return 0
        # a entrega final tem de cobrir TODAS as partes prontas; se a ultima falhou,
        # tenta de novo (uma vez) em vez de terminar com uma entrega velha
        for tentativa in range(2):
            self.agendar_entrega(finais_entrega, final=True)
            for f in list(self.futuros_entrega):
                try:
                    f.result()
                except Exception:
                    LOG.exception("entrega terminou com excecao")
            ult = self.entregas["entregas"][-1] if self.entregas["entregas"] else {}
            if ult.get("status") == "ok" and set(ult.get("partes", [])) >= set(finais_entrega):
                break
            LOG.error("entrega final nao cobriu todas as partes (tentativa %d)", tentativa + 1)
        self.pool_entrega.shutdown(wait=True)
        ult = self.entregas["entregas"][-1] if self.entregas["entregas"] else {}
        erros = [] if ult.get("status") == "ok" and set(ult.get("partes", [])) >= set(finais_entrega) else [ult]
        LOG.info("CONCLUIDO: %d entrega(s), ultima com %s imagens em %s", len(self.entregas["entregas"]),
                 (self.entregas["entregas"] or [{}])[-1].get("imagens_entregues"), self.a.saida_dataset)
        return 1 if (self.falhou_integridade or erros) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", required=True, help="pasta das mamografias de origem")
    ap.add_argument("--lote", required=True, help="pasta do lote (partes, controle e lote unificado)")
    ap.add_argument("--hospedeiras", required=True, help="incluidas.txt de selecionar_hospedeiras.py")
    ap.add_argument("--incidencias", required=True, help="incidencias.csv de selecionar_hospedeiras.py")
    ap.add_argument("--saida-dataset", required=True, help="pasta da entrega")
    ap.add_argument("--por-imagem", type=int, default=4)
    ap.add_argument("--tamanho-parte", type=int, default=60)
    ap.add_argument("--semente", type=int, default=20260923)
    ap.add_argument("--lora", default="treinos/birads_lora_v1")
    ap.add_argument("--calibracao", choices=("fixo", "quantil", "nenhuma"), default="fixo",
                    help="fixo: --ganho-fixo em todas (lote_variacoes_v5); quantil: casa o contraste de cada "
                         "lesao com a distribuicao das reais (padrao de calibrar_contraste.py); nenhuma: "
                         "a lesao como o gerador entregou")
    ap.add_argument("--ganho-fixo", type=float, default=1.25,
                    help="ganho de densidade com --calibracao fixo (o do lote_variacoes_v5)")
    ap.add_argument("--reais", default="/home/joao/medgan2/dados/birrads_5",
                    help="recortes reais: alvo da calibracao por quantil e faixas da fidelidade")
    ap.add_argument("--ganho-max", type=float, default=None, help="teto de ganho da calibracao por quantil")
    ap.add_argument("--contraste-afastado", action="store_true",
                    help="fidelidade julga o contraste na coroa afastada (a regua da calibracao)")
    ap.add_argument("--set", action="append", default=[], help="repassado ao gerador")
    ap.add_argument("--anatomia", action="append", default=[], metavar="CHAVE=VALOR",
                    help="repassado ao gerador: campo de AnatomyConfig (ex.: density_upper_percentile=75)")
    ap.add_argument("--selecionar", type=int, default=None,
                    help="repassado a montagem: entrega exatamente N aprovadas, estratificadas")
    ap.add_argument("--rotulo-saida", default="", help="repassado ao gerador: sufixo do nome de cada saida")
    ap.add_argument("--politica", default="v1",
                    help="rotulo gravado em cada parte GERADA nesta execucao (coluna politica do manifesto); "
                         "descreva o rotulo em <lote>/_controle/politicas.json")
    ap.add_argument("--max-partes", type=int, default=None, help="so as N primeiras partes (piloto)")
    ap.add_argument("--disco-min-gb", type=float, default=20.0,
                    help="nao inicia uma parte nova com menos que isso livre no disco")
    ap.add_argument("--pos-paralelo", type=int, default=2, help="partes pos-processadas ao mesmo tempo")
    ap.add_argument("--threads-cpu", type=int, default=6, help="threads por processo de pos-processamento")
    ap.add_argument("--manter-cru", action="store_true", help="nao apagar original_sem_calibracao/")
    ap.add_argument("--sem-entrega", action="store_true", help="para depois das partes (piloto)")
    ap.add_argument("--entregar-politicas", default="",
                    help="so partes destas politicas (ex.: v2,v3) entram na pasta de entrega; vazio = todas")
    ap.add_argument("--entregas-arquivo", default="entregas.json",
                    help="registro das entregas em _controle/: um por pasta de entrega (numeracao e limiar proprios)")
    ap.add_argument("--entrega-a-cada", type=int, default=500,
                    help="remonta a entrega cada vez que o total de aprovadas passa de um multiplo disto")
    a = ap.parse_args(argv)

    lote = Path(a.lote).resolve()
    (lote / "_controle").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(lote / "_controle" / "orquestrador.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    (lote / "_controle" / "orquestrador.pid").write_text(str(os.getpid()), encoding="utf-8")
    LOG.info("inicio: %s", " ".join(sys.argv))
    try:
        return Orquestrador(a).executar()
    except Exception:
        LOG.exception("orquestrador abortado")
        return 2


if __name__ == "__main__":
    sys.path.insert(0, str(SCRIPTS))
    raise SystemExit(main())
