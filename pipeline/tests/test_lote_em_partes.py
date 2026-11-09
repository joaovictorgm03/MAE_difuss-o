"""Orquestrador de lote em partes (scripts/lote_em_partes.py) e montagem incremental.

Garantias que falham em silencio num lote de horas sem supervisao:

* codigo de saida 1 e ambiguo (lesao reprovada OU excecao do Python): a etapa so
  conta se deixou a saida completa e nova;
* uma entrega na fila nao pode disparar outra igual, e uma entrega que falhou
  tem de ser refeita;
* a particao de um paciente nao muda quando a entrega cresce.
"""

import csv
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from lote_em_partes import Orquestrador  # noqa: E402
from montar_dataset_final import particoes, selecionar_estratificado  # noqa: E402


def _csv(path: Path, linhas, campos):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)


class ConferirSaida(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        # 3 lesoes + 1 hospedeira sem lesao (linha com arquivo_saida vazio)
        _csv(self.d / "manifest.csv", [{"arquivo_saida": "a"}, {"arquivo_saida": "b"},
                                        {"arquivo_saida": "c"}, {"arquivo_saida": ""}], ["arquivo_saida"])
        self.o = Orquestrador.__new__(Orquestrador)
        self.t0 = time.time() - 1

    def test_sem_arquivo_e_falha(self):
        self.assertIn("nao foi gravado", self.o.conferir_saida("auditar", self.d, self.t0))

    def test_arquivo_velho_e_falha(self):
        f = self.d / "auditoria" / "auditoria.csv"
        _csv(f, [{"saida": x} for x in "abc"], ["saida"])
        os.utime(f, (self.t0 - 100, self.t0 - 100))
        self.assertIn("nao foi gravado", self.o.conferir_saida("auditar", self.d, self.t0))

    def test_auditoria_incompleta_e_falha(self):
        _csv(self.d / "auditoria" / "auditoria.csv", [{"saida": x} for x in "ab"], ["saida"])
        self.assertIn("esperado 3", self.o.conferir_saida("auditar", self.d, self.t0))

    def test_completa_passa(self):
        _csv(self.d / "calibracao_contraste.csv", [{"saida": x} for x in "abc"], ["saida"])
        self.assertIsNone(self.o.conferir_saida("calibrar", self.d, self.t0))

    def test_fidelidade_pode_pular_lesao_nao_medivel(self):
        _csv(self.d / "fidelidade" / "fidelidade.csv", [{"saida": "a"}], ["saida"])
        self.assertIsNone(self.o.conferir_saida("fidelidade", self.d, self.t0))

    def test_etapa_sem_saida_propria_nao_confere(self):
        self.assertIsNone(self.o.conferir_saida("verificar", self.d, self.t0))


class AgendarEntrega(unittest.TestCase):
    def _orq(self, aprovadas):
        o = Orquestrador.__new__(Orquestrador)
        o.a = SimpleNamespace(entrega_a_cada=500, entregar_politicas="")
        o.estado = SimpleNamespace(d={})
        o.arq_entregas = Path(tempfile.mkdtemp()) / "entregas.json"
        o.entregas = {"partes": {}, "entregas": []}
        o._lock_entregas = threading.RLock()
        o.futuros_entrega = []
        o.aprovadas = lambda n: aprovadas[n]
        o.agendadas = []
        o.pool_entrega = SimpleNamespace(submit=lambda f, k, partes: o.agendadas.append((k, partes)))
        return o

    def test_abaixo_do_limiar_nao_entrega(self):
        o = self._orq({"p0": 300})
        o.agendar_entrega(["p0"])
        self.assertEqual(o.agendadas, [])

    def test_entrega_na_fila_nao_dispara_outra(self):
        o = self._orq({"p0": 520, "p1": 90})
        o.agendar_entrega(["p0"])
        o.agendar_entrega(["p0", "p1"])          # 610: ainda nao chegou a 1000
        self.assertEqual([k for k, _ in o.agendadas], [1])

    def test_segunda_entrega_no_proximo_multiplo(self):
        o = self._orq({"p0": 520, "p1": 490})
        o.agendar_entrega(["p0"])
        o.agendar_entrega(["p0", "p1"])
        self.assertEqual(o.agendadas, [(1, ["p0"]), (2, ["p0", "p1"])])
        self.assertEqual(o.entregas["partes"], {"p0": 1, "p1": 2})

    def test_entrega_que_falhou_e_refeita_no_fim(self):
        o = self._orq({"p0": 520})
        o.agendar_entrega(["p0"])
        o.entregas["entregas"][-1]["status"] = "erro"
        o.agendar_entrega(["p0"], final=True)
        self.assertEqual([k for k, _ in o.agendadas], [1, 2])
        self.assertEqual(o.entregas["partes"]["p0"], 2)

    def test_final_ja_coberta_nao_repete(self):
        o = self._orq({"p0": 520})
        o.agendar_entrega(["p0"])
        o.entregas["entregas"][-1]["status"] = "ok"
        o.agendar_entrega(["p0"], final=True)
        self.assertEqual(len(o.agendadas), 1)


class EntregaPorPolitica(AgendarEntrega):
    def test_so_partes_da_politica_pedida_entram(self):
        o = self._orq({"p0": 900, "p1": 600})
        o.a.entregar_politicas = "v2,v3"
        o.estado = SimpleNamespace(d={"p0": {"politica": "v1"}, "p1": {"politica": "v3"}})
        o.agendar_entrega(["p0", "p1"])
        self.assertEqual(o.agendadas, [(1, ["p1"])])

    def test_sem_parte_da_politica_nao_entrega(self):
        o = self._orq({"p0": 900})
        o.a.entregar_politicas = "v3"
        o.estado = SimpleNamespace(d={"p0": {"politica": "v1"}})
        o.agendar_entrega(["p0"], final=True)
        self.assertEqual(o.agendadas, [])


class ParticaoEstavel(unittest.TestCase):
    def test_paciente_nao_muda_quando_a_entrega_cresce(self):
        antes = particoes(["100_1_L_CC.png", "200_1_R_CC.png"], None, estavel=True)
        depois = particoes(["100_1_L_CC.png", "200_1_R_CC.png"] +
                           ["{}_9_L_CC.png".format(i) for i in range(300, 400)], None, estavel=True)
        for h, parte in antes.items():
            self.assertEqual(depois[h], parte)

    def test_mesmo_paciente_mesma_particao(self):
        p = particoes(["{}_{}_L_CC.png".format(pac, e) for pac in range(50) for e in (1, 2)], None,
                      estavel=True)
        for pac in range(50):
            self.assertEqual(p["{}_1_L_CC.png".format(pac)], p["{}_2_L_CC.png".format(pac)])

    def test_fracoes_aproximadas(self):
        p = particoes(["{}_1_L_CC.png".format(i) for i in range(4000)], None, estavel=True)
        frac = sum(v == "train" for v in p.values()) / len(p)
        self.assertAlmostEqual(frac, 0.70, delta=0.03)


class SelecaoEstratificada(unittest.TestCase):
    def _dados(self, n=1000):
        rng = np.random.default_rng(0)
        linhas = [{"saida": "l{:04d}.png".format(i), "diametro_mm": str(10 + 40 * rng.random())} for i in range(n)]
        fid = {l["saida"]: {"crel_afastado": str(rng.random())} for l in linhas}
        return linhas, fid

    def test_exatamente_n_e_deterministico(self):
        linhas, fid = self._dados()
        a = selecionar_estratificado(linhas, fid, 400)
        self.assertEqual(len(a), 400)
        self.assertEqual(len({l["saida"] for l in a}), 400)
        self.assertEqual(a, selecionar_estratificado(linhas, fid, 400))

    def test_cobre_todas_as_faixas_de_contraste(self):
        linhas, fid = self._dados()
        a = selecionar_estratificado(linhas, fid, 400)
        c = np.array([float(fid[l["saida"]]["crel_afastado"]) for l in a])
        for lo in (0.0, 0.2, 0.4, 0.6, 0.8):          # ~80 por quintil
            self.assertTrue(60 <= ((c >= lo) & (c < lo + 0.2)).sum() <= 100)


if __name__ == "__main__":
    unittest.main()
