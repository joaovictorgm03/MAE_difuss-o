"""Divisao por paciente e leitura dos recortes (mamelesion.birads_data).

Vazamento de paciente entre treino e teste nao gera erro nenhum -- so infla as
metricas de semelhanca, porque a mesma lesao (em CC e MLO) aparece dos dois
lados. Estes testes fecham essa porta.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mamelesion.birads_data import (CROP_REFERENCE, discover_crops, parse_ids, read_crop,
                                    size_catalog, split_by_patient)


def items_sinteticos(rng, n_pac_a=45, n_pac_b=15):
    items = []
    for stratum, n in (("birads5", n_pac_a), ("birads4", n_pac_b)):
        for p in range(n):
            patient = "{}{:03d}".format(stratum[-1], p)
            for k in range(int(rng.integers(1, 5))):
                items.append(("{}_{}_L_CC_box_{:03d}.png".format(patient, p, k), patient, stratum))
    return items


class TestParseIds(unittest.TestCase):

    def test_convencao(self):
        self.assertEqual(parse_ids("1400_32156_L_CC_2_box_000.png"), ("1400", "1400_32156"))
        self.assertEqual(parse_ids("35290_34620_R_MLO_box_000.png"), ("35290", "35290_34620"))

    def test_fora_da_convencao_levanta(self):
        with self.assertRaises(ValueError):
            parse_ids("6278_IM-0002_CCE_L.png")


class TestSplit(unittest.TestCase):

    def setUp(self):
        self.items = items_sinteticos(np.random.default_rng(7))
        self.split = split_by_patient(self.items, seed=99)
        self.patient_of = {k: p for k, p, _ in self.items}
        self.stratum_of = {k: s for k, _, s in self.items}

    def test_nenhum_paciente_em_duas_particoes(self):
        parts = {name: {self.patient_of[k] for k in keys} for name, keys in self.split.items()}
        self.assertFalse(parts["train"] & parts["val"])
        self.assertFalse(parts["train"] & parts["test"])
        self.assertFalse(parts["val"] & parts["test"])

    def test_todos_os_itens_atribuidos_uma_vez(self):
        todos = sorted(k for keys in self.split.values() for k in keys)
        self.assertEqual(todos, sorted(k for k, _, _ in self.items))

    def test_cada_particao_tem_as_duas_classes(self):
        for name, keys in self.split.items():
            self.assertEqual({self.stratum_of[k] for k in keys}, {"birads4", "birads5"}, name)

    def test_deterministico_e_sensivel_a_semente(self):
        self.assertEqual(self.split, split_by_patient(self.items, seed=99))
        self.assertNotEqual(self.split, split_by_patient(self.items, seed=100))

    def test_proporcoes_aproximadas_por_paciente(self):
        pacientes = {p for _, p, _ in self.items}
        n_train = len({self.patient_of[k] for k in self.split["train"]})
        self.assertAlmostEqual(n_train / len(pacientes), 0.70, delta=0.05)

    def test_paciente_em_duas_classes_nao_e_sorteado_duas_vezes(self):
        items = self.items + [("b5000_99999_R_MLO_box_009.png", "5000", "birads4")]
        split = split_by_patient(items, seed=3)
        onde = [name for name, keys in split.items() for k in keys if self.patient_of.get(k, "5000") == "5000"
                or k.startswith("b5000")]
        self.assertEqual(len(set(onde)), 1)

    def test_estrato_pequeno_demais_levanta(self):
        with self.assertRaises(ValueError):
            split_by_patient([("a.png", "1", "x"), ("b.png", "2", "x")], fractions=(0.2, 0.4, 0.4))

    def test_fracoes_invalidas_levantam(self):
        with self.assertRaises(ValueError):
            split_by_patient(self.items, fractions=(0.5, 0.3, 0.3))


class TestLeitura(unittest.TestCase):

    def test_escala_fixa_e_rejeicoes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            a = np.full((100, 120), 30000, np.uint16); a[40:60, 50:70] = 52000
            Image.fromarray(a).save(d / "1_2_L_CC_box_000.png")
            Image.fromarray(np.zeros((100, 100), np.uint8)).save(d / "3_4_R_CC_box_000.png")
            (d / "5_6_L_MLO_box_000.png").write_bytes(b"\x89PNG\r\n\x1a\n truncado")
            Image.fromarray(np.full((30, 30), 9, np.uint16)).save(d / "7_8_L_CC_box_000.png")

            img = read_crop(d / "1_2_L_CC_box_000.png")
            self.assertAlmostEqual(float(img.max()), 52000 / CROP_REFERENCE, places=6)

            crops, rejected = discover_crops({"birads5": [d]})
            self.assertEqual([c.name for c in crops], ["1_2_L_CC_box_000.png"])
            self.assertEqual(crops[0].wh, (120, 100))
            self.assertEqual(len(rejected), 3)
            self.assertEqual(size_catalog(crops), {"birads5": [[120, 100]]})

    def test_classe_desconhecida_levanta(self):
        with self.assertRaises(ValueError):
            discover_crops({"birads3": []})


if __name__ == "__main__":
    unittest.main()
