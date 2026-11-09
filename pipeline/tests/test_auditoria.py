"""Portao da auditoria independente (scripts/auditar_dataset.py).

Duas regras decidem o que entra no dataset de entrega e falhariam em silencio:

* a severidade de cada criterio -- um criterio bloqueante fora da faixa reprova,
  um aviso nao; medida ausente nao pode virar aprovacao disfarcada;
* a deduplicacao entre variantes da mesma mama. Comparando todos os pares, A
  seria descartada por causa de B e C tambem por causa de B, mesmo com A e C em
  posicoes diferentes -- a supressao tem de ser gulosa, contra as **mantidas**.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from auditar_dataset import (CRITERIOS, deduplicar, forma_do_acrescimo, grade_score,  # noqa: E402
                             julgar, nucleo_acima_tecido, robust_z)

import numpy as np  # noqa: E402


def medidas_boas():
    """Um conjunto de medidas no centro de todas as faixas bloqueantes."""
    out = {}
    for nome, (_, sev, lo, hi) in ((k, v[:4]) for k, v in CRITERIOS.items()):
        if sev != "bloqueante":
            continue
        if lo == -np.inf:
            out[nome] = hi - 1.0
        elif hi == np.inf:
            out[nome] = lo + 1.0
        else:
            out[nome] = (lo + hi) / 2.0
    return out


class TestVeredito(unittest.TestCase):

    def test_tudo_no_centro_aprova(self):
        v = julgar(medidas_boas())
        self.assertEqual(v["veredito"], "aprovada")
        self.assertEqual(v["falhas"], [])

    def test_bloqueante_fora_da_faixa_reprova(self):
        m = medidas_boas()
        m["weber_nativo"] = CRITERIOS["weber_nativo"][3] + 0.5
        v = julgar(m)
        self.assertEqual(v["veredito"], "reprovada")
        self.assertIn("weber_nativo", v["falhas"])

    def test_aviso_fora_da_faixa_nao_reprova(self):
        m = medidas_boas()
        m["ruido_roi_anel"] = 0.1                       # aviso, nao bloqueante
        v = julgar(m)
        self.assertEqual(v["veredito"], "aprovada")
        self.assertIn("ruido_roi_anel", v["avisos"])

    def test_medida_ausente_ou_nan_nao_aprova_por_omissao(self):
        m = medidas_boas()
        m["grade_z"] = float("nan")
        v = julgar(m)
        self.assertIn("grade_z", v["nao_avaliados"])
        self.assertNotIn("grade_z", v["falhas"])
        # e o criterio nao vira "ok" no relatorio
        check = next(c for c in v["checks"] if c["criterio"] == "grade_z")
        self.assertIsNone(check["ok"])

    def test_integridade_e_intolerante(self):
        m = medidas_boas()
        m["fora_da_mascara_px"] = 1                     # um unico pixel ja reprova
        self.assertEqual(julgar(m)["veredito"], "reprovada")


class TestDeduplicacao(unittest.TestCase):

    @staticmethod
    def registro(nome, caixa, conf):
        return {"origem": "mama", "saida": nome, "arquivo_saida": "/inexistente/" + nome,
                "pegada": caixa, "confianca": conf, "medidas": {}}

    def test_sobreposicao_alta_descarta_a_de_menor_confianca(self):
        a = self.registro("a.png", [0, 0, 100, 100], 0.9)
        b = self.registro("b.png", [5, 5, 105, 105], 0.5)
        deduplicar([a, b])
        self.assertLessEqual(a["medidas"]["iou_max_variantes"], CRITERIOS["iou_max_variantes"][3])
        self.assertGreater(b["medidas"]["iou_max_variantes"], CRITERIOS["iou_max_variantes"][3])

    def test_supressao_e_gulosa_nao_por_pares(self):
        """A e C nao se sobrepoem: descartar B nao pode arrastar as duas."""
        a = self.registro("a.png", [0, 0, 100, 100], 0.7)
        b = self.registro("b.png", [50, 0, 150, 100], 0.9)     # cobre A e C parcialmente
        c = self.registro("c.png", [100, 0, 200, 100], 0.6)    # sem intersecao com A
        deduplicar([a, b, c])
        lim = CRITERIOS["iou_max_variantes"][3]
        # B e a mais confiavel: fica. A e C disputam so com B, com IoU 1/3 < 0,5.
        for r in (a, b, c):
            self.assertLessEqual(r["medidas"]["iou_max_variantes"], lim, r["saida"])

    def test_posicoes_distantes_nao_competem(self):
        a = self.registro("a.png", [0, 0, 100, 100], 0.9)
        b = self.registro("b.png", [500, 500, 600, 600], 0.5)
        deduplicar([a, b])
        self.assertEqual(b["medidas"]["iou_max_variantes"], 0.0)


class TestGrade(unittest.TestCase):

    def test_grade_detecta_periodicidade_injetada(self):
        rng = np.random.default_rng(7)
        base = rng.normal(128, 8, (128, 128)).astype(np.float32)
        grade = base.copy()
        grade[::8, :] += 25.0                            # blocos de 8 px, como o latente
        self.assertGreater(grade_score(grade, 8.0), 2.0 * grade_score(base, 8.0))

    def test_z_robusto_ignora_amostra_pequena(self):
        self.assertTrue(np.isnan(robust_z(1.0, [1.0, 2.0])))
        self.assertGreater(robust_z(9.0, [1.0, 1.1, 0.9, 1.05, 0.95, 1.0]), 5.0)


class TestManchaBrancaChapada(unittest.TestCase):
    """Massa no nivel do parenquima passa; acima de todo o tecido em volta vira aviso."""

    def _cena(self, acrescimo):
        rng = np.random.default_rng(0)
        src = (20000 + 3000 * rng.standard_normal((400, 400))).clip(0, 65535).astype(np.uint16)
        yy, xx = np.mgrid[:400, :400]
        disco = (yy - 200) ** 2 + (xx - 200) ** 2 < 50 ** 2
        mascara = np.zeros((400, 400), np.uint8)
        mascara[140:260, 140:260] = 1
        out = src.astype(np.float32)
        out[disco] += acrescimo
        return src, out.clip(0, 65535).astype(np.uint16), mascara, (145, 145, 255, 255)

    def test_massa_no_nivel_do_tecido_passa(self):
        src, out, m, pg = self._cena(3000)          # 1 desvio: some no p99,5 do vizinho
        v = nucleo_acima_tecido(src, out, m, pg)
        self.assertLess(v, CRITERIOS["nucleo_acima_tecido"][3])

    def test_mancha_acima_de_todo_tecido_vira_aviso(self):
        src, out, m, pg = self._cena(20000)         # muito acima do tecido mais claro
        v = nucleo_acima_tecido(src, out, m, pg)
        self.assertGreater(v, 0.9)
        # aviso, nao reprovacao: 47% das lesoes reais de birads_4/5 passariam de 0,25
        veredito = julgar(dict(medidas_boas(), nucleo_acima_tecido=v))
        self.assertNotIn("nucleo_acima_tecido", veredito["falhas"])
        self.assertIn("nucleo_acima_tecido", veredito["avisos"])

    def test_sem_acrescimo_nao_e_mancha(self):
        src, _, m, pg = self._cena(0)
        self.assertEqual(nucleo_acima_tecido(src, src, m, pg), 0.0)


class TestBlocoRetangular(unittest.TestCase):
    """O acrescimo em forma de bloco (cantos cheios) e marcado; a oval lisa nao."""

    def _forma(self, mascara):
        rng = np.random.default_rng(1)
        src = (20000 + 800 * rng.standard_normal((400, 400))).astype(np.float32)
        out = src + 6000 * mascara.astype(np.float32)
        return forma_do_acrescimo(src.astype(np.uint16), out.clip(0, 65535).astype(np.uint16), (120, 120, 280, 280))

    def test_quadrado_tem_cantos_cheios(self):
        m = np.zeros((400, 400), bool)
        m[130:270, 130:270] = True
        ext, cantos = self._forma(m)
        self.assertGreater(ext, 0.70)
        self.assertGreaterEqual(cantos, 0.33)

    def test_oval_lisa_nao_e_bloco(self):
        yy, xx = np.mgrid[:400, :400]
        m = ((yy - 200) / 70.0) ** 2 + ((xx - 200) / 55.0) ** 2 < 1
        ext, cantos = self._forma(m)
        self.assertFalse(ext > 0.70 and cantos >= 0.33)

    def test_bloco_reprova(self):
        self.assertIn("bloco_retangular", julgar(dict(medidas_boas(), bloco_retangular=1))["falhas"])
        self.assertNotIn("bloco_retangular", julgar(dict(medidas_boas(), bloco_retangular=0))["falhas"])


if __name__ == "__main__":
    unittest.main()
