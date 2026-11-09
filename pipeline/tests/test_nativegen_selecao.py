"""Geracao nativa completa com um modelo falso, criterio de selecao e coerencia dos padroes.

Sem GPU: ``generate_fn`` e uma funcao que pinta uma lesao sintetica. Verifica o
encadeamento inteiro de :func:`nativegen.generate_native` -- a garantia de zero
pixels fora da mascara tem de valer ponta a ponta, nao so em ``writeback``.
"""

import inspect
import unittest
from pathlib import Path

import numpy as np
import yaml

from mamelesion import window
from mamelesion.config import REPO_ROOT, Config, NativeWindowConfig
from mamelesion.lesion_eval import contrast_ratio, selection_key
from mamelesion.nativegen import generate_native
from mamelesion.paste import PasteConfig


def fake_model(image_u8, mask_u8, prompt, seed, steps, guidance, negative):
    """Modelo falso: devolve a janela com um disco claro no centro da mascara e ruido fora."""
    rng = np.random.default_rng(seed)
    out = np.clip(image_u8.astype(int) + rng.integers(-20, 21, image_u8.shape), 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(mask_u8 > 127)
    cy, cx, r = ys.mean(), xs.mean(), 0.35 * min(np.ptp(ys), np.ptp(xs))
    yy, xx = np.mgrid[:image_u8.shape[0], :image_u8.shape[1]]
    out[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = 220
    return out


class TestGenerateNative(unittest.TestCase):

    def test_ponta_a_ponta_zero_fora_da_mascara_16_e_8_bits(self):
        rng = np.random.default_rng(0)
        nw = NativeWindowConfig()
        for dtype, top in ((np.uint16, 65535), (np.uint8, 255)):
            for _ in range(8):
                raw = rng.integers(0, top // 2, (2400, 1900)).astype(dtype)
                wh = (int(rng.integers(80, 700)), int(rng.integers(80, 700)))
                center = (float(rng.uniform(0, 1900)), float(rng.uniform(0, 2400)))
                nl = generate_native(fake_model, raw, float(top), center, wh, nw, "p", int(rng.integers(1e6)))
                outside = nl.mask_native == 0
                np.testing.assert_array_equal(nl.raw_out[outside], raw[outside])
                self.assertGreater(int((nl.raw_out != raw).sum()), 0)
                self.assertEqual(nl.raw_out.dtype, raw.dtype)
                crop = nl.lesion_crop(float(top))
                self.assertEqual(crop.shape, (wh[1], wh[0]))
                self.assertGreater(float(crop.mean()), float(raw[nl.footprint[1]:nl.footprint[3],
                                                                  nl.footprint[0]:nl.footprint[2]].mean()) / top)

    def test_reprodutivel(self):
        raw = np.random.default_rng(1).integers(0, 30000, (2000, 2000)).astype(np.uint16)
        a = generate_native(fake_model, raw, 65535.0, (1000, 900), (300, 260), NativeWindowConfig(), "p", 7)
        b = generate_native(fake_model, raw, 65535.0, (1000, 900), (300, 260), NativeWindowConfig(), "p", 7)
        np.testing.assert_array_equal(a.raw_out, b.raw_out)


class TestSelecao(unittest.TestCase):

    def _crops(self, contrast, n=12, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            c = np.clip(rng.normal(0.4, 0.03, (120, 110)), 0, 1).astype(np.float32)
            c[40:80, 35:75] += contrast
            out.append(np.clip(c, 0, 1))
        return out

    def test_razao_de_contraste_detecta_ausencia_de_lesao(self):
        real = self._crops(0.3)
        self.assertAlmostEqual(contrast_ratio(real, self._crops(0.3, seed=1)), 1.0, delta=0.15)
        self.assertLess(contrast_ratio(real, self._crops(0.0, seed=2)), 0.2)

    def _summary(self, auc, ratio, memo, kid=1.0):
        return {"auc_classificador": {"media": auc}, "razao_contraste": ratio,
                "memorizacao": {"fracao_acima": memo}, "kid": {"media": kid}}

    def test_sem_lesao_perde_mesmo_com_auc_e_kid_melhores(self):
        """O erro real da primeira validacao: base sem lesao com KID menor."""
        sem_lesao = self._summary(auc=0.60, ratio=0.05, memo=0.0, kid=0.5)
        com_lesao = self._summary(auc=0.80, ratio=0.90, memo=0.0, kid=1.5)
        self.assertLess(selection_key(com_lesao), selection_key(sem_lesao))

    def test_memorizacao_torna_inelegivel(self):
        copia = self._summary(auc=0.55, ratio=1.0, memo=0.60)
        honesto = self._summary(auc=0.75, ratio=1.0, memo=0.05)
        self.assertLess(selection_key(honesto), selection_key(copia))

    def test_entre_elegiveis_vence_menor_auc(self):
        self.assertLess(selection_key(self._summary(0.62, 1.1, 0.0)), selection_key(self._summary(0.71, 1.0, 0.0)))

    def _full(self, auc, ic_hi, ratio, memo, kid, kid_sd, tex):
        return {"auc_classificador": {"media": auc, "ic95": [auc - 0.02, ic_hi]}, "razao_contraste": ratio,
                "memorizacao": {"fracao_acima": memo}, "kid": {"media": kid, "desvio": kid_sd},
                "textura_mediana_w1": tex}

    def test_auc_saturado_nao_decide_por_ruido(self):
        """O caso real: AUC 0,98 x 0,99 com IC sobreposto e KID empatado -> decide a textura."""
        from mamelesion.lesion_eval import select_best
        cands = [self._full(0.98, 0.99, 0.71, 0.0, 0.824, 0.10, 0.57),
                 self._full(0.99, 0.99, 0.84, 0.0, 0.830, 0.10, 0.42),
                 self._full(0.99, 1.00, 0.68, 0.0, 0.811, 0.10, 0.73)]
        best, trail = select_best(cands)
        self.assertEqual(best, 1)
        self.assertTrue(any("textura" in t for t in trail))

    def test_auc_fora_do_ic_decide(self):
        from mamelesion.lesion_eval import select_best
        cands = [self._full(0.70, 0.75, 1.0, 0.0, 0.9, 0.1, 0.9), self._full(0.95, 0.99, 1.0, 0.0, 0.5, 0.1, 0.1)]
        self.assertEqual(select_best(cands)[0], 0)

    def test_inelegivel_nunca_vence_elegivel(self):
        from mamelesion.lesion_eval import select_best
        cands = [self._full(0.50, 0.55, 0.05, 0.0, 0.1, 0.1, 0.1), self._full(0.99, 1.0, 0.9, 0.0, 1.5, 0.1, 0.9)]
        self.assertEqual(select_best(cands)[0], 1)


class TestEscolhaDeTipo(unittest.TestCase):

    def _pipeline(self, types, weights):
        from mamelesion.pipeline import PipelineOptions, SyntheticLesionPipeline
        obj = SyntheticLesionPipeline.__new__(SyntheticLesionPipeline)   # sem carregar modelo
        obj.options = PipelineOptions(Path("."), Path("."), lesion_types=types, lesion_type_weights=weights)
        return obj

    def test_sem_pesos_alterna_como_antes(self):
        pipe = self._pipeline(["mass", "nodule", "density"], None)
        self.assertEqual([pipe._choose_type(s, 0) for s in range(6)],
                         ["mass", "nodule", "density", "mass", "nodule", "density"])

    def test_pesos_reproduzem_a_proporcao(self):
        import zlib
        pipe = self._pipeline(["birads5", "birads4"], [0.83, 0.17])
        tipos = [pipe._choose_type(zlib.crc32("img{}".format(i).encode()), 0) for i in range(4000)]
        self.assertAlmostEqual(tipos.count("birads4") / len(tipos), 0.17, delta=0.02)
        self.assertEqual(pipe._choose_type(123, 0), pipe._choose_type(123, 0))

    @unittest.skipUnless((REPO_ROOT.parent / "sem_anomalia").is_dir(), "sem_anomalia ausente")
    def test_proporcao_nos_nomes_reais_e_uniformidade(self):
        """Regressao: com crc32 os 66 nomes reais davam 33% de BI-RADS 4 (KS p=0,003)."""
        import hashlib
        import zlib
        from scipy.stats import kstest
        pipe = self._pipeline(["birads5", "birads4"], [0.83, 0.17])
        stems = sorted(p.stem for p in (REPO_ROOT.parent / "sem_anomalia").glob("*.png"))
        stables = [zlib.crc32(n.encode("utf-8")) for n in stems]
        tipos = [pipe._choose_type(st, 0) for st in stables]
        self.assertLessEqual(abs(tipos.count("birads4") / len(tipos) - 0.17), 0.08)
        u = [int(hashlib.sha256("{}:0".format(st).encode()).hexdigest()[:12], 16) / 16 ** 12 for st in stables]
        self.assertGreater(kstest(u, "uniform").pvalue, 0.01)


class TestPadroesCoerentes(unittest.TestCase):
    """A geometria que o LoRA aprende nao pode divergir entre config, YAML, treino e funcoes."""

    CAMPOS = ("context_factor", "min_scale", "min_context", "mask_margin_frac", "paste_feather_frac")

    def test_dataclass_yaml_e_pasteconfig(self):
        yaml_nw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))["native_window"]
        dc = NativeWindowConfig()
        pc = PasteConfig()
        for campo in self.CAMPOS:
            self.assertEqual(getattr(dc, campo), yaml_nw[campo], "config.py x config.yaml: " + campo)
            self.assertEqual(getattr(pc, campo), yaml_nw[campo], "PasteConfig x config.yaml: " + campo)
        self.assertEqual(dc.feather_frac, yaml_nw["feather_frac"])
        self.assertEqual(dc.margin_texture_sigma, yaml_nw["margin_texture_sigma"])
        self.assertEqual(dc.density_floor, yaml_nw["density_floor"])
        self.assertEqual(Config.load().native_window.context_factor, yaml_nw["context_factor"])

    def test_padroes_das_funcoes(self):
        dc = NativeWindowConfig()
        pw = inspect.signature(window.plan_window).parameters
        self.assertEqual(pw["context_factor"].default, dc.context_factor)
        self.assertEqual(pw["min_scale"].default, dc.min_scale)
        self.assertEqual(pw["min_context"].default, dc.min_context)
        self.assertEqual(inspect.signature(window.lesion_box_model).parameters["margin_frac"].default,
                         dc.mask_margin_frac)
        self.assertEqual(inspect.signature(window.window_delta).parameters["feather_frac"].default,
                         dc.feather_frac)


if __name__ == "__main__":
    unittest.main()
