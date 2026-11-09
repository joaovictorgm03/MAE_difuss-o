"""Invariantes da janela nativa (mamelesion.window).

Sao as garantias que, se quebrarem, produzem resultado errado sem erro visivel:
lesao fora da janela, mascara fora da grade do latente, ou pixels alterados fora
da mascara no arquivo entregue. Cada propriedade e checada em centenas de casos
aleatorios, alem dos exemplos de borda.

    .venv/bin/python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from mamelesion.masks import LesionBox
from mamelesion.window import (LATENT_GRID, Window, box_mask, extract, lesion_box_model,
                               native_footprint, plan_window, reference_max, restore_native_margin, floor_low_frequency,
                               to_u8, window_delta, writeback)

RNG = np.random.default_rng(12345)


def random_case(rng):
    h, w = int(rng.integers(900, 4000)), int(rng.integers(900, 3200))
    lw, lh = int(rng.integers(40, 900)), int(rng.integers(40, 900))
    center = (float(rng.uniform(-50, w + 50)), float(rng.uniform(-50, h + 50)))
    factor = float(rng.uniform(1.2, 2.6))
    return (h, w), (lw, lh), center, factor


class TestPlanWindow(unittest.TestCase):

    def test_janela_sempre_dentro_da_imagem_e_contem_a_pegada(self):
        for _ in range(500):
            shape, wh, center, factor = random_case(RNG)
            if max(wh) > min(shape):
                continue
            fx0, fy0, fx1, fy1 = native_footprint(center, wh, shape)
            win = plan_window(((fx0 + fx1) / 2, (fy0 + fy1) / 2), wh, shape, factor)
            self.assertGreaterEqual(win.x0, 0)
            self.assertGreaterEqual(win.y0, 0)
            self.assertLessEqual(win.x1, shape[1])
            self.assertLessEqual(win.y1, shape[0])
            self.assertTrue(win.x0 <= fx0 and fx1 <= win.x1 and win.y0 <= fy0 and fy1 <= win.y1,
                            "pegada {} fora da janela {}".format((fx0, fy0, fx1, fy1), win))

    def test_lado_minimo_e_o_do_modelo_e_lesao_pequena_fica_em_escala_nativa(self):
        win = plan_window((1000, 1000), (100, 120), (3000, 2000), context_factor=2.0)
        self.assertEqual(win.side, 512)
        self.assertEqual(win.scale, 1.0)

    def test_lesao_grande_reduz_na_medida_do_contexto_sem_teto(self):
        win = plan_window((1500, 1500), (600, 400), (3000, 3000), context_factor=2.0, min_scale=0)
        self.assertEqual(win.side, 1200)
        self.assertAlmostEqual(win.scale, 512 / 1200)

    def test_teto_de_reducao_encolhe_o_contexto_e_nao_a_lesao(self):
        # 600 px com contexto 2,0 pediria lado 1200 (escala 0,43 < 0,5): cai para 1024
        win = plan_window((1500, 1500), (600, 400), (3000, 3000), 2.0, min_scale=0.5, min_context=1.25)
        self.assertEqual(win.side, 1024)
        self.assertAlmostEqual(win.scale, 0.5)
        # 900 px: nem o teto garante 0,5 com contexto minimo 1,25 -> lado 1125
        win = plan_window((1500, 1500), (900, 700), (3000, 3000), 2.0, min_scale=0.5, min_context=1.25)
        self.assertEqual(win.side, 1125)

    def test_teto_nao_afeta_lesao_tipica(self):
        a = plan_window((1500, 1500), (323, 300), (3000, 3000), 2.0, min_scale=0.5)
        b = plan_window((1500, 1500), (323, 300), (3000, 3000), 2.0, min_scale=0)
        self.assertEqual(a, b)

    def test_lesao_maior_que_a_imagem_levanta(self):
        with self.assertRaises(ValueError):
            plan_window((100, 100), (900, 900), (800, 800))

    def test_ida_e_volta_de_coordenadas(self):
        win = Window(137, 2210, 777)
        for _ in range(100):
            x, y = RNG.uniform(137, 914), RNG.uniform(2210, 2987)
            u, v = win.to_model(x, y)
            xb, yb = win.to_native(u, v)
            self.assertAlmostEqual(x, xb, places=6)
            self.assertAlmostEqual(y, yb, places=6)


class TestLesionBox(unittest.TestCase):

    def test_caixa_na_grade_do_latente_e_cobre_a_lesao(self):
        for _ in range(500):
            shape, wh, center, factor = random_case(RNG)
            if max(wh) > min(shape):
                continue
            fx0, fy0, fx1, fy1 = native_footprint(center, wh, shape)
            c = ((fx0 + fx1) / 2, (fy0 + fy1) / 2)
            win = plan_window(c, wh, shape, factor)
            box = lesion_box_model(win, c, wh, margin_frac=0.10)
            for v in (box.x1, box.y1):
                self.assertEqual(v % LATENT_GRID, 0)
            # x2/y2 so saem da grade quando cortados na borda do modelo
            for v in (box.x2, box.y2):
                self.assertTrue(v % LATENT_GRID == 0 or v == win.model_res)
            u0, v0 = win.to_model(fx0, fy0)
            u1, v1 = win.to_model(fx1, fy1)
            tol = 1e-6
            self.assertLessEqual(box.x1, max(0.0, u0) + tol)
            self.assertLessEqual(box.y1, max(0.0, v0) + tol)
            self.assertGreaterEqual(box.x2, min(win.model_res, u1) - tol)
            self.assertGreaterEqual(box.y2, min(win.model_res, v1) - tol)

    def test_sem_margem_a_grade_ainda_cobre_a_lesao(self):
        """Com margem zero nada absorve o arredondamento: ele tem de ser para fora."""
        for _ in range(500):
            shape, wh, center, factor = random_case(RNG)
            if max(wh) > min(shape):
                continue
            fx0, fy0, fx1, fy1 = native_footprint(center, wh, shape)
            c = ((fx0 + fx1) / 2, (fy0 + fy1) / 2)
            win = plan_window(c, wh, shape, factor)
            box = lesion_box_model(win, c, wh, margin_frac=0.0)
            u0, v0 = win.to_model(fx0, fy0)
            u1, v1 = win.to_model(fx1, fy1)
            self.assertLessEqual(box.x1, u0 + 1e-6)
            self.assertLessEqual(box.y1, v0 + 1e-6)
            self.assertGreaterEqual(box.x2, min(win.model_res, u1) - 1e-6)
            self.assertGreaterEqual(box.y2, min(win.model_res, v1) - 1e-6)

    def test_box_mask_e_semiaberta(self):
        mask = box_mask(LesionBox(8, 16, 40, 64), 128)
        self.assertEqual(int((mask > 0).sum()), 32 * 48)
        self.assertEqual(mask[16, 8], 255)
        self.assertEqual(mask[63, 39], 255)
        self.assertEqual(mask[64, 39], 0)
        self.assertEqual(mask[16, 40], 0)


class TestIntensity(unittest.TestCase):

    def test_reference_max(self):
        self.assertEqual(reference_max(np.zeros((4, 4), np.uint8)), 255.0)
        a = np.zeros((4, 4), np.uint16)
        a[0, 0] = 65535
        self.assertEqual(reference_max(a), 65535.0)

    def test_extract_escala_nativa_sem_reamostrar(self):
        raw = RNG.integers(0, 65536, (1200, 1000), dtype=np.uint16)
        win = Window(100, 200, 512)
        out = extract(raw, win, 65535.0)
        np.testing.assert_allclose(out, raw[200:712, 100:612] / 65535.0, rtol=0, atol=1e-6)

    def test_extract_fora_da_imagem_levanta(self):
        with self.assertRaises(ValueError):
            extract(np.zeros((600, 600), np.uint16), Window(200, 0, 512), 65535.0)

    def test_to_u8_arredonda_e_satura(self):
        np.testing.assert_array_equal(to_u8(np.array([-0.1, 0.0, 0.5, 1.0, 1.2])),
                                      np.array([0, 0, 128, 255, 255], np.uint8))


class TestWriteback(unittest.TestCase):

    def _check(self, raw, win, ref):
        res = win.model_res
        x1, y1 = (int(v) for v in RNG.integers(0, res - 64, 2))
        box = LesionBox(x1, y1, x1 + int(RNG.integers(16, 64)), y1 + int(RNG.integers(16, 64)))
        mask = box_mask(box, res)
        delta = RNG.normal(0, 40, (res, res)).astype(np.float32)   # delta sujo tambem fora
        out, applied = writeback(raw, win, delta, mask, ref)
        self.assertEqual(out.dtype, raw.dtype)
        self.assertEqual(out.shape, raw.shape)
        outside = applied == 0
        np.testing.assert_array_equal(out[outside], raw[outside])
        self.assertGreater(int((out != raw).sum()), 0)
        self.assertTrue(np.all(applied[out != raw] == 255))

    def test_zero_pixels_alterados_fora_da_mascara_16_bits(self):
        for side in (512, 700, 1333):
            raw = RNG.integers(0, 65536, (2000, 1800), dtype=np.uint16)
            self._check(raw, Window(int(RNG.integers(0, 400)), int(RNG.integers(0, 600)), side), 65535.0)

    def test_zero_pixels_alterados_fora_da_mascara_8_bits(self):
        raw = RNG.integers(0, 256, (1500, 1500), dtype=np.uint8)
        self._check(raw, Window(300, 300, 900), 255.0)

    def test_delta_nulo_e_identidade(self):
        raw = RNG.integers(0, 65536, (1000, 1000), dtype=np.uint16)
        win = Window(10, 20, 640)
        mask = box_mask(LesionBox(100, 100, 300, 300), 512)
        out, _ = writeback(raw, win, np.zeros((512, 512), np.float32), mask, 65535.0)
        np.testing.assert_array_equal(out, raw)

    def test_satura_no_dtype(self):
        raw = np.full((600, 600), 65000, np.uint16)
        mask = box_mask(LesionBox(0, 0, 512, 512), 512)
        out, _ = writeback(raw, Window(0, 0, 512), np.full((512, 512), 255.0, np.float32), mask, 65535.0)
        self.assertEqual(int(out[:512, :512].max()), 65535)

    def test_escala_de_intensidade_do_delta(self):
        raw = np.full((600, 600), 1000, np.uint16)
        mask = box_mask(LesionBox(0, 0, 512, 512), 512)
        out, _ = writeback(raw, Window(0, 0, 512), np.full((512, 512), 10.0, np.float32), mask, 65535.0)
        self.assertEqual(int(out[100, 100]), 1000 + round(10 * 65535 / 255))


class TestRestoreNativeMargin(unittest.TestCase):

    def _par(self):
        rng = np.random.default_rng(3)
        orig = np.clip(rng.normal(120, 12, (512, 512)), 0, 255).astype(np.uint8)
        gen = cv2_blur(orig.astype(np.float32), 2.0) + 40.0      # modelo: mais claro e mais liso
        gen[200:312, 200:312] += rng.normal(0, 15, (112, 112))  # lesao com textura propria
        return orig, np.clip(gen, 0, 255)

    def test_sigma_zero_devolve_a_saida_do_modelo(self):
        orig, gen = self._par()
        np.testing.assert_array_equal(restore_native_margin(orig, gen, (200, 200, 312, 312), 0.0),
                                      gen.astype(np.float32))

    def test_margem_recupera_textura_original_e_lesao_fica(self):
        orig, gen = self._par()
        out = restore_native_margin(orig, gen, (200, 200, 312, 312), 3.0)
        hp = lambda a: a - cv2_blur(a, 3.0)                                  # noqa: E731
        margem = (slice(20, 120), slice(20, 120))
        np.testing.assert_allclose(hp(out)[margem].std(), hp(orig.astype(np.float32))[margem].std(), rtol=0.10)
        self.assertLess(hp(gen)[margem].std(), 0.6 * hp(orig.astype(np.float32))[margem].std())
        centro = (slice(236, 276), slice(236, 276))
        np.testing.assert_allclose(out[centro], gen[centro], atol=1e-3)
        self.assertAlmostEqual(float((out - orig)[margem].mean()), float((gen - orig)[margem].mean()), delta=1.0)


class TestFloorLowFrequency(unittest.TestCase):

    def test_halo_escuro_some_e_lesao_clara_fica(self):
        rng = np.random.default_rng(5)
        orig = np.clip(rng.normal(120, 10, (256, 256)), 0, 255).astype(np.uint8)
        src = orig.astype(np.float32).copy()
        yy, xx = np.mgrid[:256, :256]
        rr = np.hypot(yy - 128, xx - 128)
        src[rr < 30] += 60                                   # nucleo claro
        src[(rr >= 45) & (rr < 70)] -= 25                    # halo escuro aprendido
        src += rng.normal(0, 4, src.shape)                   # textura fina com sinal
        out = floor_low_frequency(orig, src, -3.0)
        halo = (rr >= 50) & (rr < 65)
        antes = float((src - orig)[halo].mean())
        depois = float((out - orig)[halo].mean())
        self.assertLess(antes, -20)                          # halo de -25 niveis no cenario
        self.assertGreaterEqual(depois, -3.5)                # piso de -3 na baixa frequencia
        self.assertGreater(float((out - orig)[rr < 20].mean()), 50)
        hf = lambda a: a - cv2_blur(a, 3.0)                  # noqa: E731
        self.assertLess(float(hf(out - orig)[rr > 100].min()), -3)   # textura fina ainda negativa

    def test_none_desliga(self):
        orig = np.full((64, 64), 100, np.uint8)
        src = np.full((64, 64), 80, np.float32)
        np.testing.assert_array_equal(floor_low_frequency(orig, src, None), src)


def cv2_blur(a, s):
    import cv2
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), s)


class TestWindowDelta(unittest.TestCase):

    def test_delta_nulo_fora_da_mascara(self):
        orig = RNG.integers(0, 256, (512, 512), dtype=np.uint8)
        gen = RNG.integers(0, 256, (512, 512), dtype=np.uint8)
        box = LesionBox(96, 128, 320, 288)
        mask = box_mask(box, 512)
        d = window_delta(orig, gen, mask, box)
        self.assertTrue(np.all(d[mask == 0] == 0))
        self.assertGreater(float(np.abs(d[mask > 0]).max()), 0)

    def test_interior_recebe_ganho_total(self):
        orig = np.full((512, 512), 100, np.uint8)
        gen = np.full((512, 512), 100, np.uint8)
        box = LesionBox(96, 96, 416, 416)
        gen[200:300, 200:300] = 160
        mask = box_mask(box, 512)
        d = window_delta(orig, gen, mask, box, offset_correction=True)
        self.assertAlmostEqual(float(d[250, 250]), 60.0, places=3)


if __name__ == "__main__":
    unittest.main()
