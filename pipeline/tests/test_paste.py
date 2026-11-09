"""Amostras de treino por colagem (mamelesion.paste).

Usa um hospedeiro sintetico com mapa anatomico e enquadramento conhecidos: assim
da para afirmar onde a lesao deve cair e o que a mascara deve cobrir.
"""

import unittest

import numpy as np

from mamelesion.anatomy import PlausibilityMap
from mamelesion.birads_data import LesionCrop
from mamelesion.masks import LesionBox
from mamelesion.paste import Host, PasteConfig, augment_crop, feather_rect, make_sample
from mamelesion.preprocessing import Framing
from mamelesion.window import LATENT_GRID


def host_sintetico(seed=0, h=3000, w=2400):
    rng = np.random.default_rng(seed)
    raw = (rng.normal(30000, 2000, (h, w))).clip(0, 65535).astype(np.uint16)
    raw[0, 0] = 65535
    scale = 1024 / h
    pad_x = int((1024 - w * scale) // 2)
    framing = Framing(w, h, scale, 0, 0, pad_x, 0, 1024, "breast_crop")
    breast = np.zeros((1024, 1024), np.uint8)
    breast[:, pad_x:1024 - pad_x] = 255
    score = np.zeros((1024, 1024), np.float32)
    score[150:870, pad_x + 60:1024 - pad_x - 60] = 1.0
    z = np.zeros((1024, 1024), np.float32)
    pmap = PlausibilityMap(score, z + 30, z + 80, np.zeros_like(breast), breast, "left",
                           (512, 900), "sintetico")
    return Host("h", "1", raw, 65535.0, framing, pmap, 0.28)


def crop_sintetico(h=300, w=260):
    img = np.full((h, w), 0.40, np.float32)
    yy, xx = np.mgrid[:h, :w]
    img[((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (min(h, w) / 3) ** 2] = 0.75
    return LesionCrop("1_2_L_CC_box_000.png", None, "birads5", "1", "1_2", img)


class TestPaste(unittest.TestCase):

    def test_formas_tipos_e_mascara_na_grade(self):
        host, crop, cfg = host_sintetico(), crop_sintetico(), PasteConfig()
        rng = np.random.default_rng(1)
        for _ in range(40):
            s = make_sample(crop, host, rng, cfg)
            self.assertEqual(s.image_u8.shape, (512, 512))
            self.assertEqual(s.image_u8.dtype, np.uint8)
            self.assertEqual(set(np.unique(s.mask_u8)) <= {0, 255}, True)
            self.assertEqual(s.box.x1 % LATENT_GRID, 0)
            self.assertEqual(s.box.y1 % LATENT_GRID, 0)
            self.assertEqual(int((s.mask_u8 > 0).sum()), s.box.w * s.box.h)
            self.assertEqual(s.placement, "anatomia")

    def test_mascara_cobre_a_pegada_da_lesao(self):
        host, crop, cfg = host_sintetico(), crop_sintetico(), PasteConfig()
        rng = np.random.default_rng(2)
        for _ in range(60):
            s = make_sample(crop, host, rng, cfg)
            fx0, fy0, fx1, fy1 = s.footprint
            u0, v0 = s.window.to_model(fx0, fy0)
            u1, v1 = s.window.to_model(fx1, fy1)
            self.assertTrue(s.box.x1 <= u0 and s.box.y1 <= v0 and s.box.x2 >= min(512, u1)
                            and s.box.y2 >= min(512, v1), (s.box, (u0, v0, u1, v1)))

    def test_lesao_aparece_dentro_e_fora_da_mascara_e_so_tecido(self):
        host, crop = host_sintetico(), crop_sintetico()
        cfg = PasteConfig(scale_jitter=(1.0, 1.0), flip_horizontal=False, flip_vertical=False)
        s = make_sample(crop, host, np.random.default_rng(3), cfg)
        inside = s.image_u8[s.mask_u8 > 0].astype(float)
        outside = s.image_u8[s.mask_u8 == 0].astype(float)
        self.assertGreater(inside.max(), outside.mean() + 60)       # nucleo 0,75 vs fundo ~0,46
        self.assertLess(abs(np.median(outside) - 30000 / 65535 * 255), 3)

    def test_reprodutivel_com_a_mesma_semente(self):
        host, crop, cfg = host_sintetico(), crop_sintetico(), PasteConfig()
        a = make_sample(crop, host, np.random.default_rng(42), cfg)
        b = make_sample(crop, host, np.random.default_rng(42), cfg)
        np.testing.assert_array_equal(a.image_u8, b.image_u8)
        self.assertEqual(a.meta(), b.meta())

    def test_casamento_de_nivel_alinha_o_fundo(self):
        host, crop = host_sintetico(), crop_sintetico()
        cfg = PasteConfig(scale_jitter=(1.0, 1.0), flip_horizontal=False, flip_vertical=False)
        s = make_sample(crop, host, np.random.default_rng(4), cfg)
        # fundo do recorte 0,40 -> tecido do hospedeiro ~0,458
        self.assertAlmostEqual(s.level_offset, 30000 / 65535 - 0.40, delta=0.01)

    def test_recorte_maior_que_hospedeiro_levanta(self):
        host = host_sintetico(h=900, w=900)
        with self.assertRaises(ValueError):
            make_sample(crop_sintetico(h=950, w=950), host, np.random.default_rng(0),
                        PasteConfig(scale_jitter=(1.0, 1.0)))

    def test_aumento_nao_muda_intensidades(self):
        crop = crop_sintetico()
        out = augment_crop(crop.image, np.random.default_rng(5), PasteConfig(scale_jitter=(1.0, 1.0)))
        self.assertEqual(sorted(np.unique(out).tolist()), sorted(np.unique(crop.image).tolist()))

    def test_nivel_relaxado_respeita_peitoral_pele_e_mama(self):
        """Lesao grande demais para o nivel estrito: cai no relaxado, nunca sobre o peitoral."""
        from mamelesion.window import anatomical_center
        host = host_sintetico()
        # tecido denso so numa faixa de 80 px: a caixa de ~102 px no canvas nao passa no
        # estrito. O peitoral termina em x=480, entao existem centros (x >= 532) cuja
        # caixa inteira fica fora dele -- e so esses podem ser aceitos.
        host.pmap.score[:] = 0.0
        host.pmap.score[450:530, 520:600] = 1.0
        host.pmap.pectoral[:, :480] = 255
        wh = (300, 300)
        for seed in range(20):
            center, info = anatomical_center(host.pmap, host.framing, host.mm_per_px_canvas, wh,
                                             np.random.default_rng(seed), allow_fallback=False)
            self.assertEqual(info["placement"], "anatomia_relaxada")
            b = info["canvas_box"]
            self.assertFalse((host.pmap.pectoral[b["y1"]:b["y2"], b["x1"]:b["x2"]] > 0).any())
            self.assertTrue((host.pmap.breast[b["y1"]:b["y2"], b["x1"]:b["x2"]] > 0).all())

    def test_sem_posicao_valida_e_sem_fallback_levanta(self):
        host = host_sintetico()
        # tecido denso menor que a lesao (estrito falha) e todo a 2 mm da pele (relaxado falha)
        host.pmap.score[:] = 0.0
        host.pmap.score[500:520, 500:520] = 1.0
        host.pmap.skin_distance_mm[:] = 2.0
        with self.assertRaises(ValueError):
            make_sample(crop_sintetico(), host, np.random.default_rng(0), PasteConfig())

    def test_superelipse_sem_cantos_e_platô_central(self):
        from mamelesion.paste import superellipse_alpha
        a = superellipse_alpha(120, 90, 0.15)
        self.assertEqual(float(a[60, 45]), 1.0)
        for y, x in ((0, 0), (0, 89), (119, 0), (119, 89)):
            self.assertEqual(float(a[y, x]), 0.0)
        self.assertTrue(np.all((a >= 0) & (a <= 1)))

    def test_ajuste_por_plano_remove_gradiente_do_fundo(self):
        from mamelesion.paste import BACKGROUND_RADIUS, level_correction, superellipse_radius
        h, w = 200, 160
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        host = 0.45 + 0.0 * xx
        crop = 0.30 + 0.20 * (xx / (w - 1))            # fundo com gradiente horizontal
        band = superellipse_radius(h, w) > BACKGROUND_RADIUS
        corr = level_correction(host, crop, band, "plano")
        np.testing.assert_allclose((crop + corr)[band], host[band], atol=1e-4)
        mediana = level_correction(host, crop, band, "mediana")
        self.assertGreater(float(np.abs((crop + mediana)[band] - host[band]).max()), 0.05)

    def test_feather_rect(self):
        a = feather_rect(50, 40, 5)
        self.assertEqual(a.shape, (50, 40))
        self.assertEqual(float(a[25, 20]), 1.0)
        self.assertLess(float(a[0, 20]), 0.2)
        self.assertTrue(np.all((a >= 0) & (a <= 1)))


if __name__ == "__main__":
    unittest.main()
