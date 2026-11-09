"""Tabela de incidencias e lista explicita de entrada (lotes grandes em partes).

O nome do arquivo pode mentir sobre a incidencia: no ``birads_1`` a MLO esquerda
foi exportada como ``L_CC_1``. Estes testes garantem que a tabela prevalece, que
a queda para o nome continua igual sem ela, e que uma lista com nome ausente e
erro -- nao imagem pulada em silencio.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mamelesion.pipeline import (PipelineOptions, SyntheticLesionPipeline, ValidationError,
                                 load_view_table, parse_view, resolve_view)

BIRADS_1 = Path(__file__).resolve().parents[2] / "birads_1"
MLO_COM_NOME_DE_CC = BIRADS_1 / "10047_32948_L_CC_1.png"


def _tabela(linhas: str) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "incidencias.csv"
    tmp.write_text("arquivo,view,lateralidade\n" + linhas, encoding="utf-8")
    return tmp


class TabelaDeIncidencias(unittest.TestCase):
    def test_tabela_prevalece_sobre_o_nome(self):
        self.assertEqual(parse_view("10047_32948_L_CC_1").view, "CC")   # o defeito
        tabela = load_view_table(_tabela("10047_32948_L_CC_1.png,MLO,L\n"))
        v = resolve_view("10047_32948_L_CC_1", tabela)
        self.assertEqual((v.view, v.laterality), ("MLO", "L"))
        self.assertTrue(v.source.startswith("tabela:"))

    def test_sem_entrada_cai_para_o_nome(self):
        tabela = load_view_table(_tabela("outro.png,MLO,L\n"))
        for stem in ("10228_53288_R_MLO", "1217_IM-0002_CCE_L", "10180_53352_L_CC"):
            self.assertEqual(resolve_view(stem, tabela), parse_view(stem))
        self.assertEqual(resolve_view("10180_53352_L_CC", None), parse_view("10180_53352_L_CC"))

    def test_incidencia_desconhecida_e_erro(self):
        with self.assertRaises(ValueError):
            load_view_table(_tabela("a.png,OBLIQUA_X,L\n"))


class ListaDeEntrada(unittest.TestCase):
    def _pipeline(self, pasta: Path, lista: Path):
        opts = PipelineOptions(input_dir=pasta, output_dir=pasta / "saida", file_list_path=lista)
        return SimpleNamespace(options=opts)          # list_inputs so le self.options

    def test_processa_so_a_lista_na_ordem_dada(self):
        pasta = Path(tempfile.mkdtemp())
        for n in ("a.png", "b.png", "c.png", "LEIA-ME.md"):
            (pasta / n).write_bytes(b"x")
        lista = pasta / "lista.txt"
        lista.write_text("# shard 0\nc.png\na.png\n\n", encoding="utf-8")
        files = SyntheticLesionPipeline.list_inputs(self._pipeline(pasta, lista))
        self.assertEqual([p.name for p in files], ["c.png", "a.png"])

    def test_nome_ausente_e_erro_de_configuracao(self):
        pasta = Path(tempfile.mkdtemp())
        (pasta / "a.png").write_bytes(b"x")
        lista = pasta / "lista.txt"
        lista.write_text("a.png\nsumiu.png\n", encoding="utf-8")
        with self.assertRaises(ValidationError):
            SyntheticLesionPipeline.list_inputs(self._pipeline(pasta, lista))


class ListaSemCaminho(unittest.TestCase):
    def test_nome_com_caminho_e_recusado(self):
        pasta = Path(tempfile.mkdtemp())
        (pasta / "a.png").write_bytes(b"x")
        for ruim in ("../a.png", "/etc/passwd"):
            lista = pasta / "lista.txt"
            lista.write_text(ruim + "\n", encoding="utf-8")
            opts = PipelineOptions(input_dir=pasta, output_dir=pasta / "s", file_list_path=lista)
            with self.assertRaises(ValidationError):
                SyntheticLesionPipeline.list_inputs(SimpleNamespace(options=opts))


@unittest.skipUnless(MLO_COM_NOME_DE_CC.is_file(), "birads_1 ausente nesta maquina")
class PeitoralNaMloRotuladaComoCC(unittest.TestCase):
    """O motivo da tabela, medido numa imagem real: sem ela o peitoral nao e excluido."""

    def test_peitoral_so_aparece_com_a_incidencia_certa(self):
        from mamelesion.anatomy import AnatomyConfig, plausibility_map
        from mamelesion.native import build_breast_canvas, read_source

        src = read_source(MLO_COM_NOME_DE_CC)
        canvas, framing, mama = build_breast_canvas(src.gray_u8, 1024)
        mm = 295.0 / max(src.raw.shape) / framing.scale
        pelo_nome = plausibility_map(canvas, mama, mm, parse_view(MLO_COM_NOME_DE_CC.stem).view,
                                     AnatomyConfig())
        pela_tabela = plausibility_map(canvas, mama, mm, "MLO", AnatomyConfig())
        self.assertFalse(pelo_nome.pectoral.any())
        frac = float((pela_tabela.pectoral > 0).sum()) / float((mama > 0).sum())
        self.assertGreater(frac, 0.03)
        # e a regiao do peitoral deixa de ser elegivel para a lesao
        self.assertEqual(float(np.max(pela_tabela.score[pela_tabela.pectoral > 0])), 0.0)


    def test_caixa_com_gordura_nunca_toca_peitoral_nem_pele(self):
        """min_box_score=0 tira a exigencia de tecido denso, nao as exclusoes duras."""
        import cv2 as cv
        from mamelesion.anatomy import AnatomyConfig, mm_to_px, plausibility_map, sample_anatomical_box
        from mamelesion.native import build_breast_canvas, read_source

        src = read_source(MLO_COM_NOME_DE_CC)
        canvas, framing, mama = build_breast_canvas(src.gray_u8, 1024)
        mm = 295.0 / max(src.raw.shape) / framing.scale
        cfg = AnatomyConfig(min_box_score=0.0, density_upper_percentile=60, min_glandular_fraction=0.2)
        pm = plausibility_map(canvas, mama, mm, "MLO", cfg)
        m = mm_to_px(cfg.pectoral_margin_mm, mm)
        peitoral = cv.dilate(pm.pectoral, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * m + 1,) * 2)) > 0
        rng = np.random.default_rng(0)
        caixas = [sample_anatomical_box(pm, mm, (15, 40), cfg=cfg, rng=rng) for _ in range(60)]
        caixas = [c[0] for c in caixas if c is not None]
        self.assertGreater(len(caixas), 30)
        for box in caixas:
            self.assertFalse(peitoral[box.slices()].any())
            self.assertGreaterEqual(float(pm.skin_distance_mm[box.slices()].min()), cfg.min_skin_distance_mm)


if __name__ == "__main__":
    unittest.main()
