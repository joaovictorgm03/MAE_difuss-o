"""Pipeline de producao: pasta de mamografias saudaveis -> dataset com lesoes.

Diferencas em relacao a ``scripts/batch_generate.py`` (que e um utilitario de
experimentacao):

* **Validacao de entrada** antes de qualquer inferencia: formato, integridade,
  dimensoes, profundidade de bits e presenca de tecido mamario.
* **Posicionamento anatomico** via :mod:`mamelesion.anatomy` em vez de sorteio
  uniforme dentro da mama.
* **Saida em duas resolucoes**: a original de cada arquivo (com a profundidade
  de bits preservada) e o canvas de 1024.
* **Recuperacao parcial**: falha em uma imagem e registrada e o lote continua.
  O processo so termina em erro se *nenhuma* imagem for produzida.
* **Log estruturado**: ``processing.log`` legivel e ``processing.jsonl`` com um
  registro por evento, para auditoria posterior.

Fluxo por imagem::

    validar -> ler -> enquadrar pela mama -> mapa de plausibilidade
      -> [para cada lesao] amostrar caixa -> inpainting -> QC -> (repetir)
      -> compor no canvas -> devolver a resolucao original -> gravar
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import platform
import re
import time
import traceback
import zlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2 as cv
import numpy as np

from . import quality as qmod
from .anatomy import (AnatomyConfig, PlausibilityMap, describe_region,
                      plausibility_map, sample_anatomical_box)
from .compose import compose
from .config import REPO_ROOT, Config
from .generator import Case, comparison_figure
from .inpainting import GenerationResult, LesionInpainter
from .masks import LesionBox, mask_from_box, snap_to_latent
from .native import (SourceImage, build_breast_canvas, canvas_mask_from_native, lesion_delta,
                     native_texture_metrics, read_source, save_image, writeback_native)
from .nativegen import NativeLesion, evaluate_window, generate_native, window_overlay
from .preprocessing import estimate_mm_per_px, to_uint8
from .window import anatomical_center

LOGGER = logging.getLogger("mamelesion.pipeline")

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".dicom"}

#: Limites de sanidade da entrada. Uma mamografia de rastreio tem tipicamente
#: entre 1000 e 5000 px de lado; fora disso e quase certo que o arquivo nao e
#: uma mamografia de campo cheio, e o pipeline deve avisar em vez de gerar lixo.
MIN_SIDE_PX = 256
MAX_SIDE_PX = 12000
MIN_BREAST_AREA_FRAC = 0.03


# ---------------------------------------------------------------------------
# Logging estruturado
# ---------------------------------------------------------------------------


class JsonlHandler(logging.Handler):
    """Grava um objeto JSON por linha, com os campos extras do registro.

    Handler proprio em vez de um formatter porque o interesse aqui e o
    ``record.__dict__`` inteiro (campos passados via ``extra=``), nao apenas a
    mensagem formatada.
    """

    _RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "message", "asctime", "taskName"}

    def __init__(self, path: Path) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname,
                "event": record.getMessage(),
            }
            for key, value in record.__dict__.items():
                if key not in self._RESERVED:
                    payload[key] = _jsonable(value)
            if record.exc_info:
                payload["traceback"] = "".join(traceback.format_exception(*record.exc_info))
            self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:                                  # logging nunca derruba o lote
            self.handleError(record)

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


def _jsonable(value: Any) -> Any:
    """Converte tipos numpy/Path para algo serializavel."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def setup_logging(out_dir: Path, verbose: bool = False) -> None:
    """Configura os tres destinos: console, ``processing.log`` e ``processing.jsonl``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mamelesion")
    logger.setLevel(logging.DEBUG)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    text = logging.FileHandler(out_dir / "processing.log", encoding="utf-8")
    text.setLevel(logging.DEBUG)
    text.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(text)

    logger.addHandler(JsonlHandler(out_dir / "processing.jsonl"))
    logger.propagate = False


# ---------------------------------------------------------------------------
# Metadados da incidencia
# ---------------------------------------------------------------------------

_VIEW_TOKENS = {
    "CC": "CC", "CCD": "CC", "CCE": "CC",
    "MLO": "MLO", "MLOD": "MLO", "MLOE": "MLO", "OE": "MLO", "OD": "MLO",
    "ML": "LATERAL", "LM": "LATERAL", "RL": "LATERAL", "LMO": "LATERAL",
}
_SIDE_FROM_TOKEN = {"CCD": "R", "CCE": "L", "MLOD": "R", "MLOE": "L", "OD": "R", "OE": "L"}


@dataclass
class ViewInfo:
    """Incidencia e lateralidade deduzidas do nome do arquivo."""

    view: str = "UNKNOWN"
    laterality: str = "unknown"
    source: str = "nome_do_arquivo"

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def parse_view(stem: str) -> ViewInfo:
    """Deduz incidencia e lateralidade do nome do arquivo.

    O conjunto mistura duas convencoes -- ``1217_IM-0002_CCE_L`` (sigla em
    portugues) e ``10228_53288_R_MLO`` (sigla em ingles). Tokens desconhecidos
    devolvem ``UNKNOWN``, e nesse caso :func:`plausibility_map` ainda procura o
    peitoral, mas sem o fallback geometrico: e melhor nao excluir regiao alguma
    do que excluir a errada com base num palpite sobre a incidencia.
    """
    tokens = [t.upper() for t in re.split(r"[_\-.\s]+", stem) if t]
    view, laterality = "UNKNOWN", "unknown"
    for token in tokens:
        if token in _VIEW_TOKENS and view == "UNKNOWN":
            view = _VIEW_TOKENS[token]
        if token in _SIDE_FROM_TOKEN and laterality == "unknown":
            laterality = _SIDE_FROM_TOKEN[token]
    if laterality == "unknown":
        for token in tokens:
            if token in {"L", "R"}:
                laterality = token
                break
    return ViewInfo(view, laterality)


def load_view_table(path: Path) -> Dict[str, ViewInfo]:
    """Le uma tabela de incidencias que prevalece sobre o nome do arquivo.

    CSV com as colunas ``arquivo``, ``view`` e ``lateralidade`` (``arquivo`` com
    ou sem extensao). Existe porque o nome pode mentir: no ``birads_1``, a MLO
    esquerda foi exportada como ``L_CC_1`` -- o rotulo gravado na propria imagem
    diz ``MLOE`` e o peitoral esta visivel. Pelo nome, :func:`parse_view` devolve
    CC e :func:`plausibility_map` nem procura o peitoral.
    """
    table: Dict[str, ViewInfo] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            view = (row.get("view") or "").strip().upper()
            if view not in set(_VIEW_TOKENS.values()) | {"UNKNOWN"}:
                raise ValueError("{}: incidencia desconhecida '{}' para {}".format(
                    path, row.get("view"), row.get("arquivo")))
            side = (row.get("lateralidade") or "unknown").strip() or "unknown"
            table[Path(row["arquivo"].strip()).stem] = ViewInfo(view, side, "tabela:" + Path(path).name)
    return table


def resolve_view(stem: str, table: Optional[Dict[str, ViewInfo]] = None) -> ViewInfo:
    """Incidencia da tabela, quando houver uma entrada; senao, a do nome do arquivo."""
    if table and stem in table:
        return table[stem]
    return parse_view(stem)


# ---------------------------------------------------------------------------
# Validacao de entrada
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Entrada rejeitada antes de qualquer inferencia."""


def validate_path(path: Path) -> None:
    """Checagens que nao exigem decodificar a imagem."""
    if not path.is_file():
        raise ValidationError("nao e um arquivo regular")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValidationError("extensao nao suportada: {} (aceitos: {})".format(
            path.suffix or "<sem extensao>", ", ".join(sorted(SUPPORTED_SUFFIXES))))
    if path.stat().st_size < 1024:
        raise ValidationError("arquivo vazio ou truncado ({} bytes)".format(path.stat().st_size))


def validate_source(source: SourceImage) -> List[str]:
    """Checagens sobre o conteudo. Devolve avisos; levanta em caso fatal."""
    warnings: List[str] = []
    h, w = source.shape
    if min(h, w) < MIN_SIDE_PX:
        raise ValidationError("dimensao minima {} px abaixo do limite de {} px".format(
            min(h, w), MIN_SIDE_PX))
    if max(h, w) > MAX_SIDE_PX:
        raise ValidationError("dimensao maxima {} px acima do limite de {} px".format(
            max(h, w), MAX_SIDE_PX))
    if source.bit_depth == 16 and int(source.raw.max()) < 4096:
        warnings.append("PNG de 16 bits com valor maximo {} (faixa dinamica baixa)".format(
            int(source.raw.max())))
    dark = float((source.gray_u8 < 10).mean())
    if dark > 0.95:
        raise ValidationError("imagem {:.0%} preta: nao parece uma mamografia".format(dark))
    if dark < 0.05:
        warnings.append("quase nenhum fundo escuro ({:.0%}): imagem possivelmente ja recortada"
                        .format(dark))
    return warnings


# ---------------------------------------------------------------------------
# Configuracao e resultados
# ---------------------------------------------------------------------------


@dataclass
class PipelineOptions:
    """Parametros de execucao do lote (o resto vem de ``config.yaml``)."""

    input_dir: Path
    output_dir: Path
    lesions_per_image: int = 1
    lesion_types: List[str] = field(default_factory=lambda: ["mass", "nodule", "density"])
    #: pesos por tipo (mesma ordem de ``lesion_types``); ``None`` alterna os tipos igualmente
    lesion_type_weights: Optional[List[float]] = None
    seed: int = 1337
    canvas_size: int = 1024
    save_canvas: bool = True
    save_overlay: bool = True
    save_anatomy_map: bool = True
    max_placement_attempts: int = 4
    limit: Optional[int] = None
    dry_run: bool = False
    #: tabela de incidencias que prevalece sobre o nome do arquivo (:func:`load_view_table`)
    view_table_path: Optional[Path] = None
    #: arquivo texto com um nome de imagem por linha: processa so esses, de ``input_dir``
    file_list_path: Optional[Path] = None
    #: sufixo acrescentado ao nome de cada saida (``--rotulo-saida``): a mesma
    #: mamografia pode receber lesoes novas em lotes diferentes, e os nomes nao
    #: podem colidir quando os lotes forem juntados
    output_tag: str = ""
    #: sobrescreve campos de :class:`AnatomyConfig` (``--anatomia CHAVE=VALOR``)
    anatomy_overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        for key in ("input_dir", "output_dir", "view_table_path", "file_list_path"):
            out[key] = str(out[key]) if out[key] is not None else None
        return out


@dataclass
class LesionRecord:
    """Uma lesao gerada com sucesso."""

    case: str
    index: int
    lesion_type: str
    seed: int
    attempts: int
    region: str
    box_canvas: Dict[str, int]
    box_original: Dict[str, Any]
    diameter_mm: float
    anatomy: Dict[str, float]
    metrics: Dict[str, float]
    qc_passed: bool
    qc_failed_criteria: List[str]
    plausibility_rate: float
    confidence: float
    files: Dict[str, str]
    duration_s: float
    #: como a lesao foi gerada: modo, prompt, adaptador, janela (vazio no modo canvas)
    generation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseReport:
    """Resultado do processamento de uma imagem de entrada."""

    name: str
    path: str
    status: str                       # ok | parcial | falha | ignorada
    reason: str = ""
    warnings: List[str] = field(default_factory=list)
    source: Dict[str, Any] = field(default_factory=dict)
    anatomy: Dict[str, Any] = field(default_factory=dict)
    lesions: List[LesionRecord] = field(default_factory=list)
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def confidence_score(plausibility_rate: float, anat_score_mean: float,
                     qc_passed: bool) -> float:
    """Confianca da sintese em [0, 1].

    Combina o que foi de fato medido: a fracao de criterios de imagem aprovados
    (:mod:`mamelesion.quality`), o score anatomico medio da regiao e o veredito
    do portao de QC. Os pesos sao uma escolha de projeto, nao uma calibracao
    estatistica -- o numero serve para *ordenar* geracoes, nao como probabilidade.
    """
    base = 0.55 * float(plausibility_rate) + 0.45 * float(anat_score_mean)
    return round(base if qc_passed else base * 0.6, 4)


class SyntheticLesionPipeline:
    """Executa o lote inteiro, isolando falhas por imagem."""

    def __init__(self, cfg: Config, options: PipelineOptions) -> None:
        self.cfg = cfg
        self.options = options
        self.anatomy_cfg = AnatomyConfig(**options.anatomy_overrides)
        self.reports: List[CaseReport] = []
        self._rng = np.random.default_rng(options.seed)
        self.view_table: Dict[str, ViewInfo] = {}
        if options.view_table_path:
            try:
                self.view_table = load_view_table(options.view_table_path)
            except (OSError, KeyError, ValueError) as exc:
                raise ValidationError("tabela de incidencias invalida: {}".format(exc)) from exc

        nw = cfg.native_window
        self.native = bool(nw.enabled)
        self.run_info = None
        lora_path = None
        if self.native and nw.lora_path:
            from .finetune import load_run
            path = Path(nw.lora_path)
            path = path if path.is_absolute() else (REPO_ROOT / path).resolve()
            try:
                self.run_info = load_run(path)
            except FileNotFoundError as exc:
                raise ValidationError("native_window.lora_path invalido: {}".format(exc)) from exc
            # o LoRA so vale na geometria em que foi ajustado
            for diff in self.run_info.apply_geometry(nw):
                LOGGER.warning("geometria do treino prevalece sobre o config.yaml: %s", diff)
            lora_path = str(self.run_info.lora_dir)
        if self.native and nw.size_source == "catalog":
            if self.run_info is None:
                raise ValidationError(
                    "native_window.size_source=catalog exige native_window.lora_path (o catalogo de "
                    "tamanhos vem do treino); para o modo nativo sem LoRA use size_source=preset")
            missing = [t for t in options.lesion_types if t not in self.run_info.size_catalog]
            if missing:
                raise ValidationError("tipos sem tamanhos no catalogo do treino: {} (disponiveis: {})".format(
                    ", ".join(missing), ", ".join(sorted(self.run_info.size_catalog))))
        if self.native and nw.size_source not in {"catalog", "preset"}:
            raise ValidationError("native_window.size_source desconhecido: " + str(nw.size_source))
        self.inpainter = LesionInpainter(cfg, lora_path=lora_path, lora_scale=nw.lora_scale)

    # -- infraestrutura ----------------------------------------------------

    def _dirs(self) -> Dict[str, Path]:
        out = self.options.output_dir
        dirs = {
            "original": out / "original",
            "canvas": out / "1024",
            "mask_original": out / "masks" / "original",
            "mask_canvas": out / "masks" / "1024",
            "overlay": out / "overlays",
            "anatomy": out / "anatomy",
            "metadata": out / "metadata",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs

    def list_inputs(self) -> List[Path]:
        directory = self.options.input_dir
        if not directory.is_dir():
            raise ValidationError("pasta de entrada inexistente: {}".format(directory))

        # Arquivos que nao sao imagem (LEIA-ME.md, .DS_Store, planilhas) convivem
        # com as mamografias na pasta de entrada. Sao descartados aqui, antes do
        # lote: contados como "ignorada" eles fariam toda execucao terminar em
        # falha parcial (codigo de saida 1). Um arquivo com extensao de imagem
        # que nao abre continua sendo falha, como deve ser.
        files, sem_imagem = [], []
        if self.options.file_list_path:
            # lista explicita (shards de um lote grande): nome ausente e erro de
            # configuracao, nao imagem pulada em silencio
            nomes = [l.strip() for l in Path(self.options.file_list_path).read_text(
                encoding="utf-8").splitlines() if l.strip() and not l.lstrip().startswith("#")]
            com_caminho = [n for n in nomes if Path(n).name != n]
            if com_caminho:                  # "/x.png" ou "../x.png" escapariam de input_dir
                raise ValidationError("a lista so aceita nomes de arquivo, sem caminho: {}".format(
                    ", ".join(com_caminho[:5])))
            candidatos = [directory / n for n in nomes]
            ausentes = [p.name for p in candidatos if not p.is_file()]
            if ausentes:
                raise ValidationError("{} arquivo(s) da lista ausentes em {}: {}".format(
                    len(ausentes), directory, ", ".join(ausentes[:5])))
        else:
            candidatos = sorted(p for p in directory.iterdir() if p.is_file())
        for item in candidatos:
            destino = files if item.suffix.lower() in SUPPORTED_SUFFIXES else sem_imagem
            destino.append(item)

        if sem_imagem:
            LOGGER.info("%d arquivo(s) sem extensao de imagem descartado(s) da entrada: %s",
                        len(sem_imagem), ", ".join(p.name for p in sem_imagem[:5]),
                        extra={"descartados": [p.name for p in sem_imagem]})
        if not files:
            raise ValidationError(
                "nenhuma imagem em {} (aceitos: {})".format(
                    directory, ", ".join(sorted(SUPPORTED_SUFFIXES))))

        if self.options.limit:
            files = files[: self.options.limit]
        return files

    # -- execucao ----------------------------------------------------------

    def run(self) -> List[CaseReport]:
        """Processa todas as imagens. Nunca levanta por falha de uma imagem."""
        started = time.time()
        dirs = self._dirs()
        files = self.list_inputs()

        LOGGER.info("lote iniciado", extra={
            "n_arquivos": len(files),
            "entrada": str(self.options.input_dir),
            "saida": str(self.options.output_dir),
            "opcoes": self.options.to_dict(),
            "ambiente": environment_info(),
        })

        for i, path in enumerate(files, 1):
            LOGGER.info("[%d/%d] %s", i, len(files), path.name, extra={"arquivo": path.name})
            report = self.process_one(path, dirs)
            self.reports.append(report)
            LOGGER.info(
                "[%d/%d] %s -> %s (%d lesao(oes), %.1fs)",
                i, len(files), path.name, report.status, len(report.lesions), report.duration_s,
                extra={"arquivo": path.name, "status": report.status,
                       "n_lesoes": len(report.lesions), "motivo": report.reason,
                       "duracao_s": round(report.duration_s, 2)})

        elapsed = time.time() - started
        ok = sum(1 for r in self.reports if r.status == "ok")
        partial = sum(1 for r in self.reports if r.status == "parcial")
        failed = sum(1 for r in self.reports if r.status in {"falha", "ignorada"})
        LOGGER.info("lote concluido: %d ok, %d parcial, %d falha em %.1fs",
                    ok, partial, failed, elapsed,
                    extra={"ok": ok, "parcial": partial, "falha": failed,
                           "duracao_total_s": round(elapsed, 1)})
        self.inpainter.unload()
        return self.reports

    def process_one(self, path: Path, dirs: Dict[str, Path]) -> CaseReport:
        """Processa uma imagem, capturando qualquer erro como status da imagem."""
        started = time.time()
        report = CaseReport(name=path.stem, path=str(path), status="falha")
        try:
            validate_path(path)
            try:
                source = read_source(path, self.cfg.preprocessing.max_value)
            except ValueError as exc:      # arquivo corrompido/truncado/nao-imagem
                raise ValidationError(str(exc)) from exc
            report.warnings = validate_source(source)
            report.source = source.meta()
            for warning in report.warnings:
                LOGGER.warning("%s: %s", path.name, warning,
                               extra={"arquivo": path.name, "aviso": warning})

            case, pmap, view = self._prepare(source, path)
            report.anatomy = dict(pmap.stats)
            report.anatomy["view"] = view.view
            report.anatomy["laterality"] = view.laterality
            report.anatomy["view_source"] = view.source
            report.anatomy["mm_per_px_canvas"] = case.mm_per_px

            if pmap.valid_fraction() < MIN_BREAST_AREA_FRAC:
                raise ValidationError(
                    "regiao anatomicamente valida cobre apenas {:.1%} da mama; "
                    "sem espaco para lesao plausivel".format(pmap.valid_fraction()))

            if self.options.dry_run:
                report.status = "ok"
                if self.options.save_anatomy_map:
                    self._save_anatomy_figure(case, pmap, [], dirs["anatomy"])
                return report

            for index in range(self.options.lesions_per_image):
                # crc32 e estavel entre processos; hash() do Python nao e
                # (PYTHONHASHSEED), e usa-lo aqui quebraria a reprodutibilidade
                # por semente que os metadados prometem.
                stable = zlib.crc32(path.stem.encode("utf-8"))
                lesion_type = self._choose_type(stable, index)
                one = self._one_lesion_native if self.native else self._one_lesion
                record = one(case, source, pmap, view, lesion_type, index, dirs)
                if record is not None:
                    report.lesions.append(record)

            if self.options.save_anatomy_map:
                self._save_anatomy_figure(case, pmap, report.lesions, dirs["anatomy"])

            if len(report.lesions) == self.options.lesions_per_image:
                report.status = "ok"
            elif report.lesions:
                report.status = "parcial"
                report.reason = "{} de {} lesoes geradas".format(
                    len(report.lesions), self.options.lesions_per_image)
            else:
                report.status = "falha"
                report.reason = report.reason or "nenhuma lesao aprovada no controle de qualidade"

        except ValidationError as exc:
            report.status = "ignorada"
            report.reason = str(exc)
            LOGGER.warning("%s ignorada: %s", path.name, exc,
                           extra={"arquivo": path.name, "motivo": str(exc)})
        except Exception as exc:                            # falha isolada, lote continua
            report.status = "falha"
            report.reason = "{}: {}".format(type(exc).__name__, exc)
            LOGGER.error("%s falhou: %s", path.name, report.reason,
                         exc_info=True, extra={"arquivo": path.name, "motivo": report.reason})
        finally:
            report.duration_s = time.time() - started
            self._write_case_metadata(report, dirs["metadata"])
        return report

    def _choose_type(self, stable: int, index: int) -> str:
        """Tipo da lesao, deterministico pelo nome do arquivo.

        Sem pesos, alterna os tipos (comportamento original, preservado byte a
        byte). Com pesos, sorteia pela fracao acumulada a partir de um hash
        estavel -- ex.: ``birads5:0.83,birads4:0.17`` reproduz a proporcao dos
        recortes reais em vez de 50/50.

        O hash dos pesos e SHA-256, nao crc32: nos 66 nomes de ``sem_anomalia``
        (curtos e parecidos) o crc32 nao e uniforme (KS p=0,003) e produziu 33% de
        BI-RADS 4 para um peso de 17%.
        """
        types = self.options.lesion_types
        weights = self.options.lesion_type_weights
        if not weights:
            return types[(index + stable) % len(types)]
        digest = hashlib.sha256("{}:{}".format(stable, index).encode("utf-8")).hexdigest()
        u = int(digest[:12], 16) / float(16 ** 12)
        acc = np.cumsum(np.asarray(weights, float) / float(sum(weights)))
        return types[int(min(np.searchsorted(acc, u, side="right"), len(types) - 1))]

    # -- etapas ------------------------------------------------------------

    def _prepare(self, source: SourceImage, path: Path) -> Tuple[Case, PlausibilityMap, ViewInfo]:
        """Enquadra pela mama e constroi o mapa de plausibilidade."""
        size = self.options.canvas_size
        try:
            canvas, framing, breast = build_breast_canvas(
                source.gray_u8, size, breast_threshold=self.cfg.preprocessing.breast_threshold)
        except ValueError as exc:          # sem tecido detectavel: entrada invalida
            raise ValidationError(str(exc)) from exc

        area_frac = float((breast > 0).sum()) / float(size * size)
        if area_frac < MIN_BREAST_AREA_FRAC:
            raise ValidationError(
                "tecido mamario cobre {:.1%} do canvas (minimo {:.0%})".format(
                    area_frac, MIN_BREAST_AREA_FRAC))

        mm_per_px = self.cfg.preprocessing.mm_per_px or estimate_mm_per_px(
            framing, self.cfg.preprocessing.detector_long_side_mm, source.pixel_spacing_mm)
        view = resolve_view(path.stem, self.view_table)

        pmap = plausibility_map(canvas, breast, float(mm_per_px), view.view, self.anatomy_cfg)
        case = Case(path.stem, path, canvas, framing, breast, float(mm_per_px),
                    pmap.side, source.pixel_spacing_mm)

        LOGGER.debug("%s preparada", path.name, extra={
            "arquivo": path.name, "view": view.view, "lateralidade": view.laterality,
            "mm_por_px_canvas": round(float(mm_per_px), 4),
            "area_valida_frac": round(pmap.valid_fraction(), 4),
            "peitoral": pmap.pectoral_method})
        return case, pmap, view

    def _one_lesion(self, case: Case, source: SourceImage, pmap: PlausibilityMap,
                    view: ViewInfo, lesion_type: str, index: int,
                    dirs: Dict[str, Path]) -> Optional[LesionRecord]:
        """Gera, valida e grava uma lesao. Devolve ``None`` se nao passar no QC."""
        started = time.time()
        preset = self.cfg.lesion.preset(lesion_type)
        qc_cfg = self.cfg.quality
        attempts = 0
        best: Optional[Tuple[GenerationResult, Dict[str, Any], LesionBox, Dict[str, float], int]] = None

        for placement in range(self.options.max_placement_attempts):
            sampled = self._sample_box(case, pmap, lesion_type)
            if sampled is None:
                LOGGER.warning("%s: nenhuma caixa plausivel para %s", case.name, lesion_type,
                               extra={"arquivo": case.name, "tipo": lesion_type})
                return None
            box, anat = sampled
            if self.cfg.lesion.snap_to_latent:
                box = snap_to_latent(box, case.size, self.cfg.preprocessing.model_resolution)

            max_attempts = max(1, qc_cfg.qc_max_attempts if qc_cfg.qc_enabled else 1)
            for attempt in range(max_attempts):
                attempts += 1
                seed = int(self._rng.integers(0, 2 ** 31 - 1))
                mask = mask_from_box(box, case.size, self.cfg.lesion.mask_shape)
                try:
                    result = self.inpainter.generate(case.canvas, mask, box,
                                                     lesion_type=lesion_type, seed=seed)
                except Exception as exc:
                    LOGGER.error("%s: inferencia falhou (%s)", case.name, exc,
                                 exc_info=True,
                                 extra={"arquivo": case.name, "tipo": lesion_type, "seed": seed})
                    raise RuntimeError("falha na inferencia de difusao: {}".format(exc)) from exc

                result.composed = compose(case.canvas, result.generated_model_res, mask,
                                          self._composition_for(lesion_type), box)
                evaluation = qmod.evaluate(result, case, qc_cfg, lesion_type)
                passed, failed = qmod.passes_qc(evaluation, qc_cfg.qc_required)

                LOGGER.debug("%s lesao %d tentativa %d", case.name, index, attempts, extra={
                    "arquivo": case.name, "tipo": lesion_type, "seed": seed,
                    "qc_ok": passed, "qc_falhas": failed,
                    "weber": round(float(evaluation["metrics"].get("weber_contrast", float("nan"))), 4),
                    "cnr": round(float(evaluation["metrics"].get("cnr", float("nan"))), 4)})

                if best is None or (passed and not best[1]["_passed"]):
                    best = (result, {**evaluation, "_passed": passed, "_failed": failed,
                                     "_seed": seed}, box, anat, attempts)
                if passed:
                    return self._save_lesion(case, source, pmap, view, result, evaluation,
                                             box, anat, lesion_type, index, seed, attempts,
                                             passed, failed, dirs, time.time() - started)

        if best is not None and not qc_cfg.qc_enabled:
            result, evaluation, box, anat, attempts = best
            return self._save_lesion(case, source, pmap, view, result, evaluation, box, anat,
                                     lesion_type, index, evaluation.get("_seed", -1), attempts,
                                     False, evaluation.get("_failed", []), dirs,
                                     time.time() - started)

        LOGGER.warning("%s: lesao %s descartada apos %d tentativas (QC)",
                       case.name, lesion_type, attempts,
                       extra={"arquivo": case.name, "tipo": lesion_type, "tentativas": attempts,
                              "qc_falhas": best[1]["_failed"] if best else []})
        return None

    def _sample_box(self, case: Case, pmap: PlausibilityMap,
                    lesion_type: str) -> Optional[Tuple[LesionBox, Dict[str, float]]]:
        """Sorteia uma caixa, reduzindo o diametro se o tamanho do preset nao couber.

        Mamas pequenas ou com parenquima muito fragmentado nao acomodam uma
        regiao contigua de 40 mm inteiramente plausivel, e o preset ``density``
        pede exatamente isso. Antes de desistir da lesao, tenta-se a metade
        inferior da faixa e depois uma faixa reduzida: uma assimetria focal de
        14 mm continua sendo um achado valido, enquanto nenhuma lesao nao e.
        """
        preset = self.cfg.lesion.preset(lesion_type)
        lo, hi = preset.diameter_mm_range
        candidates = [(lo, hi), (lo, lo + 0.4 * (hi - lo)), (max(5.0, lo * 0.6), lo)]

        for attempt, diameter_range in enumerate(candidates):
            sampled = sample_anatomical_box(
                pmap, case.mm_per_px, diameter_range, preset.aspect_range,
                self.anatomy_cfg, self._rng, canvas=case.size)
            if sampled is None:
                continue
            if attempt:
                LOGGER.warning(
                    "%s: %s reduzido para %.0f-%.0f mm (faixa do preset nao coube)",
                    case.name, lesion_type, diameter_range[0], diameter_range[1],
                    extra={"arquivo": case.name, "tipo": lesion_type,
                           "faixa_mm": list(diameter_range),
                           "faixa_preset_mm": list(preset.diameter_mm_range)})
            box, anat = sampled
            anat["diameter_range_mm_lo"] = float(diameter_range[0])
            anat["diameter_range_mm_hi"] = float(diameter_range[1])
            return box, anat
        return None

    def _composition_for(self, lesion_type: str):
        """Copia da configuracao de composicao com os ajustes do preset."""
        from copy import deepcopy
        comp = deepcopy(self.cfg.composition)
        preset = self.cfg.lesion.preset(lesion_type)
        comp.detail_transfer = preset.detail_transfer
        comp.lesion_gain = preset.lesion_gain
        return comp

    def _save_lesion(self, case: Case, source: SourceImage, pmap: PlausibilityMap,
                     view: ViewInfo, result: GenerationResult, evaluation: Dict[str, Any],
                     box: LesionBox, anat: Dict[str, float], lesion_type: str, index: int,
                     seed: int, attempts: int, passed: bool, failed: List[str],
                     dirs: Dict[str, Path], duration: float) -> LesionRecord:
        """Grava as duas resolucoes, mascaras, overlay e metadados da lesao."""
        suffix = ("_com_lesao" if self.options.lesions_per_image == 1 else
                  "_com_lesao_{:02d}".format(index + 1)) + self.options.output_tag
        stem = case.name + suffix

        delta = lesion_delta(result, self._composition_for(lesion_type))
        native, geom, mask_native = writeback_native(
            source, case.framing, delta, box, result.mask)

        files: Dict[str, str] = {}
        original_path = dirs["original"] / (stem + ".png")
        save_image(original_path, native)
        files["original"] = str(original_path)

        if self.options.save_canvas:
            canvas_path = dirs["canvas"] / (stem + "_1024.png")
            save_image(canvas_path, result.composed)
            files["canvas_1024"] = str(canvas_path)

        mask_path = dirs["mask_original"] / (stem + "_mask.png")
        save_image(mask_path, mask_native)
        files["mask_original"] = str(mask_path)

        if self.options.save_canvas:
            mask_canvas_path = dirs["mask_canvas"] / (stem + "_mask_1024.png")
            save_image(mask_canvas_path, result.mask)
            files["mask_1024"] = str(mask_canvas_path)

        if self.options.save_overlay:
            overlay_path = dirs["overlay"] / (stem + "_cmp.png")
            save_image(overlay_path, comparison_figure(result))
            files["overlay"] = str(overlay_path)

        metrics = {k: _finite(v) for k, v in evaluation["metrics"].items()}
        # medir tambem o arquivo entregue, nao so o canvas intermediario
        metrics.update({k: _finite(v) for k, v in native_texture_metrics(
            native, source.raw, geom, mask_native,
            background_level=self.cfg.preprocessing.breast_threshold * source.scale_to_u8
            ).items()})
        rate = float(evaluation["plausibility"]["pass_rate"])
        record = LesionRecord(
            case=case.name, index=index, lesion_type=lesion_type, seed=seed, attempts=attempts,
            region=describe_region(box, pmap, view.view),
            box_canvas=box.to_dict(), box_original=geom,
            diameter_mm=round(float(np.sqrt(box.w * box.h) * case.mm_per_px), 2),
            anatomy={k: round(float(v), 4) for k, v in anat.items()},
            metrics=metrics, qc_passed=passed, qc_failed_criteria=list(failed),
            plausibility_rate=round(rate, 4),
            confidence=confidence_score(rate, anat.get("anat_score_mean", 0.0), passed),
            files=files, duration_s=round(duration, 2))

        LOGGER.info("%s: %s em %s (conf %.2f, %d tentativa(s))",
                    case.name, lesion_type, record.region, record.confidence, attempts,
                    extra={"arquivo": case.name, "tipo": lesion_type, "regiao": record.region,
                           "confianca": record.confidence, "diametro_mm": record.diameter_mm,
                           "caixa_original": geom, "seed": seed})
        return record

    # -- modo nativo -------------------------------------------------------

    def _native_size(self, lesion_type: str, case: Case) -> Tuple[int, int]:
        """Tamanho nativo (largura, altura) da lesao: de um recorte real ou do preset em mm."""
        if self.cfg.native_window.size_source == "catalog":
            sizes = self.run_info.size_catalog[lesion_type]
            w, h = sizes[int(self._rng.integers(len(sizes)))]
            return int(w), int(h)
        preset = self.cfg.lesion.preset(lesion_type)
        native_mm = case.mm_per_px * case.framing.scale
        d_px = float(self._rng.uniform(*preset.diameter_mm_range)) / native_mm
        aspect = float(self._rng.uniform(*preset.aspect_range))
        return max(16, int(round(d_px * np.sqrt(aspect)))), max(16, int(round(d_px / np.sqrt(aspect))))

    def _native_guidance(self, preset) -> float:
        """``native_window.guidance`` > ``melhor_guidance`` do treino > guidance do preset."""
        if self.cfg.native_window.guidance is not None:
            return float(self.cfg.native_window.guidance)
        if self.run_info is not None and self.run_info.config.get("melhor_guidance") is not None:
            return float(self.run_info.config["melhor_guidance"])
        return float(preset.guidance_scale)

    def _one_lesion_native(self, case: Case, source: SourceImage, pmap: PlausibilityMap,
                           view: ViewInfo, lesion_type: str, index: int,
                           dirs: Dict[str, Path]) -> Optional[LesionRecord]:
        """Modo nativo: tamanho real, posicao anatomica, janela em resolucao original, QC.

        Mesma politica de tentativas do modo canvas: ate ``qc_max_attempts``
        sementes por posicao e ate ``max_placement_attempts`` posicoes (cada uma
        com um novo tamanho sorteado). Sem nivel de ultimo recurso no
        posicionamento: uma lesao que so caberia sobre o peitoral ou a pele e
        trocada por outro tamanho, nao gravada.
        """
        started = time.time()
        nw, qc_cfg = self.cfg.native_window, self.cfg.quality
        preset = self.cfg.lesion.preset(lesion_type)
        prompt = preset.prompt or self.cfg.lesion.prompt
        # o LoRA foi ajustado so na fonte de 16 bits; em outra fonte a textura do
        # detector difere e o resultado nao foi avaliado -- avisar, nao bloquear
        other_source = self.run_info is not None and source.bit_depth != 16
        if other_source and index == 0:
            LOGGER.warning("%s: imagem de %d bits, fonte diferente da do treino do LoRA; resultado nao avaliado",
                           case.name, source.bit_depth,
                           extra={"arquivo": case.name, "bit_depth": source.bit_depth})
        guidance = self._native_guidance(preset)
        ref = source.scale_to_u8 * 255.0
        native_mm = case.mm_per_px * case.framing.scale
        attempts, placed = 0, 0
        best = None

        for _ in range(self.options.max_placement_attempts):
            wh = self._native_size(lesion_type, case)
            if max(wh) > min(source.raw.shape):
                continue
            try:
                center, anat = anatomical_center(pmap, case.framing, case.mm_per_px, wh, self._rng,
                                                 self.anatomy_cfg, allow_fallback=False)
            except ValueError:
                LOGGER.debug("%s: %dx%d px sem posicao plausivel", case.name, wh[0], wh[1],
                             extra={"arquivo": case.name, "tipo": lesion_type, "tamanho_px": list(wh)})
                continue
            placed += 1
            max_attempts = max(1, qc_cfg.qc_max_attempts if qc_cfg.qc_enabled else 1)
            for _ in range(max_attempts):
                attempts += 1
                seed = int(self._rng.integers(0, 2 ** 31 - 1))
                try:
                    lesion = generate_native(
                        self.inpainter.generate_window, source.raw, ref, center, wh, nw, prompt, seed,
                        preset.num_inference_steps, guidance, preset.lesion_gain,
                        self.cfg.lesion.negative_prompt or None, self.cfg.preprocessing.model_resolution)
                except Exception as exc:
                    LOGGER.error("%s: inferencia falhou (%s)", case.name, exc, exc_info=True,
                                 extra={"arquivo": case.name, "tipo": lesion_type, "seed": seed})
                    raise RuntimeError("falha na inferencia de difusao: {}".format(exc)) from exc

                evaluation = evaluate_window(lesion, native_mm, qc_cfg, lesion_type,
                                             self.cfg.preprocessing.breast_threshold, case.name)
                passed, failed = bool(evaluation["qc_ok"]), list(evaluation["qc_falhas"])
                LOGGER.debug("%s lesao %d tentativa %d", case.name, index, attempts, extra={
                    "arquivo": case.name, "tipo": lesion_type, "seed": seed, "qc_ok": passed,
                    "qc_falhas": failed, "tamanho_px": list(wh), "janela": lesion.window.to_dict(),
                    "weber": _finite(evaluation["metrics"].get("weber_contrast")),
                    "cnr": _finite(evaluation["metrics"].get("cnr")),
                    "roi_delta_ratio": _finite(evaluation["metrics"].get("roi_delta_ratio"))})
                if best is None or (passed and not best[2]):
                    best = (lesion, evaluation, passed, failed, anat, seed)
                if passed:
                    return self._save_native_lesion(case, source, pmap, view, lesion, evaluation, anat,
                                                    lesion_type, index, seed, attempts, True, [], dirs,
                                                    time.time() - started)

        if best is not None and not qc_cfg.qc_enabled:
            lesion, evaluation, passed, failed, anat, seed = best
            return self._save_native_lesion(case, source, pmap, view, lesion, evaluation, anat, lesion_type,
                                            index, seed, attempts, passed, failed, dirs, time.time() - started)

        LOGGER.warning("%s: lesao %s descartada (%d posicao(oes) plausivel(is), %d tentativa(s))",
                       case.name, lesion_type, placed, attempts,
                       extra={"arquivo": case.name, "tipo": lesion_type, "tentativas": attempts,
                              "posicoes": placed, "qc_falhas": best[3] if best else []})
        return None

    def _save_native_lesion(self, case: Case, source: SourceImage, pmap: PlausibilityMap,
                            view: ViewInfo, lesion: NativeLesion, evaluation: Dict[str, Any],
                            anat: Dict[str, Any], lesion_type: str, index: int, seed: int,
                            attempts: int, passed: bool, failed: List[str],
                            dirs: Dict[str, Path], duration: float) -> LesionRecord:
        """Grava as mesmas saidas do modo canvas a partir da imagem nativa gerada."""
        suffix = ("_com_lesao" if self.options.lesions_per_image == 1 else
                  "_com_lesao_{:02d}".format(index + 1)) + self.options.output_tag
        stem = case.name + suffix
        out, mask_native = lesion.raw_out, lesion.mask_native
        ref = source.scale_to_u8 * 255.0

        files: Dict[str, str] = {}
        original_path = dirs["original"] / (stem + ".png")
        save_image(original_path, out)
        files["original"] = str(original_path)
        mask_path = dirs["mask_original"] / (stem + "_mask.png")
        save_image(mask_path, mask_native)
        files["mask_original"] = str(mask_path)

        if self.options.save_canvas:
            # mesmo enquadramento da entrada: a lesao nao altera a caixa envolvente da mama
            canvas_out, framing_out, _ = build_breast_canvas(
                to_uint8(out, ref), case.size, breast_threshold=self.cfg.preprocessing.breast_threshold)
            if framing_out != case.framing:
                LOGGER.warning("%s: enquadramento do canvas de saida divergiu do de entrada", case.name)
            canvas_path = dirs["canvas"] / (stem + "_1024.png")
            save_image(canvas_path, canvas_out)
            files["canvas_1024"] = str(canvas_path)
            mask_canvas_path = dirs["mask_canvas"] / (stem + "_mask_1024.png")
            save_image(mask_canvas_path, canvas_mask_from_native(mask_native, case.framing))
            files["mask_1024"] = str(mask_canvas_path)

        if self.options.save_overlay:
            overlay_path = dirs["overlay"] / (stem + "_cmp.png")
            save_image(overlay_path, window_overlay(lesion))
            files["overlay"] = str(overlay_path)

        ys, xs = np.nonzero(mask_native > 127)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        fx0, fy0, fx1, fy1 = lesion.footprint
        geom = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1,
                "center": [(fx0 + fx1) // 2, (fy0 + fy1) // 2],
                "pegada_lesao": [fx0, fy0, fx1, fy1], "janela": lesion.window.to_dict()}

        metrics = {k: _finite(v) for k, v in evaluation["metrics"].items()}
        metrics.update({k: _finite(v) for k, v in native_texture_metrics(
            out, source.raw, geom, mask_native,
            background_level=self.cfg.preprocessing.breast_threshold * source.scale_to_u8).items()})
        c0 = case.framing.orig_to_canvas(fx0, fy0)
        c1 = case.framing.orig_to_canvas(fx1, fy1)
        box_canvas = LesionBox(int(np.floor(c0[0])), int(np.floor(c0[1])), int(np.ceil(c1[0])), int(np.ceil(c1[1])))
        rate = float(evaluation["plausibility"]["pass_rate"])
        anat_numeric = {k: round(float(v), 4) for k, v in anat.items() if isinstance(v, (int, float))}
        run = self.run_info
        record = LesionRecord(
            case=case.name, index=index, lesion_type=lesion_type, seed=seed, attempts=attempts,
            region=describe_region(box_canvas, pmap, view.view),
            box_canvas=box_canvas.to_dict(), box_original=geom,
            diameter_mm=round(float(np.sqrt((fx1 - fx0) * (fy1 - fy0)) * case.mm_per_px * case.framing.scale), 2),
            anatomy=anat_numeric, metrics=metrics, qc_passed=passed, qc_failed_criteria=list(failed),
            plausibility_rate=round(rate, 4),
            confidence=confidence_score(rate, anat_numeric.get("anat_score_mean", 0.0), passed),
            files=files, duration_s=round(duration, 2),
            generation={
                "modo": "janela_nativa", "prompt": lesion.prompt,
                "lora": str(run.lora_dir) if run else None,
                "lora_escala": self.cfg.native_window.lora_scale if run else None,
                "guidance": self._native_guidance(self.cfg.lesion.preset(lesion_type)),
                "tamanho_px": [fx1 - fx0, fy1 - fy0], "posicionamento": anat.get("placement"),
                "fonte_igual_ao_treino": bool(run is None or source.bit_depth == 16),
                **lesion.meta(),
                "geometria": {k: getattr(self.cfg.native_window, k) for k in (
                    "context_factor", "min_scale", "min_context", "mask_margin_frac", "feather_frac")},
            })

        LOGGER.info("%s: %s em %s (conf %.2f, %d tentativa(s), janela %d px)",
                    case.name, lesion_type, record.region, record.confidence, attempts, lesion.window.side,
                    extra={"arquivo": case.name, "tipo": lesion_type, "regiao": record.region,
                           "confianca": record.confidence, "diametro_mm": record.diameter_mm,
                           "caixa_original": geom, "seed": seed})
        return record

    def _save_anatomy_figure(self, case: Case, pmap: PlausibilityMap,
                             lesions: List[LesionRecord], out_dir: Path) -> None:
        """Figura de auditoria: regiao valida em verde, peitoral em vermelho."""
        try:
            rgb = cv.cvtColor(case.canvas, cv.COLOR_GRAY2BGR).astype(np.float32)
            valid = (pmap.score > 0)
            overlay = np.zeros_like(rgb)
            overlay[..., 1] = pmap.score * 255.0
            overlay[pmap.pectoral > 0] = (0, 0, 200)
            rgb = np.where(valid[..., None] | (pmap.pectoral > 0)[..., None],
                           0.65 * rgb + 0.35 * overlay, rgb)
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            if pmap.nipple:
                cv.circle(rgb, pmap.nipple, 8, (0, 255, 255), 2)
            for record in lesions:
                b = record.box_canvas
                cv.rectangle(rgb, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 220, 255), 3)
            save_image(out_dir / (case.name + "_anatomia.png"), rgb)
        except Exception as exc:                            # figura e acessorio
            LOGGER.debug("figura de anatomia falhou para %s: %s", case.name, exc)

    def _write_case_metadata(self, report: CaseReport, out_dir: Path) -> None:
        """Um JSON por imagem de entrada, mesmo quando ela falha."""
        payload = {
            "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "arquivo": report.path,
            "status": report.status,
            "motivo": report.reason,
            "avisos": report.warnings,
            "origem": report.source,
            "anatomia": _jsonable(report.anatomy),
            "lesoes": [_jsonable(asdict(l)) for l in report.lesions],
            "duracao_s": round(report.duration_s, 2),
            "config": {
                "anatomy": self.anatomy_cfg.to_dict(),
                "diffusion": asdict(self.cfg.diffusion),
                "composition": asdict(self.cfg.composition),
                "preprocessing": asdict(self.cfg.preprocessing),
                "native_window": asdict(self.cfg.native_window),
            },
        }
        with open(out_dir / (report.name + ".json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)


def _finite(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, 6) if np.isfinite(v) else None


def environment_info() -> Dict[str, Any]:
    """Ambiente de execucao, gravado no log para reprodutibilidade."""
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "opencv": cv.__version__,
        "numpy": np.__version__,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    try:
        import diffusers
        info["diffusers"] = diffusers.__version__
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Manifesto
# ---------------------------------------------------------------------------

MANIFEST_COLUMNS = [
    "arquivo_origem", "arquivo_saida", "status", "tipo_lesao", "regiao_anatomica",
    "x1", "y1", "x2", "y2", "largura_px", "altura_px", "diametro_mm",
    "distancia_pele_mm", "posto_densidade", "score_anatomico",
    "weber_contrast", "cnr", "ssim_imagem", "fracao_fora_alterada",
    "weber_nativo", "ruido_nativo", "nitidez_nativa",
    "qc_aprovado", "criterios_aprovados", "confianca", "seed", "tentativas", "motivo",
]


def write_manifest(reports: Iterable[CaseReport], path: Path) -> int:
    """Grava ``manifest.csv`` com uma linha por lesao e uma por imagem sem lesao."""
    rows = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for report in reports:
            if not report.lesions:
                writer.writerow({"arquivo_origem": Path(report.path).name,
                                 "arquivo_saida": "", "status": report.status,
                                 "motivo": report.reason})
                rows += 1
                continue
            for lesion in report.lesions:
                box = lesion.box_original
                writer.writerow({
                    "arquivo_origem": Path(report.path).name,
                    "arquivo_saida": Path(lesion.files["original"]).name,
                    "status": report.status,
                    "tipo_lesao": lesion.lesion_type,
                    "regiao_anatomica": lesion.region,
                    "x1": box["x1"], "y1": box["y1"], "x2": box["x2"], "y2": box["y2"],
                    "largura_px": box["width"], "altura_px": box["height"],
                    "diametro_mm": lesion.diameter_mm,
                    "distancia_pele_mm": lesion.anatomy.get("skin_distance_mm_min"),
                    "posto_densidade": lesion.anatomy.get("density_rank_mean"),
                    "score_anatomico": lesion.anatomy.get("anat_score_mean"),
                    "weber_contrast": lesion.metrics.get("weber_contrast"),
                    "cnr": lesion.metrics.get("cnr"),
                    "ssim_imagem": lesion.metrics.get("ssim_full_image"),
                    "fracao_fora_alterada": lesion.metrics.get("outside_changed_fraction"),
                    "weber_nativo": lesion.metrics.get("native_weber_contrast"),
                    "ruido_nativo": lesion.metrics.get("native_noise_ratio_roi_ring"),
                    "nitidez_nativa": lesion.metrics.get("native_sharpness_ratio"),
                    "qc_aprovado": lesion.qc_passed,
                    "criterios_aprovados": lesion.plausibility_rate,
                    "confianca": lesion.confidence,
                    "seed": lesion.seed,
                    "tentativas": lesion.attempts,
                    "motivo": report.reason,
                })
                rows += 1
    return rows


__all__ = [
    "SyntheticLesionPipeline", "PipelineOptions", "CaseReport", "LesionRecord",
    "ValidationError", "setup_logging", "parse_view", "load_view_table", "resolve_view",
    "ViewInfo", "write_manifest",
    "environment_info", "confidence_score", "validate_path", "validate_source",
]
