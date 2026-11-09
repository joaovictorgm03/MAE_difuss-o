"""Recortes reais BI-RADS 4/5 e mamografias hospedeiras: descoberta, auditoria e divisao.

Os recortes sao a referencia do ajuste fino: cada um e a caixa de uma lesao
anotada, em 16 bits, na resolucao original. Nao ha mamografia inteira para
nenhum deles, por isso o treino cola o recorte em tecido saudavel da mesma fonte
(:mod:`mamelesion.paste`).

Tres cuidados que este modulo garante e os testes verificam:

1. **Divisao por paciente.** Ha ate 4 recortes do mesmo paciente (mesma lesao em
   CC e MLO, ou lesoes vizinhas). Dividir por recorte vazaria a mesma lesao para
   treino e teste e inflaria qualquer metrica de semelhanca.
2. **Estratificacao por classe.** BI-RADS 4 tem so 15 pacientes; sem estratificar,
   uma semente azarada deixa validacao ou teste sem nenhum caso da classe.
3. **Escala de intensidade fixa.** Recortes sao lidos como ``raw / 65535``, a
   mesma escala das mamografias inteiras desta fonte (maximo 65535 em todas).
   Normalizar cada recorte pelo proprio maximo alteraria o contraste que o
   modelo precisa aprender.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

#: classes, com o prompt usado no ajuste e na inferencia
CLASS_PROMPTS: Dict[str, str] = {
    "birads4": "a mammogram with a birads 4 lesion",
    "birads5": "a mammogram with a birads 5 lesion",
}
#: prompt do checkpoint original (unico usado no treino dos autores)
GENERIC_PROMPT = "a mammogram with a lesion"

#: saturacao das mamografias de 16 bits desta fonte
CROP_REFERENCE = 65535.0
MIN_CROP_SIDE = 48

_NAME = re.compile(r"^(?P<patient>\d+)_(?P<exam>\d+)_(?P<side>[LR])_(?P<view>CC|MLO)")


@dataclass
class LesionCrop:
    """Um recorte real de lesao, em float32 [0, 1] na resolucao original."""

    name: str
    path: Path
    lesion_class: str
    patient: str
    study: str
    image: np.ndarray = field(repr=False)

    @property
    def wh(self) -> Tuple[int, int]:
        return int(self.image.shape[1]), int(self.image.shape[0])


def parse_ids(name: str) -> Tuple[str, str]:
    """``(paciente, estudo)`` a partir de ``<paciente>_<exame>_<L|R>_<CC|MLO>...``."""
    m = _NAME.match(name)
    if not m:
        raise ValueError("nome fora da convencao paciente_exame_lado_vista: " + name)
    return m.group("patient"), "{}_{}".format(m.group("patient"), m.group("exam"))


def read_crop(path: Path) -> np.ndarray:
    """Le um recorte de 16 bits em float32 [0, 1]. Levanta ``ValueError`` se invalido."""
    try:
        im = Image.open(path)
        im.load()
    except Exception as exc:
        raise ValueError("nao decodifica: {}".format(type(exc).__name__)) from exc
    arr = np.asarray(im)
    if arr.ndim != 2:
        raise ValueError("esperado 2D, veio {}".format(arr.shape))
    if arr.dtype != np.uint16:
        raise ValueError("esperado 16 bits, veio {}".format(arr.dtype))
    if min(arr.shape) < MIN_CROP_SIDE:
        raise ValueError("pequeno demais: {}x{}".format(arr.shape[1], arr.shape[0]))
    if int(arr.max()) == int(arr.min()):
        raise ValueError("constante")
    return arr.astype(np.float32) / CROP_REFERENCE


def discover_crops(sources: Dict[str, Sequence[Path]]) -> Tuple[List[LesionCrop], List[Dict[str, str]]]:
    """Le todos os PNG de cada pasta de classe (recursivo).

    Devolve ``(recortes_validos, rejeitados)``; cada rejeitado traz arquivo e
    motivo. Nomes repetidos entre pastas sao rejeitados na segunda ocorrencia:
    o nome e a chave da divisao e dos metadados.
    """
    crops: List[LesionCrop] = []
    rejected: List[Dict[str, str]] = []
    seen: Dict[str, Path] = {}
    for lesion_class, dirs in sources.items():
        if lesion_class not in CLASS_PROMPTS:
            raise ValueError("classe desconhecida: {} (validas: {})".format(
                lesion_class, ", ".join(CLASS_PROMPTS)))
        for directory in dirs:
            for path in sorted(Path(directory).rglob("*.png")):
                reason = None
                if path.name in seen:
                    reason = "nome repetido (ja lido de {})".format(seen[path.name])
                else:
                    try:
                        patient, study = parse_ids(path.name)
                        image = read_crop(path)
                    except ValueError as exc:
                        reason = str(exc)
                if reason:
                    rejected.append({"arquivo": str(path), "classe": lesion_class, "motivo": reason})
                    continue
                seen[path.name] = path
                crops.append(LesionCrop(path.name, path, lesion_class, patient, study, image))
    return crops, rejected


def split_by_patient(items: Iterable[Tuple[str, str, str]],
                     fractions: Tuple[float, float, float] = (0.70, 0.15, 0.15),
                     seed: int = 20260916) -> Dict[str, List[str]]:
    """Divide ``(chave, paciente, estrato)`` em treino/validacao/teste por paciente.

    Cada estrato (classe) e dividido separadamente; todos os itens de um paciente
    caem na mesma particao. Um paciente presente em dois estratos e atribuido ao
    estrato com mais itens dele, para nao ser sorteado duas vezes. Validacao e
    teste recebem pelo menos um paciente por estrato quando o estrato tem tres
    ou mais.
    """
    items = list(items)
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fracoes devem somar 1: {}".format(fractions))

    by_patient: Dict[str, List[Tuple[str, str]]] = {}
    for key, patient, stratum in items:
        by_patient.setdefault(patient, []).append((key, stratum))

    strata: Dict[str, List[str]] = {}
    for patient, rows in sorted(by_patient.items()):
        counts: Dict[str, int] = {}
        for _, stratum in rows:
            counts[stratum] = counts.get(stratum, 0) + 1
        home = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        strata.setdefault(home, []).append(patient)

    rng = np.random.default_rng(seed)
    assignment: Dict[str, str] = {}
    for stratum in sorted(strata):
        patients = sorted(strata[stratum])
        order = [patients[i] for i in rng.permutation(len(patients))]
        n = len(order)
        n_val = int(round(fractions[1] * n))
        n_test = int(round(fractions[2] * n))
        if n >= 3:
            n_val, n_test = max(1, n_val), max(1, n_test)
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError("estrato {} com {} paciente(s) nao comporta a divisao".format(stratum, n))
        for i, patient in enumerate(order):
            assignment[patient] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for key, patient, _ in items:
        out[assignment[patient]].append(key)
    for part in out:
        out[part].sort()
    return out


def find_hosts(directory: Path) -> List[Path]:
    """Mamografias saudaveis da **mesma fonte** dos recortes: 16 bits e mesma convencao de nome.

    As de 8 bits (``<id>_IM-<n>_...``) sao de outro equipamento; usa-las como
    contexto no treino ensinaria o modelo a misturar duas texturas de detector.
    """
    hosts = []
    for path in sorted(Path(directory).glob("*.png")):
        if not _NAME.match(path.name):
            continue
        try:
            with Image.open(path) as im:
                if im.mode not in {"I;16", "I;16B", "I;16L"}:
                    continue
        except Exception:
            continue
        hosts.append(path)
    return hosts


def size_catalog(crops: Iterable[LesionCrop]) -> Dict[str, List[List[int]]]:
    """``{classe: [[largura, altura], ...]}`` dos recortes -- geometria real para a inferencia."""
    out: Dict[str, List[List[int]]] = {}
    for crop in crops:
        out.setdefault(crop.lesion_class, []).append(list(crop.wh))
    return {k: sorted(v) for k, v in sorted(out.items())}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["CLASS_PROMPTS", "GENERIC_PROMPT", "CROP_REFERENCE", "LesionCrop", "parse_ids",
           "read_crop", "discover_crops", "split_by_patient", "find_hosts", "size_catalog",
           "save_json", "load_json"]
