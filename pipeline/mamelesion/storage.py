"""Acesso ao MinIO (S3) do CEIA: listar, baixar e enviar pastas inteiras.

Credenciais vem das variaveis ``MINIO_ENDPOINT``, ``MINIO_ACCESS_KEY``,
``MINIO_SECRET_KEY`` e ``MINIO_SECURE``; o que faltar e lido de
``pipeline/.env.minio``. Nada de credencial fica no codigo.

Enderecos remotos sao escritos ``bucket/prefixo`` (``teste/birads_1``).
Baixar e enviar pulam arquivos que ja existem no destino com o mesmo tamanho,
entao repetir um comando interrompido so transfere o que faltou.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from minio import Minio
from minio.error import S3Error

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env.minio"


def _read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            values[key.strip()] = val.strip().strip("'\"")
    return values


def get_client(env_file: Path = ENV_FILE) -> Minio:
    """Cliente MinIO a partir do ambiente, com ``.env.minio`` como reserva."""
    file_values = _read_env_file(env_file)

    def get(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(key) or file_values.get(key) or default

    endpoint = get("MINIO_ENDPOINT")
    access, secret = get("MINIO_ACCESS_KEY"), get("MINIO_SECRET_KEY")
    if not (endpoint and access and secret):
        raise RuntimeError(f"MINIO_ENDPOINT/ACCESS_KEY/SECRET_KEY ausentes (ambiente ou {env_file})")
    # aceita a URL completa (https://host) alem do host puro
    secure = get("MINIO_SECURE", "true").lower() not in ("0", "false", "no")
    if "://" in endpoint:
        scheme, endpoint = endpoint.split("://", 1)
        secure = scheme == "https"
    return Minio(endpoint.rstrip("/"), access_key=access, secret_key=secret, secure=secure)


def split_remote(remote: str) -> Tuple[str, str]:
    """``bucket/a/b`` -> (``bucket``, ``a/b``)."""
    bucket, _, prefix = remote.strip("/").partition("/")
    return bucket, prefix


@dataclass
class TransferStats:
    transferred: int = 0
    skipped: int = 0
    bytes: int = 0
    failed: List[str] = field(default_factory=list)


def iter_objects(client: Minio, remote: str) -> Iterator:
    bucket, prefix = split_remote(remote)
    yield from client.list_objects(bucket, prefix=prefix or None, recursive=True)


def _is_object(client: Minio, bucket: str, name: str) -> bool:
    try:
        client.stat_object(bucket, name)
        return True
    except S3Error as exc:
        if exc.code in ("NoSuchKey", "NoSuchObject"):
            return False
        raise


def _run(tasks: List[Callable[[], Tuple[bool, int]]], names: List[str], workers: int,
         progress: Optional[Callable[[int, int], None]]) -> TransferStats:
    stats = TransferStats()

    def safe(i: int):
        try:
            return i, tasks[i](), None
        except (S3Error, OSError) as exc:
            return i, None, exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (i, result, exc) in enumerate(pool.map(safe, range(len(tasks))), 1):
            if exc is not None:
                stats.failed.append(f"{names[i]}: {exc}")
            elif result[0]:
                stats.transferred += 1
                stats.bytes += result[1]
            else:
                stats.skipped += 1
            if progress:
                progress(done, len(tasks))
    return stats


def download(client: Minio, remote: str, dest: Path, workers: int = 8,
             progress: Optional[Callable[[int, int], None]] = None) -> TransferStats:
    """Baixa tudo sob ``bucket/prefixo`` para ``dest``, preservando a arvore abaixo do prefixo."""
    bucket, prefix = split_remote(remote)
    dest = Path(dest)
    if prefix and _is_object(client, bucket, prefix):
        # um objeto so (teste/a/b.png) vira dest/b.png
        objs = [client.stat_object(bucket, prefix)]
        base = prefix.rpartition("/")[0] + "/" if "/" in prefix else ""
    else:
        # "birads_1" nao deve pegar "birads_10/": trata o prefixo como pasta
        base = f"{prefix}/" if prefix else ""
        objs = [o for o in client.list_objects(bucket, prefix=base or None, recursive=True) if not o.is_dir]

    def make(obj):
        target = dest / obj.object_name[len(base):]

        def task():
            if target.is_file() and target.stat().st_size == obj.size:
                return False, 0
            target.parent.mkdir(parents=True, exist_ok=True)
            client.fget_object(bucket, obj.object_name, str(target))
            return True, obj.size
        return task

    return _run([make(o) for o in objs], [o.object_name for o in objs], workers, progress)


def upload(client: Minio, src: Path, remote: str, workers: int = 8,
           progress: Optional[Callable[[int, int], None]] = None) -> TransferStats:
    """Envia ``src`` (arquivo ou pasta, recursivo) para ``bucket/prefixo``."""
    bucket, prefix = split_remote(remote)
    src = Path(src)
    if not client.bucket_exists(bucket):
        raise RuntimeError(f"bucket inexistente: {bucket}")
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    root = src.parent if src.is_file() else src
    prefix = f"{prefix.rstrip('/')}/" if prefix else ""
    remote_sizes = {o.object_name: o.size for o in iter_objects(client, f"{bucket}/{prefix}")}

    def make(path: Path):
        name = prefix + path.relative_to(root).as_posix()

        def task():
            size = path.stat().st_size
            if remote_sizes.get(name) == size:
                return False, 0
            client.fput_object(bucket, name, str(path))
            return True, size
        return task

    return _run([make(p) for p in files], [str(p) for p in files], workers, progress)
