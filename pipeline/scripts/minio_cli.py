#!/usr/bin/env python
"""Linha de comando para o MinIO do CEIA (credenciais em ``pipeline/.env.minio``).

    .venv/bin/python scripts/minio_cli.py ls                         # buckets
    .venv/bin/python scripts/minio_cli.py ls teste                   # pastas do bucket
    .venv/bin/python scripts/minio_cli.py ls teste/birads_1 --arquivos
    .venv/bin/python scripts/minio_cli.py get teste/birads_1 ../dados/birads_1
    .venv/bin/python scripts/minio_cli.py put ../dataset_escolhido teste/dataset_escolhido

``get`` e ``put`` sao recursivos e pulam arquivos ja presentes com o mesmo
tamanho; rodar de novo retoma uma transferencia interrompida.
"""

import argparse
import signal
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mamelesion.storage import download, get_client, iter_objects, split_remote, upload   # noqa: E402


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def progress(done: int, total: int) -> None:
    if done == total or done % 50 == 0:
        print(f"\r  {done}/{total}", end="\n" if done == total else "", flush=True)


def cmd_ls(client, args) -> int:
    if not args.remoto:
        for b in client.list_buckets():
            print(f"{b.name:30s} {b.creation_date:%Y-%m-%d}")
        return 0
    bucket, prefix = split_remote(args.remoto)
    base = f"{prefix}/" if prefix else ""
    groups = defaultdict(lambda: [0, 0])
    for obj in iter_objects(client, f"{bucket}/{base}"):
        if args.arquivos:
            print(f"{human(obj.size):>10s}  {obj.last_modified:%Y-%m-%d %H:%M}  {obj.object_name}")
            continue
        rest = obj.object_name[len(base):]
        key = rest.split("/", 1)[0] + ("/" if "/" in rest else "")
        groups[key][0] += 1
        groups[key][1] += obj.size
    for key, (n, size) in sorted(groups.items()):
        print(f"{n:8d} obj  {human(size):>10s}  {key}")
    return 0


def report(stats, verb: str) -> int:
    print(f"{verb}: {stats.transferred} ({human(stats.bytes)}), ja presentes: {stats.skipped}, "
          f"falhas: {len(stats.failed)}")
    for line in stats.failed[:20]:
        print("  FALHA", line)
    return 1 if stats.failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ls", help="lista buckets, pastas ou arquivos")
    p.add_argument("remoto", nargs="?", help="bucket ou bucket/prefixo")
    p.add_argument("--arquivos", action="store_true", help="um objeto por linha em vez de agrupar por pasta")
    p = sub.add_parser("get", help="baixa bucket/prefixo para uma pasta local")
    p.add_argument("remoto")
    p.add_argument("destino", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("put", help="envia arquivo ou pasta local para bucket/prefixo")
    p.add_argument("origem", type=Path)
    p.add_argument("remoto")
    p.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    client = get_client()
    if args.cmd == "ls":
        return cmd_ls(client, args)
    if args.cmd == "get":
        return report(download(client, args.remoto, args.destino, args.workers, progress), "baixados")
    if not args.origem.exists():
        ap.error(f"origem inexistente: {args.origem}")
    return report(upload(client, args.origem, args.remoto, args.workers, progress), "enviados")


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # `| head` sem traceback
    sys.exit(main())
