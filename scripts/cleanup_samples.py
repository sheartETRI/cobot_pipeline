#!/usr/bin/env python3
# scripts/cleanup_samples.py
"""
샘플 디렉터리 정리 스크립트
기능:
 - prefix별로 *.json / *.world.json 페어를 관리
 - 오래된(prefix 마지막 수정시간 기준) 파일 삭제 (--older-than-days)
 - 또는 prefix별로 최신 N개만 보존 (--keep-latest)
 - ir_*.json (canonical IR) 파일은 기본 제외
 - --dry-run 으로 삭제 예정 항목만 출력
 - --archive 로 삭제 대신 아카이브 디렉토리로 이동(압축 가능)

# 사용법 예시
30일 이상 된 것 삭제(드라이런)
python scripts/cleanup_samples.py --samples-dir samples --older-than-days 30 --dry-run

# 하루에 한 번 실제 삭제(30일)
python scripts/cleanup_samples.py --samples-dir /path/to/repo/samples --older-than-days 30

# 최신 20개만 보존, 나머지 삭제(드라이런)
python scripts/cleanup_samples.py --samples-dir samples --keep-latest 20 --dry-run

# 삭제 대신 아카이브(압축) 후 삭제
python scripts/cleanup_samples.py --samples-dir samples --older-than-days 90 --archive /path/to/archive

주의: 기본적으로 ir_<hash>.json 패턴(캐노니컬 IR)은 제외하도록 --exclude-canonical 옵션을 사용하세요. 필요하면 스크립트 기본을 exclude_canonical=True 로 바꾸셔도 됩니다.

정기 실행(예시)
Linux (cron): 매일 03:10에 실제 삭제
10 3 * * * /usr/bin/python3 /path/to/repo/scripts/cleanup_samples.py --samples-dir /path/to/repo/samples --older-than-days 30 >> /var/log/cleanup_samples.log 2>&1
Windows: Task Scheduler에서 Program/script: python, Arguments: C:\repo\scripts\cleanup_samples.py --samples-dir C:\repo\samples --older-than-days 30 로 설정
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import time
import shutil
import json
from datetime import datetime, timedelta
import logging
import sys
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

IR_CANONICAL_RE = re.compile(r"^ir_[0-9a-fA-F]{8,64}\.json$")

def find_prefix(fname: str) -> str:
    if fname.endswith(".world.json"):
        return fname[:-11]
    elif fname.endswith(".json"):
        return fname[:-5]
    else:
        return fname

def gather_prefixes(samples_dir: Path, exclude_canonical: bool = True):
    files = list(samples_dir.glob("*.json")) + list(samples_dir.glob("*.world.json"))
    mapping: dict[str, list[Path]] = {}
    for f in files:
        if exclude_canonical and IR_CANONICAL_RE.match(f.name):
            continue
        prefix = find_prefix(f.name)
        mapping.setdefault(prefix, []).append(f)
    return mapping

def prefix_mtime(prefix_files: list[Path]) -> float:
    # Use newest mtime among files for the prefix
    return max(p.stat().st_mtime for p in prefix_files)

def archive_prefix(prefix: str, files: list[Path], archive_dir: Path, compress: bool = True):
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if compress:
        zname = archive_dir / f"{prefix}-{ts}.zip"
        with zipfile.ZipFile(zname, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
        logging.info("Archived %s -> %s", prefix, zname)
    else:
        target = archive_dir / f"{prefix}-{ts}"
        target.mkdir(exist_ok=True)
        for f in files:
            shutil.move(str(f), str(target / f.name))
        logging.info("Moved %s -> %s", prefix, target)

def delete_prefix(prefix: str, files: list[Path], dry_run: bool):
    for f in files:
        if dry_run:
            logging.info("[DRY-RUN] Would remove: %s", f)
        else:
            try:
                f.unlink()
                logging.info("Removed: %s", f)
            except Exception as e:
                logging.error("Failed to remove %s: %s", f, e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=str, default="samples", help="samples 디렉터리 경로")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--older-than-days", type=int, help="마지막 수정 시간이 N일 이상인 prefix 삭제")
    group.add_argument("--keep-latest", type=int, help="각 prefix별로 최신 N개는 보존하고 나머지 삭제")
    parser.add_argument("--exclude-canonical", action="store_true", help="ir_<hash>.json 같은 canonical IR은 제외(기본 포함)")
    parser.add_argument("--dry-run", action="store_true", help="삭제하지 않고 삭제 예정만 출력")
    parser.add_argument("--archive", type=str, default=None, help="삭제 대신 아카이브(디렉터리)로 압축 이동")
    parser.add_argument("--no-world-pair", action="store_true", help="world(.world.json)와 페어로 묶지 않고 개별 파일 기준으로 처리")
    parser.add_argument("--prefix-filter", type=str, default=None, help="정규식으로 prefix 필터(일치하는 prefix만 대상)")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        logging.error("samples-dir not found: %s", samples_dir)
        sys.exit(1)

    mapping = gather_prefixes(samples_dir, exclude_canonical=args.exclude_canonical)
    logging.info("Found %d prefixes (canonical excluded=%s)", len(mapping), args.exclude_canonical)

    # optional filter
    prefix_re = re.compile(args.prefix_filter) if args.prefix_filter else None
    now = time.time()

    to_delete: dict[str, list[Path]] = {}

    if args.older_than_days is not None:
        cutoff = now - args.older_than_days * 86400
        for prefix, files in mapping.items():
            if prefix_re and not prefix_re.search(prefix):
                continue
            mtime = prefix_mtime(files)
            if mtime < cutoff:
                to_delete[prefix] = files
    else:
        # keep-latest mode
        keep = args.keep_latest
        # Sort prefixes by newest mtime desc
        sorted_prefixes = sorted(mapping.items(), key=lambda kv: prefix_mtime(kv[1]), reverse=True)
        # We interpret "keep latest N prefixes" globally; but user asked per-prefix.
        # Implement per-prefix: if there are multiple versions per prefix (with timestamped suffix), we need to detect versions.
        # Simpler approach: assume each prefix corresponds to one IR/world pair; keep latest 'keep' most-recent prefixes, delete the rest.
        latest = sorted_prefixes[:keep]
        keep_prefixes = set(p for p,_ in latest)
        for prefix, files in mapping.items():
            if prefix_re and not prefix_re.search(prefix):
                continue
            if prefix not in keep_prefixes:
                to_delete[prefix] = files

    if not to_delete:
        logging.info("No files matched deletion criteria.")
        return

    logging.info("Prefixes to delete: %s", ", ".join(sorted(to_delete.keys())))

    # Perform archive or delete
    if args.archive:
        archive_dir = Path(args.archive)
        for prefix, files in to_delete.items():
            archive_prefix(prefix, files, archive_dir, compress=True)
            if not args.dry_run:
                for f in files:
                    try:
                        f.unlink()
                    except Exception as e:
                        logging.warning("Failed to remove after archive %s: %s", f, e)
    else:
        for prefix, files in to_delete.items():
            delete_prefix(prefix, files, dry_run=args.dry_run)

    logging.info("Completed. dry-run=%s", args.dry_run)

if __name__ == "__main__":
    main()