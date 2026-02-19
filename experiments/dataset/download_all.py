#!/usr/bin/env python3
"""
Parallel AVA video downloader.

Downloads the 900-1800 s segment of every unique video in an AVA-format CSV
and saves it as {video_id}.mp4 in AVA_VIDEO_DIR.

Usage
-----
    export AVA_VIDEO_DIR=/data/ava/videos

    # single CSV
    python experiments/dataset/download_all.py --csv experiments/experiment_01/train.csv

    # multiple CSVs at once (train + test in one pass, de-duplicated)
    python experiments/dataset/download_all.py \
        --csv experiments/experiment_01/train.csv \
        --csv experiments/experiment_01/test.csv \
        --workers 6

Environment
-----------
    AVA_VIDEO_DIR   Required. Output directory for .mp4 files.

Options
-------
    --csv       Path to AVA-format annotation CSV (repeatable).
    --workers   Parallel download workers. Default: 4.
    --retries   Per-video retry attempts.  Default: 3.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── constants ─────────────────────────────────────────────────────────────────

_MIN_FILE_BYTES = 5_000_000  # 5 MB — rejects truncated / empty files
_URL_TIMEOUT    = 45         # seconds — yt-dlp URL resolution
_FFMPEG_TIMEOUT = 1800       # seconds — ffmpeg download + encode (30 min max)

# Format selector: prefer H.264 MP4 ≤ 480 p for speed.
# SlowFast crops to 224 px, so higher resolution is wasted bandwidth.
# Falls back to any MP4 if a constrained format is unavailable.
_YTDLP_FORMAT = (
    "bestvideo[ext=mp4][vcodec^=avc][height<=480]"
    "/bestvideo[ext=mp4][height<=480]"
    "/bestvideo[ext=mp4]"
)


# ── single-video worker ───────────────────────────────────────────────────────

def _download_one(video_id: str, video_dir: Path, retries: int, ffmpeg_threads: int) -> tuple:
    """
    Worker function executed inside the thread pool.

    Returns ("ok"|"skip"|"fail", video_id, detail_string).
    Writes to a .tmp file first; renames to .mp4 on success → atomic write,
    no corrupt files left behind on interruption or error.
    """
    out  = video_dir / f"{video_id}.mp4"
    tmp  = video_dir / f"{video_id}.tmp.mp4"
    url  = f"https://www.youtube.com/watch?v={video_id}"

    if _is_valid(out):
        return "skip", video_id, f"{out.stat().st_size // 1_000_000} MB already on disk"

    for attempt in range(1, retries + 1):
        tmp.unlink(missing_ok=True)
        try:
            # ── step 1: resolve direct stream URL ─────────────────────────
            stream_url = subprocess.check_output(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "--socket-timeout", "30",
                    "-f", _YTDLP_FORMAT,
                    "-g", url,
                ],
                timeout=_URL_TIMEOUT,
                stderr=subprocess.PIPE,      # capture stderr to detect perm errors
            ).decode().strip()

            if not stream_url:
                return "fail", video_id, "yt-dlp returned empty URL"

            # ── step 2: download + trim + encode ──────────────────────────
            # -ss before -i  → fast seek (no full decode before trim point).
            # -t             → clip to exactly 900 s.
            # Re-encoding with ultrafast ensures the output starts at the
            # exact seek position (copy codec can misalign on keyframes).
            subprocess.run(
                [
                    "ffmpeg",
                    "-ss", "900", "-t", "900",
                    "-i", stream_url,
                    "-c:v", "libx264",
                    "-crf", "23",
                    "-preset", "ultrafast",  # fastest encode; file quality fine for 224 px
                    "-threads", str(ffmpeg_threads),
                    "-r", "30",   # constant 30 FPS for SlowFast compatibility
                    "-an",        # strip audio — not used in training
                    "-y",
                    str(tmp),
                ],
                timeout=_FFMPEG_TIMEOUT,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if not _is_valid(tmp):
                raise RuntimeError(f"output too small after encode ({tmp.stat().st_size if tmp.exists() else 0} B)")

            tmp.rename(out)   # atomic — visible to other processes only when complete
            size_mb = out.stat().st_size // 1_000_000
            return "ok", video_id, f"{size_mb} MB"

        except subprocess.CalledProcessError as e:
            # yt-dlp non-zero exit → video probably deleted / private / geo-blocked.
            # Detect stderr keywords that indicate a permanent failure.
            stderr = e.stderr.decode(errors="replace") if e.stderr else ""
            if any(kw in stderr for kw in ("Video unavailable", "Private video", "removed")):
                tmp.unlink(missing_ok=True)
                return "fail", video_id, "video unavailable (permanent)"
            # Otherwise transient error — retry
            detail = f"CalledProcessError attempt {attempt}"

        except subprocess.TimeoutExpired:
            detail = f"timeout attempt {attempt}"

        except KeyboardInterrupt:
            tmp.unlink(missing_ok=True)
            raise  # propagate so the pool shuts down cleanly

        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc} attempt {attempt}"

        tmp.unlink(missing_ok=True)

        if attempt < retries:
            time.sleep(5 * attempt)  # exponential back-off: 5 s, 10 s, 15 s

    return "fail", video_id, f"all {retries} attempts failed"


# ── orchestrator ──────────────────────────────────────────────────────────────

def download_all(
    video_ids: list,
    video_dir: str | Path,
    workers: int = 4,
    retries: int = 3,
) -> dict:
    """
    Download a list of AVA video IDs in parallel.

    Returns a dict with keys "ok", "skip", "fail", each mapping to a list of
    (video_id, detail) tuples.
    """
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Divide CPU cores evenly across workers — avoids thrashing when all
    # ffmpeg instances try to use every core simultaneously.
    cpu_cores      = os.cpu_count() or 4
    ffmpeg_threads = max(1, cpu_cores // workers)

    results: dict[str, list] = {"ok": [], "skip": [], "fail": []}
    executor = ThreadPoolExecutor(max_workers=workers)

    try:
        futures = {
            executor.submit(_download_one, vid, video_dir, retries, ffmpeg_threads): vid
            for vid in video_ids
        }

        bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        with tqdm(total=len(futures), unit="video", bar_format=bar_fmt, desc="Downloading") as pbar:
            for future in as_completed(futures):
                try:
                    status, vid, detail = future.result()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    status, vid, detail = "fail", futures[future], str(exc)

                results[status].append((vid, detail))
                pbar.set_postfix(
                    ok=len(results["ok"]),
                    skip=len(results["skip"]),
                    fail=len(results["fail"]),
                    refresh=False,
                )
                pbar.update(1)

    except KeyboardInterrupt:
        print("\nInterrupted — cancelling pending downloads …", file=sys.stderr)
        executor.shutdown(wait=False, cancel_futures=True)
        # Partial results are still returned
    else:
        executor.shutdown(wait=True)

    return results


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_valid(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= _MIN_FILE_BYTES


def _collect_video_ids(csv_paths: list) -> list:
    """Read all unique video_ids from one or more AVA-format CSVs."""
    frames = [pd.read_csv(p, names=["video_id", "ts", "x1", "y1", "x2", "y2", "action", "person_id"]) for p in csv_paths]
    combined = pd.concat(frames, ignore_index=True)
    return sorted(combined["video_id"].unique().tolist())


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel AVA video downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", metavar="PATH", action="append", required=True,
        help="AVA-format annotation CSV (repeat to merge multiple files)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="Retry attempts per video (default: 3)",
    )
    args = parser.parse_args()

    video_dir = os.environ.get("AVA_VIDEO_DIR", "").strip()
    if not video_dir:
        print("Error: AVA_VIDEO_DIR environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    video_ids = _collect_video_ids(args.csv)

    print(f"Videos   : {len(video_ids)} unique IDs across {len(args.csv)} CSV file(s)")
    print(f"Output   : {video_dir}")
    print(f"Workers  : {args.workers}  |  Retries: {args.retries}")
    print(f"Format   : {_YTDLP_FORMAT}")
    print()

    t0 = time.monotonic()
    results = download_all(video_ids, video_dir, workers=args.workers, retries=args.retries)
    elapsed = time.monotonic() - t0

    ok   = results["ok"]
    skip = results["skip"]
    fail = results["fail"]

    print(f"\n{'─' * 50}")
    print(f"Downloaded : {len(ok)}")
    print(f"Skipped    : {len(skip)}  (already on disk)")
    print(f"Failed     : {len(fail)}")
    print(f"Time       : {elapsed:.0f} s")

    if fail:
        print("\nFailed videos:")
        for vid, detail in fail:
            print(f"  {vid:20s}  {detail}")

    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
