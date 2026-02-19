"""
AVA video downloader — streams YouTube clips on-demand to save disk space.

Public API
----------
download_video(video_id, video_dir, retries=3) -> bool
    Download a single AVA clip (seconds 900-1800) if not already present.
    Returns True when the file is ready, False when all retries failed.

download_batch(video_ids, video_dir, retries=3) -> tuple[list, list]
    Download a batch of clips.
    Returns (available_ids, newly_downloaded_ids).
    available_ids      — videos ready for training (pre-existing + newly downloaded)
    newly_downloaded_ids — videos downloaded THIS call (safe to delete afterwards)

delete_batch(video_ids, video_dir) -> None
    Delete a list of video files.
"""

import os
import subprocess
import time

import pandas as pd

# ── timeout constants ─────────────────────────────────────────────────────────
_URL_TIMEOUT = 60       # seconds: yt-dlp URL resolution
_FFMPEG_TIMEOUT = 1800  # seconds: ffmpeg download + encode (30 min max)
_MIN_FILE_BYTES = 1_000_000  # 1 MB — anything smaller is treated as corrupt


# ── single-video operations ───────────────────────────────────────────────────

def download_video(video_id: str, video_dir: str, retries: int = 3) -> bool:
    """
    Download and trim the AVA segment (900-1800 s) of a YouTube clip.

    The output file is ``{video_dir}/{video_id}.mp4``.
    Re-encodes to H.264 at constant 30 FPS, no audio, CRF 23 (≈300-600 MB).

    Returns True if the file is ready (was already present or just downloaded).
    Returns False if all retries failed.
    """
    out_path = os.path.join(video_dir, f"{video_id}.mp4")

    if _is_valid(out_path):
        return True  # already on disk

    url = f"https://www.youtube.com/watch?v={video_id}"

    for attempt in range(1, retries + 1):
        try:
            print(f"  [{video_id}] resolving stream URL (attempt {attempt}/{retries}) …")
            stream_url = subprocess.check_output(
                ["yt-dlp", "-g", "-f", "bestvideo[ext=mp4]", url],
                timeout=_URL_TIMEOUT,
                stderr=subprocess.DEVNULL,
            ).decode().strip()

            if not stream_url:
                raise ValueError("yt-dlp returned an empty URL")

            print(f"  [{video_id}] downloading & trimming AVA segment …")
            subprocess.run(
                [
                    "ffmpeg",
                    "-ss", "900", "-t", "900",
                    "-i", stream_url,
                    "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                    "-r", "30",   # constant 30 FPS for SlowFast compatibility
                    "-an",        # drop audio
                    "-y",         # overwrite without prompting
                    out_path,
                ],
                timeout=_FFMPEG_TIMEOUT,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if _is_valid(out_path):
                size_mb = os.path.getsize(out_path) // 1_000_000
                print(f"  [{video_id}] ready ({size_mb} MB)")
                return True

            # File too small → probably a failed partial encode
            _remove_if_exists(out_path)
            print(f"  [{video_id}] output file too small, retrying …")

        except subprocess.TimeoutExpired:
            print(f"  [{video_id}] timeout on attempt {attempt}")
        except subprocess.CalledProcessError as e:
            print(f"  [{video_id}] subprocess error on attempt {attempt}: {e}")
        except KeyboardInterrupt:
            _remove_if_exists(out_path)
            raise
        except Exception as e:
            print(f"  [{video_id}] unexpected error on attempt {attempt}: {e}")

        if attempt < retries:
            wait = 5 * attempt
            print(f"  [{video_id}] waiting {wait}s before retry …")
            time.sleep(wait)

    print(f"  [{video_id}] failed after {retries} attempts — skipping")
    return False


# ── batch operations ──────────────────────────────────────────────────────────

def download_batch(
    video_ids: list,
    video_dir: str,
    retries: int = 3,
) -> tuple:
    """
    Download a batch of videos.

    Returns
    -------
    available_ids : list
        IDs whose .mp4 file is ready for training
        (pre-existing files + successfully downloaded files).
    newly_downloaded_ids : list
        IDs that were downloaded during THIS call.
        Pass this list to ``delete_batch`` to clean up without
        accidentally removing files that were already on disk.
    """
    os.makedirs(video_dir, exist_ok=True)
    available, newly_downloaded = [], []

    for vid_id in video_ids:
        out_path = os.path.join(video_dir, f"{vid_id}.mp4")
        pre_existing = _is_valid(out_path)

        try:
            ok = download_video(vid_id, video_dir, retries)
        except KeyboardInterrupt:
            print("\nDownload interrupted — returning partial results.")
            break

        if ok:
            available.append(vid_id)
            if not pre_existing:
                newly_downloaded.append(vid_id)

    return available, newly_downloaded


def delete_batch(video_ids: list, video_dir: str) -> None:
    """Delete video files for a list of video IDs."""
    for vid_id in video_ids:
        path = os.path.join(video_dir, f"{vid_id}.mp4")
        if os.path.exists(path):
            os.remove(path)
            print(f"  [{vid_id}] deleted")


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_valid(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) >= _MIN_FILE_BYTES


def _remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


# ── standalone entry point ────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download AVA video clips.")
    parser.add_argument("--csv", required=True, help="Path to AVA-format CSV (train or val)")
    parser.add_argument("--video-dir", required=True, help="Output directory for .mp4 files")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    video_ids = df["video_id"].unique().tolist()
    print(f"Found {len(video_ids)} unique videos in {args.csv}")

    available, newly = download_batch(video_ids, args.video_dir, retries=args.retries)
    print(f"\nDone. {len(available)} available, {len(newly)} newly downloaded.")


if __name__ == "__main__":
    main()
