"""
build_splits.py — Generate train.csv / test.csv for experiment_01.

Reads the AVA train annotation CSV, filters to videos present in AVA_VIDEO_DIR,
splits by video_id (no row-level leakage), and writes the two split files.

Usage:
    AVA_VIDEO_DIR=/path/to/videos python experiments/dataset/build_splits.py

Optional env vars:
    AVA_CSV      — path to ava_train_v2.2.csv  (default: next to this script)
    OUT_DIR      — directory to write train.csv / test.csv (default: experiment_01/)
    TEST_RATIO   — fraction of videos reserved for test (default: 0.2)
    SEED         — random seed for the split (default: 42)
"""

import os
import random
import sys

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

VIDEO_DIR  = os.environ.get("AVA_VIDEO_DIR")
AVA_CSV    = os.environ.get("AVA_CSV",    os.path.join(SCRIPT_DIR, "data", "ava", "ava_train_v2.2.csv"))
OUT_DIR    = os.environ.get("OUT_DIR",    os.path.join(PROJECT_ROOT, "experiments", "experiment_01"))
TEST_RATIO = float(os.environ.get("TEST_RATIO", "0.2"))
SEED       = int(os.environ.get("SEED", "42"))

# ── validate ──────────────────────────────────────────────────────────────────

if not VIDEO_DIR:
    print("ERROR: AVA_VIDEO_DIR environment variable is not set.", file=sys.stderr)
    sys.exit(1)

if not os.path.isdir(VIDEO_DIR):
    print(f"ERROR: AVA_VIDEO_DIR={VIDEO_DIR!r} is not a directory.", file=sys.stderr)
    sys.exit(1)

if not os.path.isfile(AVA_CSV):
    print(f"ERROR: AVA_CSV={AVA_CSV!r} not found.", file=sys.stderr)
    sys.exit(1)

# ── discover local videos ─────────────────────────────────────────────────────

local_ids = {
    os.path.splitext(f)[0]
    for f in os.listdir(VIDEO_DIR)
    if f.endswith(".mp4")
}

print(f"Videos found in {VIDEO_DIR}: {len(local_ids)}")

# ── load AVA CSV (no header) ──────────────────────────────────────────────────

COL_NAMES = ["video_id", "ts", "x1", "y1", "x2", "y2", "action", "person_id"]
df = pd.read_csv(AVA_CSV, header=None, names=COL_NAMES)
df["ts"] = df["ts"].astype(int)  # strip leading zeros (e.g. 0902 → 902)

ava_ids   = set(df["video_id"].unique())
matched   = local_ids & ava_ids
only_local = local_ids - ava_ids
only_ava   = ava_ids   - local_ids

print(f"Video IDs in AVA CSV:          {len(ava_ids)}")
print(f"Matched (local ∩ AVA):         {len(matched)}")
if only_local:
    print(f"Local but not in AVA (ignored): {sorted(only_local)}")
if only_ava:
    print(f"In AVA but not local (skipped): {len(only_ava)}")

if not matched:
    print("ERROR: No overlap between local videos and AVA CSV.", file=sys.stderr)
    sys.exit(1)

# ── filter & split by video_id ────────────────────────────────────────────────

df_matched = df[df["video_id"].isin(matched)].copy()

video_ids = sorted(matched)
random.seed(SEED)
random.shuffle(video_ids)

n_test  = max(1, round(len(video_ids) * TEST_RATIO))
n_train = len(video_ids) - n_test

test_ids  = set(video_ids[:n_test])
train_ids = set(video_ids[n_test:])

df_train = df_matched[df_matched["video_id"].isin(train_ids)]
df_test  = df_matched[df_matched["video_id"].isin(test_ids)]

# ── write output ──────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
train_path = os.path.join(OUT_DIR, "train.csv")
test_path  = os.path.join(OUT_DIR, "test.csv")

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path,  index=False)

print(f"\nSplit (seed={SEED}, test_ratio={TEST_RATIO}):")
print(f"  train: {len(train_ids)} videos, {len(df_train)} rows → {train_path}")
print(f"  test:  {len(test_ids)}  videos, {len(df_test)}  rows → {test_path}")
