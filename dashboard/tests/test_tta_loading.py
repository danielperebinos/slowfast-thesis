"""Tests for merged TTA annotation loading (train + test CSVs).

Verifies that TTAComputer can be built from annotations loaded from
multiple CSV files, matching the logic in ``app._build_tta_computer``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from inference.metrics import TTAComputer  # noqa: E402

# ── Fixtures ─────────────────────────────────────────────────────────────────

_CSV_HEADER = "video_id,ts,x1,y1,x2,y2,action,person_id"


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text(_CSV_HEADER + "\n" + "\n".join(rows) + "\n")
    return path


@pytest.fixture()
def train_csv(tmp_path: Path) -> Path:
    return _write_csv(
        tmp_path / "train.csv",
        [
            "TRAIN_VID_A,902,0.1,0.1,0.5,0.5,12,0",
            "TRAIN_VID_A,903,0.1,0.1,0.5,0.5,12,0",
            "TRAIN_VID_A,904,0.1,0.1,0.5,0.5,80,0",
            "TRAIN_VID_B,902,0.2,0.2,0.6,0.6,17,1",
        ],
    )


@pytest.fixture()
def test_csv(tmp_path: Path) -> Path:
    return _write_csv(
        tmp_path / "test.csv",
        [
            "TEST_VID_X,905,0.3,0.3,0.7,0.7,9,2",
            "TEST_VID_X,906,0.3,0.3,0.7,0.7,9,2",
        ],
    )


# ── Helpers mimicking _build_tta_computer logic ─────────────────────────────


def _load_and_build(csv_paths: tuple[Path, ...], video_stem: str) -> TTAComputer | None:
    """Replicate the CSV-merge + filter logic from app._build_tta_computer."""
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        try:
            frames.append(pd.read_csv(csv_path))
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    subset = df[df["video_id"] == video_stem]
    if subset.empty:
        return None
    return TTAComputer(subset)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_tta_from_train_video(train_csv: Path, test_csv: Path) -> None:
    """TTAComputer is built for a video that exists only in train.csv."""
    computer = _load_and_build((train_csv, test_csv), "TRAIN_VID_A")
    assert computer is not None
    assert len(computer) == 3  # 3 annotation rows for TRAIN_VID_A


def test_tta_from_test_video(train_csv: Path, test_csv: Path) -> None:
    """TTAComputer is built for a video that exists only in test.csv."""
    computer = _load_and_build((train_csv, test_csv), "TEST_VID_X")
    assert computer is not None
    assert len(computer) == 2


def test_tta_unknown_video_returns_none(train_csv: Path, test_csv: Path) -> None:
    """Returns None when the video_id is in neither CSV."""
    computer = _load_and_build((train_csv, test_csv), "UNKNOWN_VID")
    assert computer is None


def test_tta_missing_csv_graceful(train_csv: Path, tmp_path: Path) -> None:
    """If one CSV is missing, the other still provides annotations."""
    missing = tmp_path / "nonexistent.csv"
    computer = _load_and_build((train_csv, missing), "TRAIN_VID_A")
    assert computer is not None
    assert len(computer) == 3


def test_tta_all_csvs_missing_returns_none(tmp_path: Path) -> None:
    """Returns None when no CSVs can be loaded at all."""
    computer = _load_and_build(
        (tmp_path / "a.csv", tmp_path / "b.csv"),
        "ANY_VID",
    )
    assert computer is None
