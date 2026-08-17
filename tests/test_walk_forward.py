"""
test_walk_forward.py — Purge/embargo split semantics and validation-based trial
selection (no in-sample cherry-picking).
"""

import numpy as np
import pandas as pd

from src.optimization.window_spec import (
    assign_splits,
    purge_embargo_mask,
    split_window,
    validation_leaderboard,
)


def _bars(start="2024-01-01", n=200, freq="1h"):
    ts = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(
        {
            "open_time": ts,
            "close_time": ts + pd.Timedelta("30min"),
            "close": np.linspace(100, 200, n),
        }
    )


def test_purge_removes_label_overlap():
    bars = _bars()
    horizon = pd.Timedelta("2h")
    # Boundary at bar #100.
    boundary = bars["open_time"].iloc[100]
    mask = purge_embargo_mask(bars, boundary, horizon, pd.Timedelta(0))
    # Bar whose close + horizon crosses the boundary must be purged.
    crossing = bars[mask == False]  # noqa: E712
    assert len(crossing) > 0
    assert (crossing["close_time"] + horizon > boundary).all()
    # Bars well before keep their close+horizon before the boundary.
    assert (bars[mask]["close_time"] + horizon <= boundary).all()


def test_embargo_expands_purge_zone():
    bars = _bars()
    boundary = bars["open_time"].iloc[100]
    no_emb = purge_embargo_mask(bars, boundary, pd.Timedelta("2h"), pd.Timedelta(0)).sum()
    with_emb = purge_embargo_mask(bars, boundary, pd.Timedelta("2h"), pd.Timedelta("4h")).sum()
    assert with_emb < no_emb


def test_assign_splits_with_purge():
    bars = _bars(n=300)
    spec = split_window(
        train_start="2024-01-01",
        train_end="2024-01-05",
        val_end="2024-01-10",
        test_end="2024-01-15",
        label_horizon=pd.Timedelta("2h"),
        embargo=pd.Timedelta("1h"),
    )
    out = assign_splits(bars, spec)
    assert set(out["split"]) <= {"train", "validation", "test", "purged"}
    # Train and validation must be mutually exclusive chronological blocks.
    train_rows = out[out["split"] == "train"]
    val_rows = out[out["split"] == "validation"]
    test_rows = out[out["split"] == "test"]
    assert len(train_rows) > 0 and len(val_rows) > 0 and len(test_rows) > 0
    assert train_rows["open_time"].max() < val_rows["open_time"].min()
    assert val_rows["open_time"].max() < test_rows["open_time"].min()


def test_validation_leaderboard_prefers_validation_not_train():
    trials = [
        {"train_score": 0.99, "validation_score": 0.30},  # overfit, high train
        {"train_score": 0.45, "validation_score": 0.80},  # robust
        {"train_score": 0.90, "validation_score": 0.70},  # middle
    ]
    board = validation_leaderboard(trials)
    assert board.iloc[0]["validation_score"] == 0.80
    assert board.iloc[0]["robust"]


def test_validation_leaderboard_drops_trials_without_validation():
    trials = [
        {"validation_score": 0.60},
        {"train_score": 0.95},  # no validation score -> ineligible
    ]
    board = validation_leaderboard(trials)
    assert len(board) == 1
