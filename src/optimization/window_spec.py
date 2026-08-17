"""
window_spec.py — Formal walk-forward train/validation/test splits with purge+embargo.

Implements López de Prado's leakage controls (AFML Ch. 7) for the DRB pipeline:

- purge   : any bar whose LABEL window (close_time .. close_time + horizon)
            overlaps the next split is excluded from the training/selection set.
- embargo : additionally exclude bars whose label window ends within ``embargo``
            of the boundary, so short-horizon autocorrelation cannot leak.

Selection rule enforced here: the "best trial" is chosen on the VALIDATION
split only; the TEST split is evaluated once with frozen params and must never
feed back into bounds, trials, or study selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    """Chronological, non-overlapping split boundaries (ISO-8601 strings)."""

    train_start: str
    train_end: str  # exclusive; equals val_start
    val_start: str
    val_end: str  # exclusive; equals test_start
    test_start: str
    test_end: str  # exclusive
    label_horizon: pd.Timedelta
    embargo: pd.Timedelta = pd.Timedelta("1D")

    def boundaries(self) -> dict:
        return {
            "train": (self.train_start, self.train_end),
            "validation": (self.val_start, self.val_end),
            "test": (self.test_start, self.test_end),
            "label_horizon": str(self.label_horizon),
            "embargo": str(self.embargo),
        }


def split_window(
    train_start: str,
    train_end: str,
    val_end: str,
    test_end: str,
    label_horizon: pd.Timedelta,
    embargo: pd.Timedelta = pd.Timedelta("1D"),
) -> WindowSpec:
    """Build a WindowSpec. train_end == val_start and val_end == test_start."""
    return WindowSpec(
        train_start=train_start,
        train_end=train_end,
        val_start=train_end,
        val_end=val_end,
        test_start=val_end,
        test_end=test_end,
        label_horizon=label_horizon,
        embargo=embargo,
    )


def purge_embargo_mask(
    bars: pd.DataFrame,
    boundary: pd.Timestamp,
    label_horizon: pd.Timedelta,
    embargo: pd.Timedelta = pd.Timedelta("1D"),
) -> pd.Series:
    """Boolean mask of rows usable for training when predictions start at ``boundary``.

    A row is purged if its label window reaches into the prediction period
    beyond the embargo: close_time + label_horizon > boundary - embargo.

    Rows whose label entirely precedes ``boundary - embargo`` are kept.
    """
    close_t = pd.to_datetime(bars["close_time"])
    label_end = close_t + label_horizon
    usable = label_end <= boundary - embargo
    return usable


def assign_splits(
    bars: pd.DataFrame,
    spec: WindowSpec,
) -> pd.DataFrame:
    """Add a ``split`` column ('train'|'validation'|'test') with purge applied.

    Training rows are purged so their labels never overlap validation/test;
    validation rows are purged so their labels never overlap test. Test rows
    are never used for selection (enforced by the caller).
    """
    out = bars.copy()
    open_t = pd.to_datetime(out["open_time"])
    t_train = pd.Timestamp(spec.train_end)
    t_test = pd.Timestamp(spec.test_start)

    t_start = pd.Timestamp(spec.train_start)
    t_end = pd.Timestamp(spec.test_end)
    split = pd.Series("outside", index=out.index, dtype="object")
    split[(open_t >= t_start) & (open_t < t_train)] = "train"
    split[(open_t >= t_train) & (open_t < t_test)] = "validation"
    split[(open_t >= t_test) & (open_t < t_end)] = "test"

    # Purge train rows whose label leaks into validation/test.
    train_keep = purge_embargo_mask(out, pd.Timestamp(spec.val_start), spec.label_horizon, spec.embargo)
    # Purge validation rows whose label leaks into test.
    val_keep = purge_embargo_mask(out, t_test, spec.label_horizon, spec.embargo)

    split[(split == "train") & ~train_keep] = "purged"
    split[(split == "validation") & ~val_keep] = "purged"
    out["split"] = split
    return out


def validation_leaderboard(
    trials_meta: list[dict],
    score_key: str = "validation_score",
    stability_key: str | None = "stability",
) -> pd.DataFrame:
    """Rank candidate trials by their VALIDATION score (never in-sample).

    Parameters
    ----------
    trials_meta : list of dicts, one per completed trial, each containing at
        least ``score_key`` plus optional ``train_score`` / ``stability_key``.
        Trials missing a validation score are dropped (they cannot be selected).

    Returns a DataFrame sorted by validation score descending. A note column
    flags trials whose train-vs-validation gap signals overfitting, so the
    caller can prefer a robust (not merely highest) trial.
    """
    if not trials_meta:
        return pd.DataFrame()

    df = pd.DataFrame(trials_meta)
    if score_key not in df.columns:
        raise ValueError(f"trials_meta missing '{score_key}' column")

    df = df.dropna(subset=[score_key])
    df = df.sort_values(score_key, ascending=False).reset_index(drop=True)

    if "train_score" in df.columns:
        df["gap"] = (df.get("train_score", np.nan) - df[score_key]).clip(lower=0)
    else:
        df["gap"] = 0.0
    if stability_key and stability_key in df.columns:
        df["robust"] = (df["gap"] <= df["gap"].quantile(0.5)) & df[stability_key].notna()
    else:
        df["robust"] = df["gap"] <= df["gap"].quantile(0.5)

    return df
