"""Quality gates and monthly audit reports for training datasets."""

from __future__ import annotations

from calendar import monthrange
from datetime import date

import numpy as np
import pandas as pd

from src.bars.bars_statistics import BarsStatistics


RECOVERY_SOURCE_PREFIX = "failed_days_"
RECOVERY_STATUSES = {
    "normal",
    "recovered",
    "failed_irrecoverable",
    "missing_ticks",
    "excluded_partial",
}


def daily_audit_frame(audits: list[dict] | None) -> pd.DataFrame:
    """Normalize per-day tick/bar audit records for durable persistence."""
    if not audits:
        return pd.DataFrame(
            columns=[
                "date", "source", "ticks_received", "ticks_invalid",
                "ticks_mad_removed", "ticks_clean", "ticks_consumed",
                "ticks_residual", "bars", "mad_removed_pct", "errors",
            ]
        )
    frame = pd.DataFrame(audits).copy()
    for col in ("ticks_received", "ticks_invalid", "ticks_mad_removed", "ticks_clean", "ticks_consumed", "ticks_residual", "bars"):
        if col not in frame:
            frame[col] = 0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype("int64")
    frame["mad_removed_pct"] = np.where(
        frame["ticks_received"] > 0,
        100.0 * frame["ticks_mad_removed"] / frame["ticks_received"],
        0.0,
    )
    if "errors" not in frame:
        frame["errors"] = ""
    return frame.sort_values("date").reset_index(drop=True)


def validate_daily_audit(
    audit: pd.DataFrame,
    *,
    max_mad_removed_pct: float = 5.0,
) -> dict:
    """Validate tick accounting and flag suspicious daily MAD removal."""
    errors: list[str] = []
    if audit is None or audit.empty:
        return {"valid": False, "errors": ["daily audit is empty"]}
    suspicious = audit[audit["mad_removed_pct"] > max_mad_removed_pct]
    if not suspicious.empty:
        errors.append(
            f"{len(suspicious)} days exceed MAD removal limit "
            f"{max_mad_removed_pct:.1f}%"
        )
    if audit["ticks_residual"].lt(0).any():
        errors.append("negative residual tick count")
    if audit["bars"].lt(0).any():
        errors.append("negative bar count")
    return {
        "valid": not errors,
        "errors": errors,
        "days": int(len(audit)),
        "max_mad_removed_pct": float(audit["mad_removed_pct"].max()),
    }


def is_partial_month(year: int, month: int, today: date | None = None) -> bool:
    """Return whether the calendar month is the currently incomplete month."""
    today = today or date.today()
    return year == today.year and month == today.month


def _recovery_mask(df: pd.DataFrame) -> pd.Series:
    if "recovered_day" in df.columns:
        return df["recovered_day"].astype("boolean").fillna(False).astype(bool)
    if "study_source" in df.columns:
        return df["study_source"].astype(str).str.startswith(RECOVERY_SOURCE_PREFIX)
    return pd.Series(False, index=df.index)


def monthly_quality_report(
    df: pd.DataFrame,
    *,
    expected_days: dict[int, int] | None = None,
    include_quality_score: bool = True,
) -> pd.DataFrame:
    """Build one auditable quality row per month in ``df``.

    The report intentionally distinguishes bars created by recovery from days
    that remain irrecoverable. Missing calendar days are not silently counted
    as successful days.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    frame = df.copy()
    times = pd.to_datetime(frame["open_time"], errors="coerce")
    frame["_month"] = times.dt.month.astype("Int64")
    frame["_day"] = times.dt.strftime("%Y-%m-%d")
    frame["_recovered"] = _recovery_mask(frame).to_numpy()
    if "failed_day" in frame.columns:
        frame["_failed"] = frame["failed_day"].astype(bool)
    else:
        frame["_failed"] = False

    rows: list[dict] = []
    for month, group in frame.groupby("_month", dropna=True, sort=True):
        month = int(month)
        day_counts = group.groupby("_day", dropna=True).size()
        recovery_days = int(group.loc[group["_recovered"], "_day"].nunique())
        failed_days = int(group.loc[group["_failed"], "_day"].nunique())
        bars = len(group)
        quality_score = np.nan
        if include_quality_score and bars >= 100:
            try:
                quality_score = float(BarsStatistics().compute_quality_score_fast(group))
            except Exception:
                quality_score = np.nan

        expected = (expected_days or {}).get(month, monthrange(int(times.dt.year.iloc[0]), month)[1])
        rows.append(
            {
                "month": month,
                "partial_month": is_partial_month(int(times.dt.year.iloc[0]), month),
                "bars": bars,
                "observed_days": int(group["_day"].nunique()),
                "expected_days": int(expected),
                "coverage_pct": 100.0 * group["_day"].nunique() / max(expected, 1),
                "normal_bars": int((~group["_recovered"]).sum()),
                "recovered_bars": int(group["_recovered"].sum()),
                "recovery_pct": 100.0 * group["_recovered"].mean(),
                "recovery_days": recovery_days,
                "recovery_day_pct": 100.0 * recovery_days / max(int(group["_day"].nunique()), 1),
                "failed_days": failed_days,
                "mean_bars_per_day": float(day_counts.mean()) if len(day_counts) else 0.0,
                "median_bars_per_day": float(day_counts.median()) if len(day_counts) else 0.0,
                "p10_bars_per_day": float(day_counts.quantile(0.10)) if len(day_counts) else 0.0,
                "p90_bars_per_day": float(day_counts.quantile(0.90)) if len(day_counts) else 0.0,
                "quality_score": quality_score,
            }
        )
    return pd.DataFrame(rows)


def validate_monthly_quality(
    report: pd.DataFrame,
    *,
    max_recovery_pct: float = 50.0,
    min_mean_bars_per_day: float = 10.0,
    min_quality_score: float = 20.0,
    allow_partial: bool = True,
) -> dict:
    """Return a publication gate report for monthly quality metrics."""
    errors: list[str] = []
    if report is None or report.empty:
        return {"valid": False, "errors": ["monthly quality report is empty"]}

    for row in report.itertuples(index=False):
        prefix = f"month {int(row.month):02d}"
        partial = allow_partial and bool(row.partial_month)
        if row.recovery_pct > max_recovery_pct and not partial:
            errors.append(
                f"{prefix}: recovered bars {row.recovery_pct:.1f}% > {max_recovery_pct:.1f}%"
            )
        if row.recovery_day_pct > max_recovery_pct and not partial:
            errors.append(
                f"{prefix}: recovered days {row.recovery_day_pct:.1f}% > {max_recovery_pct:.1f}%"
            )
        if row.mean_bars_per_day < min_mean_bars_per_day and not partial:
            errors.append(
                f"{prefix}: mean bars/day {row.mean_bars_per_day:.1f} < {min_mean_bars_per_day:.1f}"
            )
        if np.isfinite(row.quality_score) and row.quality_score < min_quality_score and not partial:
            errors.append(f"{prefix}: quality score {row.quality_score:.1f} < {min_quality_score:.1f}")
    return {"valid": not errors, "errors": errors, "months": len(report)}


def validate_study_metadata(
    *,
    completion_rate: float,
    failed_days: list[str],
    min_completion_rate: float = 0.80,
    max_failed_day_ratio: float = 0.50,
    expected_days: int | None = None,
) -> dict:
    """Validate study metadata without treating recovery as success."""
    errors: list[str] = []
    if not 0.0 <= float(completion_rate) <= 1.0:
        errors.append(f"completion_rate outside [0, 1]: {completion_rate}")
    if completion_rate < min_completion_rate:
        errors.append(f"completion_rate {completion_rate:.3f} < {min_completion_rate:.3f}")
    if expected_days:
        ratio = len(set(failed_days)) / expected_days
        if ratio > max_failed_day_ratio:
            errors.append(f"failed-day ratio {ratio:.3f} > {max_failed_day_ratio:.3f}")
    return {"valid": not errors, "errors": errors}
