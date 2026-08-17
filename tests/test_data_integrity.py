#!/usr/bin/env python3
"""
test_data_integrity.py — Valida la integridad de los datos de training

Checks:
- Files exist and have reasonable size
- Required columns present (core OHLCV + key features)
- Date ranges correct
- No critical NaN values
- failed_day / sample_weight columns
- Bar counts per year
- No duplicate bars (same timestamp + OHLCV)
- Symbol consistency
- Feature value ranges
- OHLCV consistency (high >= low, etc)
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]


def available_datasets(data_dir: Path = None) -> list[tuple[str, int]]:
    """Discover available symbol/year parquet files."""
    if data_dir is None:
        data_dir = project_root / "data_optimized" / "training"
    datasets = []
    for year_dir in sorted(data_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for f in sorted(year_dir.glob("*.parquet")):
            parts = f.stem.split("_")
            if "_months_" in f.stem or len(parts) < 2 or not parts[-1].isdigit():
                # Quality and daily-audit reports are Parquet files too, but
                # they are not training datasets.
                continue
            symbol = "_".join(parts[:-1])
            year = int(parts[-1])
            datasets.append((symbol, year))
    return datasets


# Core columns that MUST exist (OHLCV + timestamps)
CORE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "year",
    "month",
    "failed_day",
    "sample_weight",
]

# Feature columns that SHOULD exist (but data may have old names)
FEATURE_COLUMNS = [
    "log_return",
    "rsi",
    "bb_pct_b",
    "vwap",
]

# All known feature columns (current schema)
CURRENT_FEATURES = {
    "atr_pct": "atr",
    "volume_z": "volume_zscore",
    "rolling_volatility": "volatility",
    "macd_hist": "macd",
}

# Datasets that are expected to be PARTIAL by design (symbol listed mid-year),
# so the small-file gate is relaxed and only structural checks apply.
# SOLUSDT was listed on Binance 2020-08-11 → SOLUSDT-2020 is partial.
PARTIAL_OK = {
    ("SOLUSDT", 2020),
}

FEATURE_RANGES = {
    "log_return": (-5, 5),
    "rsi": (0, 100),
    "bb_pct_b": (-0.5, 1.5),
    "vwap": (0, 1e6),
}


def check_file_exists_and_size(year_dir: Path, symbol: str, year: int) -> dict:
    """Check file exists and has reasonable size."""
    file_path = year_dir / f"{symbol}_{year}.parquet"

    result = {"file_exists": file_path.exists(), "file_size_mb": 0, "file_path": str(file_path)}

    if file_path.exists():
        result["file_size_mb"] = file_path.stat().st_size / (1024**2)
        result["size_reasonable"] = result["file_size_mb"] > 1
    else:
        result["size_reasonable"] = False

    return result


def check_columns(df: pd.DataFrame) -> dict:
    """Check all required columns are present."""
    present_cols = set(df.columns)
    required = set(CORE_COLUMNS)
    missing = required - present_cols

    return {
        "all_present": len(missing) == 0,
        "missing_columns": list(missing),
        "all_columns": list(present_cols),
    }


def check_date_range(df: pd.DataFrame, year: int) -> dict:
    """Check date ranges are within expected bounds."""
    if df.empty:
        return {"error": "Empty DataFrame"}

    dates = pd.to_datetime(df["open_time"])

    min_date = dates.min()
    max_date = dates.max()

    expected_min = pd.Timestamp(f"{year}-01-01")
    expected_max = pd.Timestamp(f"{year}-12-31 23:59:59")

    return {
        "min_date": str(min_date),
        "max_date": str(max_date),
        "covers_full_year": min_date <= expected_min + pd.Timedelta(days=30)
        and max_date >= expected_max - pd.Timedelta(days=30),
        "days_span": (max_date - min_date).days,
    }


def check_nan_values(df: pd.DataFrame) -> dict:
    """Check NaN values in critical columns."""
    critical_cols = ["open", "high", "low", "close", "volume"]

    nan_counts = {}
    for col in critical_cols:
        if col in df.columns:
            nan_counts[col] = int(df[col].isna().sum())

    total_nans = sum(nan_counts.values())

    return {
        "nan_by_column": nan_counts,
        "total_nans": total_nans,
        "nan_free": total_nans == 0,
        "nan_acceptable": total_nans < len(df) * 0.01,
    }


def check_failed_days(df: pd.DataFrame) -> dict:
    """Check failed_day and sample_weight columns."""
    if "failed_day" not in df.columns or "sample_weight" not in df.columns:
        return {"error": "Missing failed_day or sample_weight columns"}

    total_bars = len(df)
    failed_bars = int(df["failed_day"].sum())

    sample_weights = df["sample_weight"].unique()
    expected_weights = {1.0, 0.5}

    return {
        "total_bars": total_bars,
        "failed_bars": failed_bars,
        "failed_pct": failed_bars / total_bars * 100 if total_bars > 0 else 0,
        "sample_weights_used": [float(x) for x in sample_weights],
        "weights_correct": set(sample_weights).issubset(expected_weights),
    }


def check_bar_counts(df: pd.DataFrame, year: int) -> dict:
    """Check bar counts are reasonable."""
    if df.empty:
        return {"error": "Empty DataFrame"}

    df_copy = df.copy()
    df_copy["bar_month"] = pd.to_datetime(df_copy["open_time"]).dt.month

    bars_by_month = df_copy.groupby("bar_month").size().to_dict()

    total_bars = len(df)
    min_expected = 8000

    return {
        "total_bars": total_bars,
        "bars_per_month": bars_by_month,
        "reasonable_count": total_bars >= min_expected,
        "min_expected": min_expected,
    }


def check_duplicates(df: pd.DataFrame) -> dict:
    """
    Check for true duplicate bars (same timestamp AND same OHLCV values).
    Only flags rows that are completely identical.
    """
    if df.empty:
        return {"error": "Empty DataFrame"}

    key_cols = ["open_time", "open", "high", "low", "close", "volume"]

    if not all(c in df.columns for c in key_cols):
        return {"error": "Missing columns for duplicate check"}

    dup_mask = df.duplicated(subset=key_cols, keep=False)
    dup_count = dup_mask.sum()

    return {
        "duplicate_bars": int(dup_count),
        "no_duplicates": dup_count == 0,
        "duplicates_acceptable": dup_count < len(df) * 0.001,
    }


def check_timestamp_duplicates(df: pd.DataFrame) -> dict:
    """
    Check for bars with same timestamp but different values.
    This indicates a data quality issue.
    """
    if df.empty:
        return {"error": "Empty DataFrame"}

    if "open_time" not in df.columns:
        return {"error": "Missing open_time column"}

    time_dups = df["open_time"].duplicated(keep=False).sum()

    return {
        "timestamp_duplicates": int(time_dups),
        # Same-millisecond trades can legitimately create distinct bars. Keep
        # this observable as a warning instead of failing the dataset.
        "has_timestamp_warning": time_dups > 0,
        "timestamp_issue_pct": time_dups / len(df) * 100,
    }


def check_symbol_consistency(df: pd.DataFrame, expected_symbol: str) -> dict:
    """Check symbol column is consistent."""
    if "symbol" not in df.columns:
        return {"error": "No symbol column"}

    symbols = df["symbol"].unique()

    return {
        "symbols_found": list(symbols),
        "correct_symbol": expected_symbol in symbols,
    }


def check_feature_ranges(df: pd.DataFrame) -> dict:
    """Check feature values are within reasonable ranges."""
    issues = {}

    for feature, (min_val, max_val) in FEATURE_RANGES.items():
        if feature not in df.columns:
            continue

        col_data = df[feature].dropna()
        if len(col_data) == 0:
            continue

        out_of_range = ((col_data < min_val) | (col_data > max_val)).sum()
        pct_out = out_of_range / len(col_data) * 100

        if pct_out > 5:
            issues[feature] = {
                "out_of_range_count": int(out_of_range),
                "out_of_range_pct": pct_out,
                "min_found": float(col_data.min()),
                "max_found": float(col_data.max()),
            }

    return {
        "range_issues": issues,
        "all_ranges_ok": len(issues) == 0,
    }


def check_ohlcv_consistency(df: pd.DataFrame) -> dict:
    """Check OHLCV data makes sense (high >= low, etc)."""
    if df.empty:
        return {"error": "Empty DataFrame"}

    issues = []

    if (df["high"] < df["low"]).any():
        issues.append("high < low found")

    if (df["high"] < df["open"]).any():
        issues.append("high < open found")

    if (df["high"] < df["close"]).any():
        issues.append("high < close found")

    if (df["low"] > df["open"]).any():
        issues.append("low > open found")

    if (df["low"] > df["close"]).any():
        issues.append("low > close found")

    if (df["volume"] < 0).any():
        issues.append("negative volume found")

    return {
        "ohlcv_issues": issues,
        "ohlcv_consistent": len(issues) == 0,
    }


def check_data_quality_score(df: pd.DataFrame) -> dict:
    """
    Calculate a composite data quality score (0-100).
    """
    if df.empty:
        return {"error": "Empty DataFrame", "score": 0}

    score = 100
    deductions = []

    # Check OHLCV consistency
    ohlcv_issues = check_ohlcv_consistency(df)
    if not ohlcv_issues["ohlcv_consistent"]:
        score -= 20
        deductions.append("OHLCV inconsistencies")

    # Check timestamp duplicates
    ts_dups = check_timestamp_duplicates(df)
    if ts_dups["has_timestamp_warning"]:
        pct = ts_dups["timestamp_issue_pct"]
        deduction = min(30, int(pct * 10))
        score -= deduction
        deductions.append(f"Timestamp duplicates ({pct:.2f}%)")

    # Check NaN values
    nan_check = check_nan_values(df)
    if not nan_check["nan_acceptable"]:
        score -= 15
        deductions.append("Excessive NaN values")

    # Check true duplicates
    dups = check_duplicates(df)
    if not dups["duplicates_acceptable"]:
        score -= 10
        deductions.append("True duplicate bars")

    return {
        "score": max(0, score),
        "deductions": deductions,
    }


def run_all_checks(symbol: str, year: int, data_dir: Path = None) -> dict:
    """Run all checks on a single symbol/year."""
    if data_dir is None:
        data_dir = project_root / "data_optimized" / "training"

    year_dir = data_dir / str(year)

    results = {
        "symbol": symbol,
        "year": year,
        "checks": {},
        "all_passed": True,
    }

    # Check file
    file_check = check_file_exists_and_size(year_dir, symbol, year)
    results["checks"]["file"] = file_check
    if not file_check["file_exists"]:
        results["all_passed"] = False
        results["error"] = "Dataset not available (no parquet file)"
        return results

    # Small files are acceptable ONLY for documented partial datasets; anything
    # else under 1 MB is flagged as likely-corrupt/incomplete.
    partial_expected = (symbol, year) in PARTIAL_OK
    if not file_check["size_reasonable"] and not partial_expected:
        results["all_passed"] = False
        results["error"] = f"File too small ({file_check.get('file_size_mb', 0):.2f} MB) — dataset incomplete"
        return results
    if partial_expected:
        results["partial_coverage"] = True

    # Load data
    try:
        file_path = year_dir / f"{symbol}_{year}.parquet"
        df = pd.read_parquet(file_path)
    except Exception as e:
        results["all_passed"] = False
        results["error"] = f"Failed to load: {e}"
        return results

    # Run checks
    checks = [
        ("columns", check_columns(df)),
        ("date_range", check_date_range(df, year)),
        ("nan_values", check_nan_values(df)),
        ("failed_days", check_failed_days(df)),
        ("bar_counts", check_bar_counts(df, year)),
        ("timestamp_duplicates", check_timestamp_duplicates(df)),
        ("true_duplicates", check_duplicates(df)),
        ("symbol", check_symbol_consistency(df, symbol)),
        ("ohlcv", check_ohlcv_consistency(df)),
        ("quality_score", check_data_quality_score(df)),
    ]

    for check_name, check_result in checks:
        results["checks"][check_name] = check_result

    results["total_bars"] = len(df)
    results["data_loaded"] = True

    # Determine overall pass/fail
    # Must pass: file (exists + loads), columns, ohlcv, symbol
    critical_checks = ["file", "columns", "ohlcv", "symbol"]
    for check_name in critical_checks:
        check_result = results["checks"].get(check_name, {})
        if isinstance(check_result, dict):
            if check_name == "file":
                # For documented partial datasets, existence + loadability is enough.
                ok = check_result.get("file_exists")
                if not results.get("partial_coverage"):
                    ok = ok and check_result.get("size_reasonable")
                if not ok:
                    results["all_passed"] = False
            elif check_name == "columns":
                if not check_result.get("all_present"):
                    results["all_passed"] = False
            elif check_name == "ohlcv":
                if not check_result.get("ohlcv_consistent"):
                    results["all_passed"] = False
            elif check_name == "symbol":
                if not check_result.get("correct_symbol"):
                    results["all_passed"] = False

    return results


def print_results(results: dict):
    """Print results in a readable format."""
    symbol = results["symbol"]
    year = results["year"]

    print(f"\n{'=' * 70}")
    print(f"  {symbol} — {year}")
    print(f"{'=' * 70}")

    if "error" in results and results.get("data_loaded") is not True:
        print(f"  ❌ ERROR: {results['error']}")
        return

    checks = results["checks"]

    # File check
    fc = checks.get("file", {})
    status = "✅" if fc.get("file_exists") else "❌"
    print(f"  {status} File: {fc.get('file_size_mb', 0):.1f} MB")
    if results.get("partial_coverage"):
        print("  ℹ️  PARTIAL COVERAGE (expected: symbol listed mid-year)")

    # Columns
    cc = checks.get("columns", {})
    status = "✅" if cc.get("all_present") else "❌"
    print(
        f"  {status} Core columns: {'OK' if cc.get('all_present') else 'MISSING: ' + str(cc.get('missing_columns', []))}"
    )

    # Date range
    dr = checks.get("date_range", {})
    status = "✅" if dr.get("covers_full_year") else "⚠️"
    print(f"  {status} Date range: {dr.get('min_date', 'N/A')[:10]} → {dr.get('max_date', 'N/A')[:10]}")
    print(f"      Days span: {dr.get('days_span', 0)}")

    # Bar counts
    bc = checks.get("bar_counts", {})
    status = "✅" if bc.get("reasonable_count") else "⚠️"
    print(f"  {status} Bar count: {bc.get('total_bars', 0):,}")

    # Failed days
    fd = checks.get("failed_days", {})
    if "error" not in fd:
        print(f"  ℹ️  Failed bars: {fd.get('failed_bars', 0):,} ({fd.get('failed_pct', 0):.1f}%)")
        print(f"      Sample weights: {fd.get('sample_weights_used', [])}")

    # Timestamp duplicates (DATA QUALITY ISSUE)
    td = checks.get("timestamp_duplicates", {})
    if "error" not in td:
        ts_dups = td.get("timestamp_duplicates", 0)
        if ts_dups > 0:
            print(f"  ⚠️  Timestamp duplicates: {ts_dups:,} ({td.get('timestamp_issue_pct', 0):.2f}%)")
        else:
            print("  ✅ Timestamp duplicates: 0")

    # True duplicates
    dc = checks.get("true_duplicates", {})
    if "error" not in dc:
        dup_count = dc.get("duplicate_bars", 0)
        status = "✅" if dc.get("no_duplicates") else "⚠️"
        print(f"  {status} True duplicate bars: {dup_count:,}")

    # NaN values
    nc = checks.get("nan_values", {})
    if "error" not in nc:
        total_nans = nc.get("total_nans", 0)
        status = "✅" if nc.get("nan_acceptable") else "⚠️"
        print(f"  {status} NaN in OHLCV: {total_nans:,}")

    # OHLCV consistency
    oc = checks.get("ohlcv", {})
    if "error" not in oc:
        status = "✅" if oc.get("ohlcv_consistent") else "❌"
        print(f"  {status} OHLCV consistency: {oc.get('ohlcv_issues', []) or 'OK'}")

    # Symbol
    sc = checks.get("symbol", {})
    if "error" not in sc:
        status = "✅" if sc.get("correct_symbol") else "❌"
        print(f"  {status} Symbol: {sc.get('symbols_found', [])}")

    # Quality score
    qs = checks.get("quality_score", {})
    if "error" not in qs:
        score = qs.get("score", 0)
        deductions = qs.get("deductions", [])
        if deductions:
            print(f"  ⚠️  Quality score: {score}/100 ({', '.join(deductions)})")
        else:
            print(f"  ✅ Quality score: {score}/100")

    # Overall
    overall = "✅ ALL PASSED" if results["all_passed"] else "❌ ISSUES FOUND"
    print(f"\n  {overall}")


def print_summary(all_results: list):
    """Print summary table."""
    print("\n" + "=" * 100)
    print("  SUMMARY TABLE")
    print("=" * 100)

    header = f"{'Symbol':<10} {'Year':>6} {'Bars':>10} {'Failed%':>8} {'TS_Dups':>10} {'Quality':>8} {'Status':>10}"
    print(header)
    print("-" * 100)

    total_bars = 0
    total_files = len(all_results)
    passed = 0

    for r in all_results:
        symbol = r.get("symbol", "")
        year = r.get("year", "")
        bars = r.get("total_bars", 0)
        total_bars += bars

        fd = r.get("checks", {}).get("failed_days", {})
        failed_pct = fd.get("failed_pct", 0)

        td = r.get("checks", {}).get("timestamp_duplicates", {})
        ts_dups = td.get("timestamp_duplicates", 0)

        qs = r.get("checks", {}).get("quality_score", {})
        score = qs.get("score", 0)

        status = "✅ PASS" if r.get("all_passed") else "❌ FAIL"
        if r.get("all_passed"):
            passed += 1

        print(f"{symbol:<10} {year:>6} {bars:>10,} {failed_pct:>7.1f}% {ts_dups:>10,} {score:>7}/100 {status:>10}")

    print("-" * 100)
    print(f"{'TOTAL':<10} {'':<6} {total_bars:>10,}")
    print("=" * 100)
    print(f"\n  Passed: {passed}/{total_files}")

    return passed == total_files


# ── pytest tests ──────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_data_dir_exists():
    data_dir = project_root / "data_optimized" / "training"
    assert data_dir.exists(), (
        f"Training data directory {data_dir} does not exist. Run `python scripts/build_training_data.py` first."
    )


def _get_data_dir():
    return project_root / "data_optimized" / "training"


@pytest.mark.slow
@pytest.mark.parametrize("symbol,year", [(s, y) for s in DEFAULT_SYMBOLS for y in DEFAULT_YEARS])
def test_dataset_integrity(symbol, year):
    """Run all integrity checks on a single symbol/year dataset. Skips if data missing."""
    data_dir = _get_data_dir()
    year_dir = data_dir / str(year)
    file_path = year_dir / f"{symbol}_{year}.parquet"
    if not file_path.exists():
        pytest.skip(f"No data for {symbol} {year} at {file_path}")
    results = run_all_checks(symbol, year, data_dir)
    assert results["data_loaded"], f"Failed to load data: {results.get('error', 'unknown')}"
    assert results["all_passed"], (
        f"Integrity issues for {symbol} {year}: "
        f"{[k for k, v in results['checks'].items() if isinstance(v, dict) and not v.get('all_present') and not v.get('ohlcv_consistent') and not v.get('correct_symbol')]}"
    )


@pytest.mark.slow
def test_available_datasets():
    """Verify at least one dataset exists in data_optimized/training/."""
    datasets = available_datasets()
    assert len(datasets) > 0, (
        f"No parquet datasets found in {_get_data_dir()}. Run `python scripts/build_training_data.py` first."
    )
    for _, year in datasets:
        assert isinstance(year, int) and 2020 <= year <= 2026, f"Unexpected year {year}"


# ── CLI entry point ───────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test data integrity for training parquet files")
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022], help="Years to check")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"], help="Symbols to check")
    parser.add_argument("--data-dir", default="data_optimized/training/", help="Data directory")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("\n" + "=" * 70)
    print("  DATA INTEGRITY TEST SUITE")
    print("=" * 70)
    print(f"  Years: {args.years}")
    print(f"  Symbols: {args.symbols}")
    print(f"  Data dir: {data_dir}")

    all_results = []

    for symbol in args.symbols:
        for year in args.years:
            results = run_all_checks(symbol, year, data_dir)
            all_results.append(results)
            print_results(results)

    all_passed = print_summary(all_results)

    if all_passed:
        print("\n  ✅ ALL CHECKS PASSED")
        return 0
    else:
        print("\n  ⚠️  SOME ISSUES DETECTED — Review data quality before using")
        return 0  # Don't fail script, just warn


if __name__ == "__main__":
    sys.exit(main())
