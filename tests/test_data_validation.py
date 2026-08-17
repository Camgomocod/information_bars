import numpy as np
import pandas as pd
import pytest

from scripts.build_yearly_training_data import _merge_recovered_day_bars, _recovered_bar_mask
from src.features.base_features import compute_all_features
from src.pipeline import bar_builder
from src.pipeline.data_validation import (
    TrainingDataValidationError,
    validate_training_frame,
    write_validated_parquet,
)
from src.pipeline.quality_report import (
    daily_audit_frame,
    monthly_quality_report,
    validate_daily_audit,
    validate_monthly_quality,
)


def _feature_frame(n: int = 500) -> pd.DataFrame:
    t = pd.date_range("2026-01-01", periods=n, freq="min")
    close = 2000 + np.cumsum(np.sin(np.arange(n) / 7.0) * 2.0 + 0.2)
    bars = pd.DataFrame(
        {
            "open_time": t,
            "close_time": t + pd.Timedelta(minutes=1),
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "n_ticks": np.arange(100, 100 + n),
            "volume": np.arange(1, n + 1, dtype=float),
            "dollar_value": close * np.arange(1, n + 1, dtype=float),
        }
    )
    return compute_all_features(bars, drop_warmup=True, winsorize=False).assign(
        symbol="ETHUSDT",
        year=2026,
        month=1,
        study_source="test",
        completion_rate=1.0,
        failed_day=0,
        sample_weight=1.0,
        exp_lambda=0.99,
        init_exp_T=1000,
        funding_rate_mean=0.0,
    )


def test_validation_rejects_constant_features():
    df = _feature_frame()
    df["rsi"] = 1.0
    report = validate_training_frame(df, symbol="ETHUSDT", year=2026)
    assert not report["valid"]
    assert "constant bar-level features: ['rsi']" in report["errors"]


def test_validation_accepts_partial_dataset_with_varying_features():
    report = validate_training_frame(_feature_frame(), symbol="ETHUSDT", year=2026)
    assert report["valid"], report["errors"]


def test_write_validated_parquet_round_trip(tmp_path):
    path = tmp_path / "ETHUSDT_2026.parquet"
    df = _feature_frame()
    write_validated_parquet(df, path, symbol="ETHUSDT", year=2026)
    loaded = pd.read_parquet(path)
    assert len(loaded) == len(df)


def test_write_validated_parquet_does_not_replace_invalid_file(tmp_path):
    path = tmp_path / "ETHUSDT_2026.parquet"
    path.write_bytes(b"previous")
    df = _feature_frame()
    df["vwap"] = 1.0
    with pytest.raises(TrainingDataValidationError):
        write_validated_parquet(df, path, symbol="ETHUSDT", year=2026)
    assert path.read_bytes() == b"previous"


def test_recovered_mask_treats_missing_marker_as_normal():
    df = pd.DataFrame({"_recovered_bar": [np.nan, True, False]})
    assert _recovered_bar_mask(df).tolist() == [False, True, False]


def test_validation_rejects_missing_positioning_value():
    df = _feature_frame()
    df.loc[df.index[0], "funding_rate_mean"] = np.nan
    report = validate_training_frame(df, symbol="ETHUSDT", year=2026)
    assert not report["valid"]
    assert "funding_rate_mean contains NaN values" in report["errors"]


def test_monthly_report_separates_recovered_bars():
    df = _feature_frame(500)
    df["recovered_day"] = 0
    df.loc[df.index[-100:], "recovered_day"] = 1
    df["failed_day"] = 0
    report = monthly_quality_report(df, include_quality_score=False)
    row = report.iloc[0]
    assert row["normal_bars"] + row["recovered_bars"] == len(df)
    assert row["recovered_bars"] == 100
    assert row["recovery_pct"] == pytest.approx(100.0 * 100 / len(df))
    assert row["recovery_days"] == 1


def test_monthly_quality_gate_rejects_recovery_concentration():
    report = pd.DataFrame(
        [
            {
                "month": 1,
                "partial_month": False,
                "recovery_pct": 75.0,
                "recovery_day_pct": 75.0,
                "mean_bars_per_day": 30.0,
                "quality_score": 50.0,
            }
        ]
    )
    result = validate_monthly_quality(report, max_recovery_pct=50.0)
    assert not result["valid"]
    assert "recovered bars 75.0%" in result["errors"][0]


def test_recovery_merge_replaces_only_recovered_days():
    base = pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "close": [1.0, 2.0, 3.0],
            "_bar_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "_recovered_bar": False,
        }
    )
    recovered = pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2026-01-02"]),
            "close": [20.0],
        }
    )
    merged = _merge_recovered_day_bars(base, recovered, {"2026-01-02"})
    assert merged["close"].tolist() == [1.0, 3.0, 20.0]
    assert merged["_recovered_bar"].tolist() == [False, False, True]


def test_daily_builder_retains_day_provenance(monkeypatch):
    class FakeDownloader:
        def __init__(self, **kwargs):
            pass

        def download_day(self, date_str):
            if date_str not in {"2026-01-01", "2026-01-02"}:
                return None
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range(date_str, periods=3, freq="s"),
                    "price": [100.0, 101.0, 100.5],
                    "quantity": [1.0, 1.0, 1.0],
                }
            )

    class FakeNormalizer:
        def clean_raw_ticks(self, frame, **kwargs):
            return frame

    class FakeBars:
        def __init__(self, **kwargs):
            pass

        def get_drbs_audited(self, df, **kwargs):
            t = pd.to_datetime(df["timestamp"].iloc[0])
            bars = pd.DataFrame(
                {
                    "open_time": [t, t + pd.Timedelta(seconds=1)],
                    "close_time": [t + pd.Timedelta(seconds=1), t + pd.Timedelta(seconds=2)],
                    "open": [100.0, 100.0],
                    "high": [101.0, 101.0],
                    "low": [99.0, 99.0],
                    "close": [100.0, 100.0],
                    "n_ticks": [10, 10],
                    "volume": [1.0, 1.0],
                    "dollar_value": [100.0, 100.0],
                }
            )

            class Audit:
                consumed_ticks = 3
                residual_ticks = 0
                violations = []

            return bars, Audit()

    monkeypatch.setattr(bar_builder, "DownloadData", FakeDownloader)
    monkeypatch.setattr(bar_builder, "DataNormalizer", FakeNormalizer)
    monkeypatch.setattr(bar_builder, "OptimizedInfoRunBars", FakeBars)
    result = bar_builder.build_monthly_daily_bars(
        "ETHUSDT",
        2026,
        1,
        {"exp_lambda": 0.99, "init_exp_T": 100},
        min_days=1,
        min_bars=2,
        verbose=False,
    )
    assert result["_bar_date"].unique().tolist() == ["2026-01-01", "2026-01-02"]
    assert result["_base_failed_day"].eq(0).all()
    assert result["_recovered_bar"].eq(False).all()


def test_daily_audit_reports_mad_removal_and_gate():
    audit = daily_audit_frame(
        [
            {
                "date": "2026-01-01",
                "source": "base",
                "ticks_received": 1000,
                "ticks_invalid": 10,
                "ticks_mad_removed": 20,
                "ticks_clean": 970,
                "ticks_consumed": 900,
                "ticks_residual": 70,
                "bars": 10,
                "errors": "",
            }
        ]
    )
    assert audit.loc[0, "mad_removed_pct"] == pytest.approx(2.0)
    assert validate_daily_audit(audit)["valid"]
    audit.loc[0, "mad_removed_pct"] = 7.0
    assert not validate_daily_audit(audit, max_mad_removed_pct=5.0)["valid"]
