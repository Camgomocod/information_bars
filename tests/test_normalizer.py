"""
test_normalizer.py — Tick normalization: UTC semantics, dedup, causal MAD,
auditable counts, and no-silent-loss reconciliation.
"""

import numpy as np
import pandas as pd

from src.normalizers.data_normalizer import DataNormalizer
from src.pipeline.temporal_contract import TickAudit, normalize_ticks, reconcile_ticks_to_bars


def _raw_ticks(n: int = 2000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 50000.0
    ts = pd.date_range("2024-01-01", periods=n, freq="ms")
    price = base + np.cumsum(rng.normal(0, 2, n))
    price = np.maximum(price, 1000)
    return pd.DataFrame(
        {
            "id": np.arange(1000000, 1000000 + n),
            "timestamp": ts,
            "price": price,
            "quantity": rng.exponential(0.1, n),
            "is_buyer_maker": rng.choice([True, False], n),
        }
    )


def test_normalize_ticks_utc_and_sort():
    df = _raw_ticks(500)
    clean, audit = normalize_ticks(df)
    assert audit.ticks_received == len(df)
    assert clean["timestamp"].is_monotonic_increasing
    # All timestamps are tz-naive (documented UTC contract).
    assert clean["timestamp"].dt.tz is None


def test_normalize_ticks_dedups_exact_duplicates():
    df = _raw_ticks(500)
    dup = df.iloc[[100]].copy()
    df = pd.concat([df, dup, dup], ignore_index=True)
    clean, audit = normalize_ticks(df, dedup="exact")
    assert audit.ticks_duplicates == 2
    assert len(clean) == len(df) - 2


def test_normalize_ticks_sanity_filter():
    df = _raw_ticks(500)
    df.loc[df.index[5], "price"] = -5.0
    df.loc[df.index[6], "quantity"] = 0.0
    clean, audit = normalize_ticks(df)
    assert audit.ticks_invalid == 2
    assert not (clean["price"] <= 0).any()
    assert not (clean["quantity"] <= 0).any()


def test_normalize_ticks_dollar_value_imputed():
    df = _raw_ticks(200).drop(columns=["is_buyer_maker"])
    clean, _ = normalize_ticks(df)
    assert "dollar_value" in clean.columns
    np.testing.assert_allclose(
        clean["dollar_value"].to_numpy()[:50],
        (clean["price"] * clean["quantity"]).to_numpy()[:50],
        rtol=1e-4,
    )


def test_audit_reconciles_counts():
    df = _raw_ticks(1000)
    df.loc[df.index[0], "price"] = -1.0
    clean, audit = normalize_ticks(df)
    assert audit.reconcile(len(clean))


def test_reconcile_ticks_to_bars():
    # 5000 clean ticks → 4900 consumed + 100 residual = no silent loss.
    assert reconcile_ticks_to_bars(5000, 4900, 100)
    assert not reconcile_ticks_to_bars(5000, 4900, 50)


def test_clean_raw_ticks_is_causal():
    """The MAD filter must NOT use future rows to decide on past rows.

    Feed a low-noise series followed by a huge spike, then back to normal.
    With causal (ffill-only) MAD, the ticks BEFORE the spike are never flagged
    because of the future spike, and the spike itself is removed.
    """
    rng = np.random.default_rng(0)
    n_flat = 500
    base = 100.0 + rng.normal(0, 0.1, n_flat)
    series = np.concatenate([base, [base[-1], base[-1] + 0.01], [100000.0], [base[-1] + 0.02]])
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(series), freq="ms"),
            "price": series,
            "quantity": np.ones(len(series)),
        }
    )
    clean = DataNormalizer.clean_raw_ticks(df, mad_window=50, k=5.0)
    # The ticks right before the spike are normal — must survive.
    assert clean["price"].iloc[-4] < 110.0
    assert clean["price"].iloc[-3] < 110.0
    # The spike itself (MAD z >> k against past level) is removed.
    assert 100000.0 not in clean["price"].values


def test_clean_raw_ticks_tracks_audit_counts():
    df = _raw_ticks(1000)
    audit = TickAudit()
    clean = DataNormalizer.clean_raw_ticks(df, mad_window=50, k=5.0, audit=audit)
    assert audit.ticks_mad_removed == len(df) - len(clean)
    # Naive contract check: received == mad_removed + cleaned (no invalids in this input).
    assert audit.ticks_mad_removed + len(clean) == len(df)
