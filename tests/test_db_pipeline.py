#!/usr/bin/env python3
"""
test_db_pipeline.py — Tests rápidos para verificar TimescaleDB integration

Uso:
    micromamba run -n trading-core python tests/test_db_pipeline.py

Tests:
    1. DB connectivity
    2. DBReader.get_params() replaces HyperparamLoader
    3. DBReader.get_bars() returns correct data
    4. TimescaleDBClient.write_bars() inserts correctly
    5. End-to-end: write + read roundtrip
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage.db_reader import DBReader
from src.storage.timescale_client import TimescaleDBClient

pytestmark = pytest.mark.db


# ─────────────────────────────────────────
# Test 1: Conexión
# ─────────────────────────────────────────
def test_connection():
    print("\n[TEST 1] Conexión a TimescaleDB")
    reader = DBReader()
    result = reader._get_conn().cursor()
    result.execute("SELECT version()")
    version = result.fetchone()[0]
    result.close()
    reader.close()
    assert "PostgreSQL" in version
    print(f"  ✅ {version[:50]}")


# ─────────────────────────────────────────
# Test 2: DBReader.get_params() vs HyperparamLoader
# ─────────────────────────────────────────
def test_get_params():
    print("\n[TEST 2] DBReader.get_params() (reemplaza HyperparamLoader)")
    reader = DBReader()

    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        params = reader.get_params(sym, 2024, 6)
        assert "exp_lambda" in params
        assert "init_exp_T" in params
        assert "source" in params
        assert params["exp_lambda"] > 0.9  # No debe ser default 0.9975
        print(f"  ✅ {sym}: λ={params['exp_lambda']:.4f}, T={params['init_exp_T']}, source={params['source']}")

    reader.close()


# ─────────────────────────────────────────
# Test 3: DBReader.get_bars()
# ─────────────────────────────────────────
def test_get_bars():
    print("\n[TEST 3] DBReader.get_bars()")
    reader = DBReader()

    df = reader.get_bars("BTCUSDT", "2024-01-01", "2024-01-02")
    assert len(df) > 0
    assert "close" in df.columns
    assert "exp_lambda" in df.columns
    assert "init_exp_t" in df.columns
    print(f"  ✅ BTC 2024-01-01: {len(df)} bars")

    # Verificar que init_exp_t no es NULL
    null_t = df["init_exp_t"].isna().sum()
    assert null_t == 0, f"Found {null_t} rows with NULL init_exp_t"
    print("  ✅ All rows have init_exp_t (no NULLs)")

    reader.close()


# ─────────────────────────────────────────
# Test 4: Roundtrip write + read
# ─────────────────────────────────────────
def test_roundtrip():
    print("\n[TEST 4] Roundtrip write + read")
    client = TimescaleDBClient()

    # Crear barra de prueba
    test_df = pd.DataFrame(
        {
            "open_time": [datetime(2099, 1, 1, 0, 0, 0)],
            "close_time": [datetime(2099, 1, 1, 0, 5, 0)],
            "symbol": ["TESTUSDT"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "n_ticks": [1000],
            "volume": [10.0],
            "dollar_value": [1000.0],
            "exp_lambda": [0.99],
            "init_exp_t": [500],
            "study_source": ["test_roundtrip"],
            "completion_rate": [0.95],
            "failed_day": [0],
            "sample_weight": [1.0],
            "log_return": [0.001],
            "bar_duration_secs": [300.0],
        }
    )

    # Escribir
    n = client.write_bars(test_df)
    assert n == 1
    print(f"  ✅ Written {n} test bar")

    # Leer
    reader = DBReader()
    df = reader.get_bars("TESTUSDT", "2099-01-01", "2099-01-02")
    assert len(df) == 1
    assert df.iloc[0]["close"] == 100.5
    assert df.iloc[0]["init_exp_t"] == 500
    print(f"  ✅ Read back: close={df.iloc[0]['close']}, T={df.iloc[0]['init_exp_t']}")

    # Limpiar
    client.execute("DELETE FROM bars WHERE symbol = 'TESTUSDT'")
    print("  ✅ Cleanup complete")

    reader.close()
    client.close()


# ─────────────────────────────────────────
# Test 5: Study quality summary
# ─────────────────────────────────────────
def test_study_summary():
    print("\n[TEST 5] Study quality summary")
    reader = DBReader()

    summary = reader.get_study_quality_summary("BTCUSDT")
    assert len(summary) > 0
    assert "composite_score" in summary.columns
    assert "avg_bar_duration" in summary.columns
    print(f"  ✅ {len(summary)} studies found for BTCUSDT")
    print(
        f"  ✅ Latest: score={summary.iloc[0]['composite_score']:.1f}, "
        f"avg_duration={summary.iloc[0]['avg_bar_duration']:.0f}s"
    )

    reader.close()


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TIMESCALEDB PIPELINE TESTS")
    print("=" * 60)

    tests = [
        test_connection,
        test_get_params,
        test_get_bars,
        test_roundtrip,
        test_study_summary,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
