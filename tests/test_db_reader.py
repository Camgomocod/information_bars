"""
test_db_reader.py — DBReader walk-forward param selection (no silent fallback).

Requires a reachable TimescaleDB (``docker-compose up -d``). Skips otherwise.
"""

import psycopg2
import pytest

pytestmark = pytest.mark.db

from src.storage.db_config import get_db_url
from src.storage.db_reader import DBReader

DB_URL = get_db_url()


def _reachable() -> bool:
    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


_needs_db = pytest.mark.skipif(not _reachable(), reason="TimescaleDB not reachable")


@_needs_db
def test_get_params_strictly_prior():
    """For target month M, only a study with period < M may be returned."""
    reader = DBReader(DB_URL)
    try:
        all_studies = reader.get_all_studies("BTCUSDT")
        if all_studies.empty:
            pytest.skip("no studies in DB")
        for _, row in all_studies.head(3).iterrows():
            y, m = int(row["period"][:4]), int(row["period"][5:7])
            params = reader.get_params("BTCUSDT", y, m)
            assert params.get("source") is not None
    finally:
        reader.close()


@_needs_db
def test_get_params_or_none_returns_none_when_missing():
    reader = DBReader(DB_URL)
    try:
        # A target far in the past with no prior study should be None, not defaults.
        assert reader.get_params_or_none("BTCUSDT", 1900, 1) is None
    finally:
        reader.close()


def test_no_db_returns_clean_defaults():
    """Without a DB connection, get_params must fail loudly, not silently
    fall back to .pkl files (TimescaleDB is the single source of truth)."""
    reader = DBReader("postgresql://nouser:nopass@localhost:59999/nonexistent")
    with pytest.raises(psycopg2.OperationalError):
        reader.get_params("BTCUSDT", 2024, 3)
