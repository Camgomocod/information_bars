"""
TimescaleDB Reader — Reemplaza HyperparamLoader
Centraliza todas las queries de lectura.
"""

from typing import Any

import pandas as pd
import psycopg2

from src.storage.db_config import get_db_url

DB_URL = get_db_url()


class DBReader:
    """
    Reemplaza src.pipeline.hyperparam_loader.HyperparamLoader

    Antes:
        loader = HyperparamLoader("experiments/")
        params = loader.get_params("BTCUSDT", 2024, 3)

    Ahora:
        reader = DBReader()
        params = reader.get_params("BTCUSDT", 2024, 3)
        df = reader.get_bars("BTCUSDT", "2024-01-01", "2024-06-30")
    """

    def __init__(self, db_url: str = DB_URL):
        self.db_url = db_url
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ─────────────────────────────────────────
    # BARS
    # ─────────────────────────────────────────

    def get_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        SELECT * FROM bars WHERE symbol = ? AND open_time BETWEEN ? AND ?
        """
        cols = ', '.join(columns) if columns else '*'
        sql = f"""
        SELECT {cols} FROM bars
        WHERE symbol = %s
          AND open_time BETWEEN %s AND %s
        ORDER BY open_time
        """
        return pd.read_sql(sql, self._get_conn(), params=(symbol, start, end))

    def get_bars_by_year(
        self,
        symbol: str,
        year: int,
    ) -> pd.DataFrame:
        """
        SELECT * FROM bars WHERE symbol = ? AND EXTRACT(YEAR FROM open_time) = ?
        """
        sql = """
        SELECT * FROM bars
        WHERE symbol = %s
          AND open_time >= %s AND open_time < %s
        ORDER BY open_time
        """
        start = f"{year}-01-01"
        end = f"{year + 1}-01-01"
        return pd.read_sql(sql, self._get_conn(), params=(symbol, start, end))

    # ─────────────────────────────────────────
    # STUDIES (reemplaza HyperparamLoader)
    # ─────────────────────────────────────────

    def get_params(
        self,
        symbol: str,
        target_year: int,
        target_month: int,
    ) -> dict[str, Any]:
        """
        Reemplaza HyperparamLoader.get_params()

        Walk-forward: busca el estudio más reciente cuyo periodo es ESTRICTAMENTE
        anterior al target. Un estudio bayesian_2024_03_w2m cubre datos hasta Marzo,
        y es válido para procesar Abril y Mayo (no Marzo).
        """
        period = f"{target_year}-{target_month:02d}"

        sql = """
        SELECT exp_lambda, init_exp_T, composite_score, window_name
        FROM studies
        WHERE symbol = %s AND period < %s
        ORDER BY period DESC
        LIMIT 1
        """
        df = pd.read_sql(sql, self._get_conn(), params=(symbol, period))

        if df.empty:
            return {
                "exp_lambda": 0.9975,
                "init_exp_T": 5000,
                "source": "default",
            }

        row = self._row_to_params(df)

        return row

    def get_params_or_none(
        self,
        symbol: str,
        target_year: int,
        target_month: int,
    ) -> dict[str, Any] | None:
        """
        Como get_params() pero retorna None en vez de defaults.
        Útil para saber si realmente existe un estudio walk-forward válido.
        """
        period = f"{target_year}-{target_month:02d}"

        sql = """
        SELECT exp_lambda, init_exp_T, composite_score, window_name
        FROM studies
        WHERE symbol = %s AND period < %s
        ORDER BY period DESC
        LIMIT 1
        """
        df = pd.read_sql(sql, self._get_conn(), params=(symbol, period))

        if df.empty:
            return None

        return self._row_to_params(df)

    @staticmethod
    def _row_to_params(df: pd.DataFrame) -> dict[str, Any]:
        """
        Convert the first row of a studies query result into canonical params
        with the init_exp_T key (see schema_contract).

        The underlying DB column may be named init_exp_T or init_exp_t depending
        on migration state; we normalize before reading so neither name breaks.
        """
        from src.storage.schema_contract import normalize_columns, normalize_params

        df = df.copy()
        df.columns = normalize_columns(list(df.columns))
        row = df.iloc[0]
        raw = {
            "exp_lambda": float(row['exp_lambda']),
            "init_exp_T": int(row['init_exp_T']),
            "composite_score": float(row['composite_score']),
            "source": row['window_name'],
        }
        return normalize_params(raw)

    def has_study_for_window(self, symbol: str, year: int, month: int) -> bool:
        """
        Verifica si existe un estudio completo para una ventana específica.
        Útil para detectar si el estudio de la ventana recién cerrada ya fue creado.
        """
        period = f"{year}-{month:02d}"
        sql = """
        SELECT 1 FROM studies
        WHERE symbol = %s AND period = %s
        LIMIT 1
        """
        df = pd.read_sql(sql, self._get_conn(), params=(symbol, period))
        return not df.empty

    def get_study(self, symbol: str, period: str) -> dict | None:
        """
        Obtiene un estudio específico por (symbol, period).
        """
        sql = "SELECT * FROM studies WHERE symbol = %s AND period = %s"
        df = pd.read_sql(sql, self._get_conn(), params=(symbol, period))
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_all_studies(self, symbol: str | None = None) -> pd.DataFrame:
        """
        SELECT * FROM studies ORDER BY period
        """
        if symbol:
            sql = "SELECT * FROM studies WHERE symbol = %s ORDER BY period"
            return pd.read_sql(sql, self._get_conn(), params=(symbol,))
        return pd.read_sql("SELECT * FROM studies ORDER BY symbol, period", self._get_conn())

    # ─────────────────────────────────────────
    # FAILED DAYS
    # ─────────────────────────────────────────

    def get_failed_days(self, symbol: str, period: str) -> pd.DataFrame:
        """
        SELECT * FROM failed_days WHERE symbol = ? AND period = ?
        """
        sql = """
        SELECT * FROM failed_days
        WHERE symbol = %s AND period = %s
        ORDER BY failed_date
        """
        return pd.read_sql(sql, self._get_conn(), params=(symbol, period))

    # ─────────────────────────────────────────
    # POSITIONING
    # ─────────────────────────────────────────

    def get_positioning(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        SELECT * FROM positioning WHERE symbol = ? AND funding_time BETWEEN ? AND ?
        """
        sql = """
        SELECT * FROM positioning
        WHERE symbol = %s
          AND funding_time BETWEEN %s AND %s
        ORDER BY funding_time
        """
        return pd.read_sql(sql, self._get_conn(), params=(symbol, start, end))

    # ─────────────────────────────────────────
    # VISTAS
    # ─────────────────────────────────────────

    def get_study_quality_summary(self, symbol: str | None = None) -> pd.DataFrame:
        """
        SELECT * FROM study_quality_summary
        """
        if symbol:
            sql = "SELECT * FROM study_quality_summary WHERE symbol = %s ORDER BY period"
            return pd.read_sql(sql, self._get_conn(), params=(symbol,))
        return pd.read_sql("SELECT * FROM study_quality_summary ORDER BY symbol, period", self._get_conn())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
