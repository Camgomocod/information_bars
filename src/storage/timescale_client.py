"""
TimescaleDB Client — Reemplaza timescaledb_writer.py
Centraliza toda la escritura a la base de datos.
"""

import json
from pathlib import Path

import pandas as pd
import psycopg2
import yaml
from psycopg2.extras import execute_values

from src.storage.db_config import get_db_url

DB_URL = get_db_url()


class TimescaleDBClient:
    """
    Cliente central para TimescaleDB.
    Reemplaza: src/storage/timescaledb_writer.py
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

    def write_bars(self, df: pd.DataFrame, symbol: str | None = None,
                   replace_window: bool = True) -> int:
        """
        Bulk insert de barras usando COPY (más rápido que INSERT).

        Idempotencia: con ``replace_window=True`` (default) se borra primero el
        rango temporal de la fuente para el símbolo, de modo que re-ejecutar un
        build reemplaza la ventana en vez de duplicar filas. No depende de una
        constraint de unicidad (el dataset histórico contiene timestamps
        compartidos con OHLCV distinto).
        """
        if df.empty:
            return 0

        # Asegurar que symbol esté en el DataFrame
        if symbol and 'symbol' not in df.columns:
            df = df.copy()
            df['symbol'] = symbol
        if symbol is None and 'symbol' in df.columns:
            symbol = str(df['symbol'].iloc[0])

        conn = self._get_conn()

        # Eliminar la ventana temporal ya persistida para idempotencia.
        if replace_window and symbol and 'open_time' in df.columns:
            t0 = pd.to_datetime(df['open_time']).min()
            t1 = pd.to_datetime(df['open_time']).max()
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM bars WHERE symbol = %s AND open_time BETWEEN %s AND %s",
                    (symbol, t0, t1),
                )
            conn.commit()
        columns = [
            'open_time', 'close_time', 'symbol',
            'open', 'high', 'low', 'close', 'n_ticks', 'volume', 'dollar_value',
            'exp_lambda', 'init_exp_t', 'study_source', 'completion_rate',
            'failed_day', 'sample_weight',
            'log_return', 'frac_diff_return', 'rolling_volatility',
            'bar_range', 'atr_pct', 'volume_z', 'dollar_value_z', 'n_ticks_z',
            'vwap', 'price_to_vwap', 'bar_duration_secs',
            'rsi', 'bb_pct_b', 'macd_hist', 'funding_rate_mean'
        ]

        # Renombrar init_exp_T -> init_exp_t si viene del parquet (antes de filtrar)
        if 'init_exp_T' in df.columns and 'init_exp_t' not in df.columns:
            df = df.copy()
            df = df.rename(columns={'init_exp_T': 'init_exp_t'})

        # Filtrar solo columnas que existen en el DataFrame
        cols = [c for c in columns if c in df.columns]
        sub = df[cols].copy()

        # Manejar NaT/None
        for col in ['open_time', 'close_time']:
            if col in sub.columns:
                sub[col] = pd.to_datetime(sub[col], errors='coerce')

        # Usar COPY para velocidad
        from io import StringIO
        buffer = StringIO()
        sub.to_csv(buffer, index=False, header=False, sep='|', na_rep='\\N')
        buffer.seek(0)

        with conn.cursor() as cur:
            cur.copy_from(
                buffer, 'bars',
                sep='|',
                null='\\N',
                columns=[c for c in cols if c in sub.columns]
            )
        conn.commit()
        return len(sub)

    # ─────────────────────────────────────────
    # STUDIES
    # ─────────────────────────────────────────

    def upsert_study(
        self,
        symbol: str,
        period: str,
        window_name: str,
        exp_lambda: float,
        init_exp_t: int,
        composite_score: float,
        quality_component: float,
        coverage_component: float,
        stability_component: float,
        granularity_component: float,
        total_trials: int,
        completed_trials: int,
        pruned_trials: int,
        completion_rate: float,
        avg_bars_per_day: float,
        t_min: int,
        t_max: int,
        lambda_min: float,
        lambda_max: float,
        sampler_type: str,
        device: str,
        search_bounds_json: dict | None = None,
    ):
        """
        Inserta o actualiza un estudio (UPSERT).
        """
        conn = self._get_conn()
        sql = """
        INSERT INTO studies (
            symbol, period, window_name,             exp_lambda, init_exp_t,
            composite_score, quality_component, coverage_component,
            stability_component, granularity_component,
            total_trials, completed_trials, pruned_trials,
            completion_rate, avg_bars_per_day,
            t_min, t_max, lambda_min, lambda_max,
            sampler_type, device, search_bounds_json
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, period) DO UPDATE SET
            window_name = EXCLUDED.window_name,
            exp_lambda = EXCLUDED.exp_lambda,
            init_exp_t = EXCLUDED.init_exp_t,
            composite_score = EXCLUDED.composite_score,
            quality_component = EXCLUDED.quality_component,
            coverage_component = EXCLUDED.coverage_component,
            stability_component = EXCLUDED.stability_component,
            granularity_component = EXCLUDED.granularity_component,
            total_trials = EXCLUDED.total_trials,
            completed_trials = EXCLUDED.completed_trials,
            pruned_trials = EXCLUDED.pruned_trials,
            completion_rate = EXCLUDED.completion_rate,
            avg_bars_per_day = EXCLUDED.avg_bars_per_day,
            t_min = EXCLUDED.t_min,
            t_max = EXCLUDED.t_max,
            lambda_min = EXCLUDED.lambda_min,
            lambda_max = EXCLUDED.lambda_max,
            sampler_type = EXCLUDED.sampler_type,
            device = EXCLUDED.device,
            search_bounds_json = EXCLUDED.search_bounds_json,
            created_at = NOW();
        """
        with conn.cursor() as cur:
            cur.execute(sql, (
                symbol, period, window_name, exp_lambda, init_exp_t,
                composite_score, quality_component, coverage_component,
                stability_component, granularity_component,
                total_trials, completed_trials, pruned_trials,
                completion_rate, avg_bars_per_day,
                t_min, t_max, lambda_min, lambda_max,
                sampler_type, device,
                json.dumps(search_bounds_json) if search_bounds_json else None,
            ))
        conn.commit()

    def upsert_study_from_report(
        self,
        symbol: str,
        period: str,
        report_path: Path,
        bounds_path: Path | None = None,
    ):
        """
        Parsea un bayesian_report.txt y un _search_bounds.yaml
        para insertar en la DB.
        """
        import re
        text = report_path.read_text()

        def extract(pattern, cast=str):
            m = re.search(pattern, text)
            if m:
                return cast(m.group(1))
            return None

        window_name = report_path.parent.parent.name
        exp_lambda = extract(r'exp_lambda:\s+([0-9.]+)', float) or 0.0
        init_exp_T = extract(r'init_exp_T:\s+([0-9]+)', int) or 0
        composite_score = extract(r'Composite Score:\s+([0-9.]+)', float) or 0.0
        total_trials = extract(r'Total trials:\s+([0-9]+)', int) or 0
        completed_trials = extract(r'Completed trials:\s+([0-9]+)', int) or 0
        pruned_trials = extract(r'Pruned trials:\s+([0-9]+)', int) or 0
        completion_rate = extract(r'Completion Rate:\s+([0-9.]+)%', float) or 0.0
        avg_bars_per_day = extract(r'bars/day=([0-9.]+)', float) or 0.0
        sampler_type = extract(r'Sampler:\s+(\w+)') or 'unknown'

        # Defaults para componentes (pueden no estar en report viejos)
        quality = composite_score * 0.6
        coverage = composite_score * 0.25
        stability = composite_score * 0.15
        granularity = 0.0  # removed from composite (v3: 60/25/15 IID-aligned)

        # Search bounds desde YAML
        bounds_json = None
        if bounds_path and bounds_path.exists():
            with open(bounds_path) as f:
                bounds = yaml.safe_load(f)
                bounds_json = bounds.get('search_bounds', {})
                t_min = bounds_json.get('T_min', 1)
                t_max = bounds_json.get('T_max', 10000)
                lambda_min = bounds_json.get('lambda_min', 0.9)
                lambda_max = bounds_json.get('lambda_max', 0.999)
        else:
            t_min, t_max, lambda_min, lambda_max = 1, 10000, 0.9, 0.999

        self.upsert_study(
            symbol=symbol,
            period=period,
            window_name=window_name,
            exp_lambda=exp_lambda,
            init_exp_t=init_exp_T,
            composite_score=composite_score,
            quality_component=quality,
            coverage_component=coverage,
            stability_component=stability,
            granularity_component=granularity,
            total_trials=total_trials,
            completed_trials=completed_trials,
            pruned_trials=pruned_trials,
            completion_rate=completion_rate / 100.0,
            avg_bars_per_day=avg_bars_per_day,
            t_min=t_min,
            t_max=t_max,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            sampler_type=sampler_type,
            device='cuda',
            search_bounds_json=bounds_json,
        )

    # ─────────────────────────────────────────
    # FAILED DAYS
    # ─────────────────────────────────────────

    def insert_failed_days(
        self,
        symbol: str,
        period: str,
        analysis_path: Path,
    ):
        """
        Parsea _failed_days_analysis.txt e inserta en failed_days.
        """
        if not analysis_path.exists():
            return

        conn = self._get_conn()
        text = analysis_path.read_text()

        dates = []
        for line in text.split('\n'):
            if line.startswith('DATE:'):
                date_str = line.split('DATE:')[1].strip()
                dates.append((symbol, period, date_str, 1, 1, 0, ['insufficient_bars']))

        if dates:
            sql = """
            INSERT INTO failed_days (symbol, period, failed_date, total_trials, insufficient_bars, errors, reasons)
            VALUES %s
            ON CONFLICT (symbol, period, failed_date) DO NOTHING;
            """
            with conn.cursor() as cur:
                execute_values(cur, sql, dates)
            conn.commit()

    # ─────────────────────────────────────────
    # POSITIONING (Funding Rate)
    # ─────────────────────────────────────────

    def write_positioning(self, df: pd.DataFrame, symbol: str):
        """
        Bulk insert de funding rates.
        """
        if df.empty:
            return

        conn = self._get_conn()
        records = [
            (symbol, row['funding_time'], row.get('funding_rate'), row.get('mark_price'))
            for _, row in df.iterrows()
        ]

        sql = """
        INSERT INTO positioning (symbol, funding_time, funding_rate, mark_price)
        VALUES %s
        ON CONFLICT (symbol, funding_time) DO UPDATE SET
            funding_rate = EXCLUDED.funding_rate,
            mark_price = EXCLUDED.mark_price;
        """
        with conn.cursor() as cur:
            execute_values(cur, sql, records)
        conn.commit()

    # ─────────────────────────────────────────
    # UTILIDADES
    # ─────────────────────────────────────────

    def execute(self, sql: str, params=None):
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description:
                return cur.fetchall()
        conn.commit()
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
