"""
daily_processor.py — Inference Pipeline (VPS Daily)

Processes yesterday's tick data for the PREDICTION layer. Completely separated
from the training pipeline (build_yearly_training_data.py).

Key differences from the training pipeline:
- Output: data_optimized/inference/{SYMBOL}_latest.parquet (rotating 300 bars)
- Fallback chain: walk-forward → default → mini-Optuna → aggregated bar
- NEVER skips a trading day (guarantees at least 1 bar with real price data)
- MAD filter per-day (not per-month)
- Idempotent: re-processing a day replaces it

Fallback chain per day:
    0. Download ticks from Binance
    1. Walk-forward params → DRBs             (if >= 40 bars → done)
    2. Default params by symbol → DRBs         (if >= 40 bars → done)
    3. Mini Optuna (20 trials, 60s) → DRBs    (if >= 40 bars → done)
    4. Aggregated OHLCV bar from raw ticks     (last resort, 1 bar)
    5. Post-write validation: frac_diff_return is NOT NaN
"""

import time
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


WINDOW_SIZE = 300
MIN_BARS_THRESHOLD = 40
CONTEXT_BARS = 300
SAMPLE_WEIGHT_NORMAL = 1.0
SAMPLE_WEIGHT_FAILED = 0.5
FEATURE_NAN_PADDING = 50

DEFAULT_PARAMS_BY_SYMBOL = {
    "BTCUSDT": {"exp_lambda": 0.9975, "init_exp_T": 14738},
    "ETHUSDT": {"exp_lambda": 0.9975, "init_exp_T": 5000},
    "BNBUSDT": {"exp_lambda": 0.9975, "init_exp_T": 500},
    "SOLUSDT": {"exp_lambda": 0.9975, "init_exp_T": 2000},
    "_default": {"exp_lambda": 0.9975, "init_exp_T": 5000},
}

OPTIMIZATION_WINDOWS = [
    {"target_month": 1},
    {"target_month": 3},
    {"target_month": 5},
    {"target_month": 7},
    {"target_month": 9},
    {"target_month": 11},
]


@dataclass
class DayResult:
    symbol: str
    target_date: str
    status: str
    bars: int = 0
    fallback_level: int = 0
    exp_lambda: float = 0.0
    init_exp_T: int = 0
    study_source: str = "default"
    message: str = ""
    failed_day: bool = False
    elapsed_sec: float = 0.0


@dataclass
class BatchResult:
    results: List[DayResult] = field(default_factory=list)
    optimization_triggered: List[str] = field(default_factory=list)
    total_elapsed_sec: float = 0.0

    @property
    def processed(self) -> int:
        return sum(1 for r in self.results if r.status == "processed")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    def summary(self) -> str:
        lines = [f"\n📊 {self.processed} processed | {self.skipped} skipped | {self.failed} failed"]
        if self.optimization_triggered:
            lines.append(f"   ⚡ Optimization: {', '.join(self.optimization_triggered)}")
        for r in self.results:
            flag = " ⚠️ FAILED" if r.failed_day else ""
            lvl = f" (level {r.fallback_level})" if r.fallback_level > 0 else ""
            if r.status == "skipped":
                lines.append(f"   ⏭️  {r.symbol} {r.target_date}: {r.message}")
            else:
                lines.append(f"   ✅ {r.symbol} {r.target_date}: {r.bars} bars "
                             f"(λ={r.exp_lambda:.4f}, T={r.init_exp_T}){lvl}{flag}")
        return "\n".join(lines)


class DailyProcessor:
    def __init__(
        self,
        output_dir: str = "data_optimized/inference/",
        experiments_dir: str = str(Path(__file__).resolve().parent.parent.parent / "experiments"),
        db_url: str = None,
        data_dir: str = "data_raw",
        verbose: bool = True,
        dry_run: bool = False,
    ):
        if db_url is None:
            from src.storage.db_config import get_db_url
            db_url = get_db_url()
        self.output_dir = Path(output_dir)
        self.experiments_dir = Path(experiments_dir)
        self.db_url = db_url
        self.data_dir = data_dir
        self.verbose = verbose
        self.dry_run = dry_run

        self._db_reader = None
        self._hyperparam_loader = None
        self._timescale_client = None

    @property
    def db_reader(self):
        if self._db_reader is None:
            try:
                from src.storage.db_reader import DBReader
                self._db_reader = DBReader(self.db_url)
            except Exception:
                self._db_reader = False
        return self._db_reader if self._db_reader is not False else None

    @property
    def hyperparam_loader(self):
        if self._hyperparam_loader is None:
            try:
                from src.pipeline.hyperparam_loader import HyperparamLoader
                self._hyperparam_loader = HyperparamLoader(self.experiments_dir)
            except Exception:
                self._hyperparam_loader = False
        return self._hyperparam_loader if self._hyperparam_loader is not False else None

    @property
    def timescale_client(self):
        if self._timescale_client is None:
            try:
                from src.storage.timescale_client import TimescaleDBClient
                self._timescale_client = TimescaleDBClient(self.db_url)
            except Exception:
                self._timescale_client = False
        return self._timescale_client if self._timescale_client is not False else None

    # ──────────────────────────────────────
    # Hyperparam resolution
    # ──────────────────────────────────────

    def _load_walk_forward_params(self, symbol: str, year: int, month: int) -> Dict:
        reader = self.db_reader
        if reader:
            try:
                p = reader.get_params_or_none(symbol, year, month)
                if p:
                    return {**p, "source": p.get("source", "db")}
            except Exception:
                pass

        loader = self.hyperparam_loader
        if loader:
            try:
                p = loader.get_params(symbol, year, month, verbose=False)
                if p:
                    return {**p, "source": "file"}
            except Exception:
                pass

        d = DEFAULT_PARAMS_BY_SYMBOL.get(symbol, DEFAULT_PARAMS_BY_SYMBOL["_default"])
        return {"exp_lambda": d["exp_lambda"], "init_exp_T": d["init_exp_T"],
                "init_exp_t": d["init_exp_T"], "source": "default"}

    def _load_default_params(self, symbol: str) -> Dict:
        d = DEFAULT_PARAMS_BY_SYMBOL.get(symbol, DEFAULT_PARAMS_BY_SYMBOL["_default"])
        return {"exp_lambda": d["exp_lambda"], "init_exp_T": d["init_exp_T"],
                "init_exp_t": d["init_exp_T"], "source": "default"}

    # ──────────────────────────────────────
    # Month boundary
    # ──────────────────────────────────────

    def _needs_optimization(self, symbol: str, target_date: date) -> Optional[Dict]:
        for w in OPTIMIZATION_WINDOWS:
            if w["target_month"] != target_date.month:
                continue
            year = target_date.year
            window_name = f"bayesian_{year}_{target_date.month:02d}_w2m"

            reader = self.db_reader
            if reader:
                try:
                    if reader.has_study_for_window(symbol, year, target_date.month):
                        return None
                except Exception:
                    pass

            pkl_path = (self.experiments_dir / window_name / symbol /
                        f"{symbol}_optuna_study.pkl")
            if pkl_path.exists():
                return None

            return w
        return None

    def _run_optimization_for_window(
        self, symbol: str, year: int, target_month: int,
        source: str = "auto", trials: int = 50,
    ) -> bool:
        import subprocess
        import sys

        window_name = f"bayesian_{year}_{target_month:02d}_w2m"
        cmd = [
            sys.executable, "optimization/tune_multiasset_hyperparams.py",
            "--year", str(year), "--month", str(target_month),
            "--window", "2", "--trials", str(trials),
            "--parallel", "1", "--source", source,
            "--data-dir", self.data_dir, "--symbols", symbol,
        ]
        project_root = Path(__file__).resolve().parent.parent.parent

        print(f"\n{'─' * 60}")
        print(f"  ⚡ AUTO-OPT: {window_name} for {symbol}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'─' * 60}\n")

        try:
            subprocess.run(cmd, cwd=str(project_root), check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Optimization FAILED (exit {e.returncode})")
            return False

    # ──────────────────────────────────────
    # Tick download and cleaning
    # ──────────────────────────────────────

    def _download_and_clean(self, symbol: str, date_str: str, source: str
                            ) -> Optional[pd.DataFrame]:
        from src.connectors.download_data import DownloadData
        from src.normalizers.data_normalizer import DataNormalizer

        downloader = DownloadData(symbol=symbol, source=source, data_dir=self.data_dir)
        df_raw = downloader.download_day(date_str)

        if df_raw is None or df_raw.empty:
            return None

        normalizer = DataNormalizer()
        needed = ["timestamp", "price", "quantity"]
        if "dollar_value" in df_raw.columns:
            needed.append("dollar_value")
        df_raw = df_raw[needed].copy()
        df_raw = normalizer.clean_raw_ticks(df_raw, mad_window=100, k=10.0)

        if df_raw is None or df_raw.empty:
            return None

        if "dollar_value" not in df_raw.columns:
            df_raw["dollar_value"] = df_raw["price"] * df_raw["quantity"]

        return df_raw

    # ──────────────────────────────────────
    # Bar generation
    # ──────────────────────────────────────

    def _generate_drbs(self, df_raw: pd.DataFrame, params: Dict) -> Optional[pd.DataFrame]:
        from src.bars.info_bars import OptimizedInfoRunBars

        bars_gen = OptimizedInfoRunBars(save_path=None)
        init_exp_T = params.get("init_exp_T", params.get("init_exp_t", 5000))
        new_bars = bars_gen.get_drbs(
            df=df_raw,
            exp_lambda=params["exp_lambda"],
            init_exp_T=init_exp_T,
        )

        if new_bars is None or new_bars.empty:
            return None

        for col in ["open_time", "close_time"]:
            if col in new_bars.columns and new_bars[col].dtype != "datetime64[ns]":
                new_bars[col] = pd.to_datetime(new_bars[col])

        for col in ["open", "high", "low", "close", "volume", "dollar_value"]:
            if col in new_bars.columns:
                new_bars[col] = new_bars[col].astype(np.float32)
        if "n_ticks" in new_bars.columns:
            new_bars["n_ticks"] = new_bars["n_ticks"].astype(np.int32)

        return new_bars

    def _build_aggregated_bar(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        prices = df_raw["price"].values
        quantities = df_raw["quantity"].values
        dollar_values = df_raw["dollar_value"].values
        timestamps = df_raw["timestamp"].values

        return pd.DataFrame([{
            "open_time": pd.Timestamp(timestamps[0]),
            "close_time": pd.Timestamp(timestamps[-1]),
            "open": float(prices[0]),
            "high": float(prices.max()),
            "low": float(prices.min()),
            "close": float(prices[-1]),
            "volume": float(quantities.sum()),
            "dollar_value": float(dollar_values.sum()),
            "n_ticks": len(df_raw),
        }])

    # ──────────────────────────────────────
    # Context loading
    # ──────────────────────────────────────

    def _load_context(self, latest_parquet: Path, date_str: str) -> pd.DataFrame:
        ohclv_cols = [
            "open_time", "close_time", "open", "high", "low", "close",
            "n_ticks", "volume", "dollar_value",
        ]

        if latest_parquet.exists():
            existing = pd.read_parquet(latest_parquet)
            yesterday_dt = pd.Timestamp(date_str)
            before = existing[pd.to_datetime(existing["open_time"]) < yesterday_dt]
            if len(before) == 0:
                before = existing.head(1)
            context = before[ohclv_cols].tail(CONTEXT_BARS).copy()
            if self.verbose and len(context) > 0:
                last_dt = pd.to_datetime(context["open_time"].iloc[-1]).strftime("%Y-%m-%d")
                print(f"   📚 Context: {len(context)} bars (last: {last_dt})")
            return context
        else:
            if self.verbose:
                print(f"   🆕 No existing inference parquet — fresh start")
            return pd.DataFrame(columns=ohclv_cols)

    # ──────────────────────────────────────
    # Feature computation
    # ──────────────────────────────────────

    def _compute_features(self, combined: pd.DataFrame, new_bars_count: int,
                          symbol: str, year: int, month: int
                          ) -> Optional[pd.DataFrame]:
        from src.features.base_features import compute_all_features, FEATURE_COLUMNS
        from src.normalizers.data_normalizer import Winsorizer

        try:
            # Compute features WITHOUT winsorization first; the causal Winsorizer
            # is fit on the historical context bars only, so the new day's rows
            # cannot influence their own clip bounds (no lookahead leakage).
            kwargs = dict(
                drop_warmup=True, winsorize=False,
                symbol=symbol, positioning_data_dir=self.data_dir + "/futures",
            )
            features = compute_all_features(combined, **kwargs)
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Feature error: {e}")
            return None

        if features is None or features.empty:
            return None

        try:
            context_count = max(len(features) - new_bars_count, 0)
            context = features.iloc[:context_count]
            feat_present = [c for c in FEATURE_COLUMNS if c in features.columns]
            if context_count > 0 and feat_present:
                wz = Winsorizer(limits=(0.01, 0.01)).fit(context[feat_present])
                features = wz.transform(features)
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Winsorizer failed, using raw features: {e}")

        return features.tail(new_bars_count).copy()

    def _attach_metadata(self, df: pd.DataFrame, symbol: str, year: int,
                         month: int, params: Dict, is_failed: bool) -> pd.DataFrame:
        df["symbol"] = symbol
        df["year"] = np.int16(year)
        df["month"] = np.int8(month)
        df["exp_lambda"] = np.float32(params["exp_lambda"])
        init_exp_T = params.get("init_exp_T", params.get("init_exp_t", 5000))
        df["init_exp_T"] = np.int32(init_exp_T)
        df["study_source"] = "inference_pipeline"
        df["completion_rate"] = np.float32(1.0)
        df["failed_day"] = np.int8(1 if is_failed else 0)
        df["sample_weight"] = np.float32(
            SAMPLE_WEIGHT_FAILED if is_failed else SAMPLE_WEIGHT_NORMAL
        )
        return df

    # ──────────────────────────────────────
    # Parquet I/O
    # ──────────────────────────────────────

    def _save_with_rotation(self, latest_parquet: Path, new_features: pd.DataFrame,
                            target_date: date) -> int:
        latest_parquet.parent.mkdir(parents=True, exist_ok=True)

        if latest_parquet.exists():
            existing = pd.read_parquet(latest_parquet)
            existing_dates = pd.to_datetime(existing["open_time"]).dt.date
            if target_date in existing_dates.values:
                existing = existing[
                    pd.to_datetime(existing["open_time"]).dt.date != target_date
                ]
            full = pd.concat([existing, new_features], ignore_index=True)
            full = full.sort_values("open_time").reset_index(drop=True)
        else:
            full = new_features

        if len(full) > WINDOW_SIZE:
            full = full.tail(WINDOW_SIZE).reset_index(drop=True)
            if self.verbose:
                print(f"   🔄 Rotated to {WINDOW_SIZE} bars")

        full.to_parquet(latest_parquet, index=False, engine="pyarrow", compression="snappy")
        return len(full)

    def _validate_features(self, latest_parquet: Path) -> bool:
        if not latest_parquet.exists():
            return True

        df = pd.read_parquet(latest_parquet)
        tail = df.tail(FEATURE_NAN_PADDING)
        critical = ["log_return", "frac_diff_return", "rolling_volatility"]
        cols_present = [c for c in critical if c in tail.columns]

        if not cols_present:
            return True

        for col in cols_present:
            if tail[col].isna().any():
                if self.verbose:
                    print(f"   ⚠️  NaN in {col} — features may be incomplete")
                return False
        return True

    def _write_to_timescaledb(self, df: pd.DataFrame, symbol: str) -> bool:
        client = self.timescale_client
        if client is None:
            return False
        try:
            df_db = df.copy()
            if "init_exp_t" not in df_db.columns and "init_exp_T" in df_db.columns:
                df_db["init_exp_t"] = df_db["init_exp_T"]
            client.write_bars(df_db, symbol=symbol)
            return True
        except Exception:
            return False

    # ──────────────────────────────────────
    # Core: process a single day
    # ──────────────────────────────────────

    def process_day(
        self,
        symbol: str,
        target_date: date,
        source: str = "auto",
        new_month_strategy: str = "use-latest",
        optimize_trials: int = 50,
    ) -> DayResult:
        t0 = time.time()
        date_str = target_date.strftime("%Y-%m-%d")
        year, month = target_date.year, target_date.month

        latest_parquet = self.output_dir / f"{symbol}_latest.parquet"

        if self.verbose:
            print(f"\n📅 {symbol} — {date_str}")
            print(f"   Output: {latest_parquet}")

        if target_date.day <= 2:
            w = self._needs_optimization(symbol, target_date)
            if w and new_month_strategy == "optimize":
                self._run_optimization_for_window(
                    symbol, year, w["target_month"],
                    source=source, trials=optimize_trials,
                )
            elif w and new_month_strategy == "fail":
                return DayResult(symbol=symbol, target_date=date_str,
                                 status="failed",
                                 message=f"Study missing for month {month}",
                                 elapsed_sec=time.time() - t0)

        df_raw = self._download_and_clean(symbol, date_str, source)
        if df_raw is None:
            return DayResult(symbol=symbol, target_date=date_str,
                             status="failed",
                             message="No data available (download failed after retries)",
                             elapsed_sec=time.time() - t0)

        if self.verbose:
            print(f"   📥 {len(df_raw):,} ticks")

        params = None
        new_bars = None
        fallback_level = 0

        for level in range(4):
            fallback_level = level

            if level == 0:
                params = self._load_walk_forward_params(symbol, year, month)
            elif level == 1:
                params = self._load_default_params(symbol)
                if self.verbose:
                    print(f"   🔄 Fallback L1: defaults (λ={params['exp_lambda']:.4f}, "
                          f"T={params.get('init_exp_T', params.get('init_exp_t', 0))})")
            elif level == 2:
                if self.verbose:
                    print(f"   🔄 Fallback L2: mini-Optuna (20 trials, 60s)")
                try:
                    from src.pipeline.mini_optimizer import MiniOptimizer
                    opt = MiniOptimizer(symbol, verbose=self.verbose)
                    best = opt.optimize(df_raw, trials=20, timeout_sec=60)
                    if best:
                        params = {"exp_lambda": best["exp_lambda"],
                                  "init_exp_T": best["init_exp_T"],
                                  "source": "mini_optuna"}
                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️  Mini-optimizer failed: {e}")

                if params is None:
                    continue
            elif level == 3:
                if self.verbose:
                    print(f"   🔄 Fallback L3: aggregated bar")
                new_bars = self._build_aggregated_bar(df_raw)
                break

            new_bars = self._generate_drbs(df_raw, params)
            if new_bars is not None and len(new_bars) >= MIN_BARS_THRESHOLD:
                break

            if self.verbose:
                n = len(new_bars) if new_bars is not None else 0
                print(f"   ⚠️  Level {level}: {n} bars (< {MIN_BARS_THRESHOLD})")

        is_failed = False
        if new_bars is None:
            if self.verbose:
                print(f"   ❌ All levels exhausted — building aggregated bar")
            new_bars = self._build_aggregated_bar(df_raw)
            is_failed = True

        if self.verbose:
            lvl_label = {0: "WF", 1: "DEF", 2: "OPT", 3: "AGG"}
            print(f"   📊 {len(new_bars)} bars (level: {lvl_label.get(fallback_level, '?')})")

        context = self._load_context(latest_parquet, date_str)
        combined = pd.concat([context, new_bars], ignore_index=True)
        combined = combined.sort_values("open_time").reset_index(drop=True)

        new_features = self._compute_features(combined, len(new_bars),
                                              symbol, year, month)
        if new_features is None or new_features.empty:
            if self.verbose:
                print(f"   ⚠️  Feature warm-up ate all bars — using raw bars")
            new_features = new_bars.copy()
            is_failed = True

        if len(new_features) < MIN_BARS_THRESHOLD:
            is_failed = True

        new_features = self._attach_metadata(new_features, symbol, year, month,
                                             params or self._load_default_params(symbol),
                                             is_failed)

        p = params or self._load_default_params(symbol)
        if self.verbose:
            print(f"   ✅ {len(new_features)} bars with features")

        if self.dry_run:
            if self.verbose:
                print(f"   🔍 DRY RUN — skip save ({len(new_features)} bars)")
        else:
            total = self._save_with_rotation(latest_parquet, new_features, target_date)
            sz_mb = latest_parquet.stat().st_size / 1e6 if latest_parquet.exists() else 0
            if self.verbose:
                print(f"   💾 Saved: {latest_parquet} (total: {total} bars, {sz_mb:.1f} MB)")

            self._write_to_timescaledb(new_features, symbol)
            self._validate_features(latest_parquet)

        return DayResult(
            symbol=symbol, target_date=date_str,
            status="processed", bars=len(new_features),
            fallback_level=fallback_level,
            exp_lambda=p["exp_lambda"],
            init_exp_T=p.get("init_exp_T", p.get("init_exp_t", 0)),
            study_source=p.get("source", "default"),
            failed_day=is_failed,
            elapsed_sec=time.time() - t0,
        )

    def bootstrap_from_training(
        self,
        symbol: str,
        training_dir: str = "data_optimized/training/",
        context_bars: int = 500,
    ) -> int:
        """
        Seed the inference parquet from the training data.
        Copies the last `context_bars` bars from the training parquet into
        the inference parquet so that feature computation has warm-up data.

        Should be run once per symbol before the first daily processing.

        Returns: number of bars copied
        """
        training_dir = Path(training_dir)
        latest_parquet = self.output_dir / f"{symbol}_latest.parquet"

        if latest_parquet.exists():
            existing = pd.read_parquet(latest_parquet)
            if len(existing) >= context_bars:
                if self.verbose:
                    print(f"   ✅ Already bootstrapped: {len(existing)} bars")
                return len(existing)

        training_files = sorted(training_dir.rglob(f"{symbol}_*.parquet"))
        if not training_files:
            if self.verbose:
                print(f"   ⚠️  No training data found for {symbol}")
            return 0

        full = []
        for f in training_files:
            full.append(pd.read_parquet(f))
        df = pd.concat(full, ignore_index=True)
        df = df.sort_values("open_time").reset_index(drop=True)

        bootstrap = df.tail(context_bars).copy()
        latest_parquet.parent.mkdir(parents=True, exist_ok=True)
        bootstrap.to_parquet(latest_parquet, index=False, engine="pyarrow",
                            compression="snappy")

        if self.verbose:
            sz_mb = latest_parquet.stat().st_size / 1e6
            print(f"   🚀 Bootstrapped {len(bootstrap)} bars from training "
                  f"({sz_mb:.1f} MB)")

        return len(bootstrap)

    def process_yesterday(
        self,
        symbols: List[str],
        source: str = "auto",
        new_month_strategy: str = "use-latest",
        optimize_trials: int = 50,
        target_date: Optional[date] = None,
    ) -> BatchResult:
        yesterday = target_date or (date.today() - timedelta(days=1))
        t0 = time.time()
        results = []

        for symbol in symbols:
            r = self.process_day(
                symbol=symbol, target_date=yesterday,
                source=source, new_month_strategy=new_month_strategy,
                optimize_trials=optimize_trials,
            )
            results.append(r)
            if self.verbose:
                lvl = f" L{r.fallback_level}" if r.fallback_level else ""
                if r.status == "processed":
                    print(f"   ✓ {r.symbol}: {r.bars} bars{lvl} ({r.elapsed_sec:.1f}s)")
                elif r.status == "skipped":
                    print(f"   → {r.symbol}: {r.message}")
                else:
                    print(f"   ✗ {r.symbol}: {r.message}")

        batch = BatchResult(results=results, total_elapsed_sec=time.time() - t0)
        if self.verbose:
            print(batch.summary())
        return batch
