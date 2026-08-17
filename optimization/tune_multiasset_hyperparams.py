"""
Multi-Asset Hyperparameter Tuning for Dollar Run Bars (DRBs)
Optimized version: Lambda fixed at 0.99, only tuning T parameter
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Iterator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import json
import yaml
import sys
import warnings
import gc
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.connectors.download_data import DownloadData
from src.bars.info_bars import OptimizedInfoRunBars, _compute_drbs_core
from src.bars.base_bars import BaseBars
from src.bars.bars_statistics import BarsStatistics
from src.normalizers.data_normalizer import DataNormalizer
from src.optimization.samplers.factory import StudyFactory, SamplerType, DeviceType

warnings.filterwarnings("ignore")

# Style configuration
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class EarlyStoppingCallback:
    """
    Callback for early stopping when:
    - N consecutive pruned trials are detected, OR
    - N consecutive completed trials produce no improvement in best value.
    """

    def __init__(self, n_pruned_consecutive: int = 10, n_no_improvement: int = 15):
        self.n_pruned_consecutive = n_pruned_consecutive
        self.n_no_improvement = n_no_improvement
        self.consecutive_pruned = 0
        self.consecutive_no_improvement = 0
        self.best_value_seen = None

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        if trial.state == optuna.trial.TrialState.PRUNED:
            self.consecutive_pruned += 1
            self.consecutive_no_improvement += 1

            if self.consecutive_pruned >= self.n_pruned_consecutive:
                print(f"\n{'!' * 60}")
                print(f"🛑 EARLY STOPPING: {self.n_pruned_consecutive} consecutive pruned trials")
                best_str = f"{study.best_value:.2f}" if study.best_value is not None and study.best_value > 0 else "none"
                print(f"   Best value found: {best_str}")
                print(f"{'!' * 60}\n")
                study.stop()

        elif trial.state == optuna.trial.TrialState.COMPLETE:
            self.consecutive_pruned = 0
            current_best = study.best_value

            if current_best is not None and current_best > 0:
                if self.best_value_seen is None or current_best > self.best_value_seen + 1e-6:
                    # Genuine improvement
                    self.best_value_seen = current_best
                    self.consecutive_no_improvement = 0
                else:
                    self.consecutive_no_improvement += 1
                    if self.consecutive_no_improvement >= self.n_no_improvement:
                        print(f"\n{'!' * 60}")
                        print(f"🛑 EARLY STOPPING: {self.n_no_improvement} trials with no improvement")
                        print(f"   Best value: {current_best:.2f} (stagnated)")
                        print(f"{'!' * 60}\n")
                        study.stop()
            else:
                self.consecutive_no_improvement += 1
        else:
            self.consecutive_pruned = 0


class MultiAssetExperiment:
    """
        Experiments with hyperparameters for multiple assets
        Version with Bayesian Optimization using Optuna

        v2 Improvements:
        - D1: Compound score (quality 60% + coverage 25% + stability 15%)
        - D2: Adaptive search bounds derived from previous study
        - D3: Regime detection for adaptive window
    """

    # Default ranges (fallback if no prior study exists)
    # T_MIN floors per symbol prevent degenerate T=1 collapse
    DEFAULT_SEARCH_BOUNDS = {
        "BTCUSDT":  {"T_min": 500,  "T_max": 15000, "lambda_min": 0.96, "lambda_max": 0.999},
        "ETHUSDT":  {"T_min": 500,  "T_max": 5000,  "lambda_min": 0.96, "lambda_max": 0.999},
        "BNBUSDT":  {"T_min": 50,   "T_max": 2000,  "lambda_min": 0.96, "lambda_max": 0.999},
        "SOLUSDT":  {"T_min": 100,  "T_max": 5000,  "lambda_min": 0.96, "lambda_max": 0.999},
        "_default": {"T_min": 100,  "T_max": 5000,  "lambda_min": 0.96, "lambda_max": 0.999},
    }

    def __init__(
        self, symbol: str, results_dir: Path, prev_study_dir: Path = None,
        source: str = "binance", data_dir: str = "data_raw"
    ):
        self.symbol = symbol
        self.results_dir = Path(results_dir) / symbol
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.stats_calculator = BarsStatistics()

        # Cache for daily data
        self.daily_data_cache = {}

        # Adaptive threshold (default 100%, adjusted by diagnostics)
        self.completion_threshold = 1.0
        self.diagnostics_results = None

        # Fixed seed for the pre-optimization diagnostic day sampling
        # (independent from the Optuna sampler seed, but stable across runs).
        self._diagnostic_seed = 42

        # D2: Load adaptive bounds from previous study
        self.search_bounds = self._load_adaptive_bounds(prev_study_dir)

        # Import necessary classes
        self.downloader = DownloadData(symbol=symbol, source=source, data_dir=data_dir)
        self.bar_generator = OptimizedInfoRunBars(self.results_dir)
        self.normalizer = DataNormalizer()

    def _load_adaptive_bounds(self, prev_study_dir: Path = None) -> Dict:
        """
        D2: Derives adaptive search bounds from the previous study.
        If no previous study exists, utilizes the default fallbacks.
        """
        defaults = self.DEFAULT_SEARCH_BOUNDS.get(
            self.symbol, self.DEFAULT_SEARCH_BOUNDS["_default"]
        )

        if prev_study_dir is None:
            print(f"   📐 {self.symbol}: Using DEFAULT search bounds (no previous study)")
            print(f"      T: [{defaults['T_min']}, {defaults['T_max']}], λ: [{defaults['lambda_min']}, {defaults['lambda_max']}]")
            return defaults

        study_pkl = Path(prev_study_dir) / self.symbol / f"{self.symbol}_optuna_study.pkl"

        if not study_pkl.exists():
            print(f"   📐 {self.symbol}: Previous study not found at {study_pkl}, using defaults")
            return defaults

        try:
            import joblib
            prev_study = joblib.load(study_pkl)
            bounds = self._compute_adaptive_bounds(prev_study, defaults, self.symbol)
            print(f"   📐 {self.symbol}: ADAPTIVE bounds from {study_pkl.parent.parent.name}")
            print(f"      T: [{bounds['T_min']}, {bounds['T_max']}], λ: [{bounds['lambda_min']:.4f}, {bounds['lambda_max']:.4f}]")
            return bounds
        except Exception as e:
            print(f"   ⚠️  Error loading previous study: {e}, using defaults")
            return defaults

    @staticmethod
    def _compute_adaptive_bounds(study: optuna.Study, current_defaults: Dict, symbol: str = None) -> Dict:
        """
        Extracts search bounds for the next study based
        on the distribution of the best trials from the current study.
        """
        # Symbol-dependent lambda hard caps to prevent cascade lock
        LAMBDA_CAPS = {
            "BTCUSDT": 0.993,
            "ETHUSDT": 0.995,
            "SOLUSDT": 0.996,
            "BNBUSDT": 0.995,
        }
        lam_cap = LAMBDA_CAPS.get(symbol, 0.995)

        # Symbol-dependent T floors to prevent E[T] → 0 collapse
        # When E[T] shrinks below the floor, imbalance threshold ≈ min_ticks
        # becomes the sole bar-close trigger, degenerating DRBs into tick-count bars
        T_FLOORS = {
            "BTCUSDT": 200,
            "ETHUSDT": 300,
            "SOLUSDT": 100,
            "BNBUSDT": 50,
            "_default": 100,
        }
        T_floor = T_FLOORS.get(symbol, T_FLOORS["_default"])

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value > 0
        ]

        if len(completed) < 3:
            return current_defaults

        # Top 20% de trials (mínimo 3)
        top_n = max(3, len(completed) // 5)
        top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_n]

        t_values = [t.params["init_exp_T"] for t in top_trials]
        lambda_values = [t.params["exp_lambda"] for t in top_trials]

        t_center = np.median(t_values)
        t_spread = np.std(t_values) if len(t_values) > 1 else t_center * 0.3

        # Rango: mediana ± 2σ abajo, ± 3σ arriba (asimétrico para exploración)
        next_T_min = max(T_floor, int(t_center - 2.0 * t_spread))
        next_T_max = int(t_center + 3.0 * t_spread)

        # FIX: Force minimum T diversity to avoid cascade collapse (same as lambda fix)
        # If the adaptive T range is too narrow relative to the floor, expand around center
        if next_T_max - next_T_min < T_floor:
            print(f"      ⚠️  T range too narrow ({next_T_min}–{next_T_max}). Forcing diversity (floor={T_floor}).")
            next_T_min = max(T_floor, int(t_center - T_floor))
            next_T_max = int(t_center + T_floor)
            if next_T_max <= next_T_min:
                next_T_max = next_T_min * 3
            print(f"      ✅ Adjusted T range: [{next_T_min}, {next_T_max}]")

        # Safety: expandir si los mejores tocan los bordes del rango anterior
        p90 = np.percentile(t_values, 90)
        p10 = np.percentile(t_values, 10)

        if p90 > current_defaults["T_max"] * 0.85:
            next_T_max = int(current_defaults["T_max"] * 1.5)
        if p10 < current_defaults["T_min"] * 1.15:
            next_T_min = max(T_floor, int(current_defaults["T_min"] * 0.5))

        # Asegurar rango mínimo razonable
        if next_T_max <= next_T_min:
            next_T_max = next_T_min * 3

        # Lambda: rango más conservador con HARD CAP por símbolo
        lam_center = np.median(lambda_values)
        lam_spread = np.std(lambda_values) if len(lambda_values) > 1 else 0.02
        next_lam_min = max(0.90, lam_center - 2 * lam_spread)
        next_lam_max = min(lam_cap, lam_center + 2 * lam_spread)

        if next_lam_max <= next_lam_min:
            next_lam_max = min(lam_cap, next_lam_min + 0.05)

        # FIX: Force minimum lambda diversity to avoid cascade lock (lambda_min == lambda_max)
        # If the adaptive range is too narrow (< 0.01), expand it around the center
        # while respecting symbol-dependent absolute bounds [0.96, lam_cap].
        if next_lam_max - next_lam_min < 0.01:
            print(f"      ⚠️  Lambda range too narrow ({next_lam_min:.4f}–{next_lam_max:.4f}). Forcing diversity.")
            next_lam_min = max(0.96, round(float(lam_center) - 0.015, 4))
            next_lam_max = min(lam_cap, round(float(lam_center) + 0.015, 4))
            if next_lam_max <= next_lam_min:
                next_lam_max = min(lam_cap, next_lam_min + 0.05)
            print(f"      ✅ Adjusted lambda range: {next_lam_min:.4f}–{next_lam_max:.4f}")

        return {
            "T_min": next_T_min,
            "T_max": next_T_max,
            "lambda_min": round(float(next_lam_min), 4),
            "lambda_max": round(float(next_lam_max), 4),
            "derived_from_study": True,
            "top_trials_used": top_n,
            "t_center": float(t_center),
            "t_spread": float(t_spread),
        }

    @staticmethod
    def _sanitize_for_yaml(obj):
        """Recursively convert numpy types to Python native types for clean YAML serialization."""
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: MultiAssetExperiment._sanitize_for_yaml(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [MultiAssetExperiment._sanitize_for_yaml(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(MultiAssetExperiment._sanitize_for_yaml(item) for item in obj)
        else:
            return obj

    def define_search_space(self, trial: optuna.Trial) -> Dict:
        """
        Defines search space using adaptive bounds (D2).
        Bounds originate from _load_adaptive_bounds.
        """
        b = self.search_bounds

        exp_lambda = trial.suggest_float(
            "exp_lambda", b["lambda_min"], b["lambda_max"], log=False
        )
        init_exp_T = trial.suggest_int(
            "init_exp_T", b["T_min"], b["T_max"], log=True
        )

        return {"exp_lambda": exp_lambda, "init_exp_T": init_exp_T}

    def load_data_period(self, start_date: datetime, end_date: datetime):
        """
        Loads specific data period into memory (cache).

        OPTIMIZATION: Precomputes b_t (tick directions) and dollar_values once
        per day, storing numpy arrays ready for _compute_drbs_core().
        This avoids ~3,100 DataFrame copies + recalculations per symbol.
        """
        print(f"\n📥 Loading data period: {start_date.date()} to {end_date.date()}...")

        base_bars = BaseBars()
        current = start_date
        dates_loaded = 0

        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")

            # Skip if already loaded
            if date_str in self.daily_data_cache:
                current += timedelta(days=1)
                continue

            print(f"   Loading {date_str}...", end="", flush=True)

            try:
                df_raw = self.downloader.download_day(date_str)

                if df_raw is None or len(df_raw) == 0:
                    print(" ⚠️  No data")
                    current += timedelta(days=1)
                    continue

                # Limpiar datos
                df_raw = self.normalizer.clean_raw_ticks(df_raw)

                if len(df_raw) == 0:
                    print(" ⚠️  No valid data after cleaning")
                    current += timedelta(days=1)
                    continue

                # Solo columnas necesarias
                df_raw = df_raw[["timestamp", "price", "quantity"]].copy()

                # PRECOMPUTE: tick directions + dollar values (una sola vez por día)
                # Calcular ANTES de reducir dtypes (necesita float64 para precisión)
                b_t_arr = base_bars.get_tick_directions(df_raw).values.astype(np.float64)
                dollar_values_arr = (df_raw["price"].values * df_raw["quantity"].values).astype(np.float64)

                # Reducir dtypes del DataFrame para ahorrar RAM
                # price y quantity como float32 (suficiente para OHLCV bars)
                df_raw["price"] = df_raw["price"].astype(np.float32)
                df_raw["quantity"] = df_raw["quantity"].astype(np.float32)

                # Cache: DataFrame MÍNIMO (3 cols) + arrays precomputados separados
                # NO añadir b_t/dollar_value al DataFrame (ya están en arrays)
                self.daily_data_cache[date_str] = {
                    "df": df_raw,
                    "b_t": b_t_arr,
                    "dollar_values": dollar_values_arr,
                }
                dates_loaded += 1
                print(f" ✅ {len(df_raw):,} ticks")

            except Exception as e:
                print(f" ❌ Error: {e}")

            current += timedelta(days=1)

        # Estimar uso de memoria del cache
        total_mem_mb = 0
        for day_cache in self.daily_data_cache.values():
            df_mem = day_cache["df"].memory_usage(deep=True).sum()
            arr_mem = day_cache["b_t"].nbytes + day_cache["dollar_values"].nbytes
            total_mem_mb += (df_mem + arr_mem) / (1024 * 1024)

        print(f"\n✅ Cache loaded: {dates_loaded} new days (Total: {len(self.daily_data_cache)})")
        print(f"   💾 Estimated cache memory: {total_mem_mb:.0f} MB ({total_mem_mb/1024:.1f} GB)\n")
        return len(self.daily_data_cache)

    @staticmethod
    def _determine_threshold(success_rate: float) -> float:
        """Determines adaptive threshold based on success rate"""
        if success_rate >= 0.80:
            return 1.00  # Excellent month, demand 100%
        elif success_rate >= 0.60:
            return 0.90  # Good month, 90% acceptable
        elif success_rate >= 0.40:
            return 0.85  # Problematic month, relax to 85%
        else:
            return 0.80  # Very poor month, minimum 80%

    @staticmethod
    def _assess_quality(success_rate: float) -> str:
        """Legible evaluation of period quality"""
        if success_rate >= 0.80:
            return "EXCELLENT"
        elif success_rate >= 0.60:
            return "GOOD"
        elif success_rate >= 0.40:
            return "FAIR"
        else:
            return "POOR"

    def diagnose_period_quality(self) -> Dict:
        """
        Fast pre-diagnostic of period quality in cache.
        Tests 5 representative configs across sampled days.

        OPTIMIZATION: Unified loop (formerly 2 duplicated loops = 100 exp,
        now a single loop = 50 exp).
        """
        import random

        print(f"\n🔍 Running pre-optimization diagnostics on loaded period...")

        # Configuraciones representativas que cubren el espacio de búsqueda
        sample_configs = [
            {"exp_lambda": 0.99, "init_exp_T": 1000},   # High lambda, low T
            {"exp_lambda": 0.99, "init_exp_T": 2500},   # High lambda, high T
            {"exp_lambda": 0.95, "init_exp_T": 1000},   # Low lambda, low T
            {"exp_lambda": 0.95, "init_exp_T": 2500},   # Low lambda, high T
            {"exp_lambda": 0.975, "init_exp_T": 1500},  # Middle ground
        ]

        # Samplear hasta 10 días del mes.
        # REPRODUCIBILITY: fixed seed so the diagnostic day sample is stable
        # across runs of the same window (independent of Optuna's own seed).
        total_days = len(self.daily_data_cache)
        sample_size = min(10, total_days)
        dates_sorted = sorted(self.daily_data_cache.keys())
        sampled_dates = random.Random(self._diagnostic_seed).sample(dates_sorted, sample_size)

        print(f"   Testing {len(sample_configs)} configs on {sample_size} day sample...")

        # Loop UNIFICADO: trackear successes Y failed days en una sola pasada
        success_count = 0
        total_tests = len(sample_configs) * len(sampled_dates)
        failed_days_set = set()

        for config_idx, config in enumerate(sample_configs, 1):
            config_successes = 0
            for date in sampled_dates:
                day_cache = self.daily_data_cache[date]
                result = self.run_single_day_experiment(day_cache, config)
                if result is not None:
                    success_count += 1
                    config_successes += 1
                else:
                    failed_days_set.add(date)

            print(f"      Config {config_idx}: {config_successes}/{sample_size} days successful")

        # Calcular métricas
        success_rate = success_count / total_tests
        recommended_threshold = self._determine_threshold(success_rate)
        quality_assessment = self._assess_quality(success_rate)

        num_failed_unique_days = len(failed_days_set)
        failed_day_ratio = num_failed_unique_days / sample_size

        # Ajuste post-diagnostic: Si muchos días únicos fallan, bajar threshold
        if num_failed_unique_days >= 3 and recommended_threshold >= 0.90:
            original_threshold = recommended_threshold
            recommended_threshold = max(0.80, recommended_threshold * 0.85)
            print(f"      🔧 Threshold adjusted: {original_threshold:.0%} → {recommended_threshold:.0%}")
            print(f"         Reason: {num_failed_unique_days}/{sample_size} unique days have failures")

        diagnostics = {
            "success_rate": success_rate,
            "total_days_in_month": total_days,
            "sampled_days": sample_size,
            "recommended_threshold": recommended_threshold,
            "quality_assessment": quality_assessment,
            "configs_tested": len(sample_configs),
            "total_tests": total_tests,
            "successful_tests": success_count,
            "failed_unique_days": num_failed_unique_days,
            "failed_day_ratio": failed_day_ratio
        }

        # Mostrar resultados
        print(f"\n   📊 Diagnostic Results:")
        print(f"      Success Rate: {success_rate:.1%}")
        print(f"      Unique Failed Days: {num_failed_unique_days}/{sample_size}")
        print(f"      Quality: {quality_assessment}")
        print(f"      Recommended Threshold: {recommended_threshold:.0%}")

        if quality_assessment in ["POOR", "FAIR"]:
            print(f"      ⚠️  WARNING: Period has {quality_assessment.lower()} data quality")
            print(f"      Consider using previous month's parameters for production")

        return diagnostics

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function using compound IID-aligned score (D1).

        Score = quality (60%) + coverage (25%) + stability (15%)
        - quality: average statistical quality of bars (0-100)
          (low ACF, stationarity, Levene homoscedasticity, low CV ticks, normality)
        - coverage: successful days ratio, non-linear saturation (0-100)
        - stability: consistency of quality score across days (0-100)

        Hard limits: T > 11,000 rejected immediately. Lambda capped by symbol.
        Early stopping: If 10 consecutive trials are pruned, stop optimization.
        """

        # Definir hiperparámetros a probar
        params = self.define_search_space(trial)

        # HARD CAP: Reject T > 11,000 to prevent excessive bar width
        if params['init_exp_T'] > 11000:
            print(f"   Trial {trial.number}: ❌ T={params['init_exp_T']:,} exceeds 11,000 limit - skipped")
            return 0.0

        # Contadores y tracking detallado
        scores = []
        successful_days = 0
        failed_days = []
        total_bars = 0
        total_days = len(self.daily_data_cache)

        # Evaluar en cada día del cache
        # Early termination: si tras evaluar la mitad de los días
        # la completion rate ya es imposible de rescatar, abortar
        min_days_before_check = max(5, total_days // 2)

        for day_idx, (date_str, day_cache) in enumerate(self.daily_data_cache.items()):
            try:
                # Ejecutar experimento para este día (usa cache precomputado)
                result = self.run_single_day_experiment(day_cache, params)

                if result is None:
                    # Registrar día fallido con razón
                    failed_days.append({
                        "date": date_str,
                        "reason": "insufficient_bars",
                        "details": "Generated < 50 bars"
                    })

                    # Reportar 0 para pruning
                    trial.report(0.0, successful_days)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    continue

                quality_score = result["quality_score"]
                scores.append(quality_score)
                total_bars += result["n_bars"]
                successful_days += 1

                # Report intermediate value para pruning
                if successful_days > 0:
                    intermediate_value = np.mean(scores)
                    trial.report(intermediate_value, successful_days)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise

            except Exception as e:
                # Error inesperado, registrar
                failed_days.append({
                    "date": date_str,
                    "reason": "error",
                    "details": str(e)
                })
                continue

            # EARLY TERMINATION: verificar si es matemáticamente posible
            # alcanzar el umbral mínimo de 60%
            days_processed = day_idx + 1
            days_remaining = total_days - days_processed
            max_possible_success = successful_days + days_remaining
            if days_processed >= min_days_before_check and max_possible_success / total_days < 0.60:
                # Imposible alcanzar 60% → abortar temprano
                break

        # Calcular métricas
        completion_rate = successful_days / total_days if total_days > 0 else 0.0
        avg_bars_per_day = total_bars / successful_days if successful_days > 0 else 0.0

        # GC controlado: una vez por trial, no por día
        self._gc_maybe(trial.number)

        # Guardar información de fallos en el trial
        trial.set_user_attr("failed_days", failed_days)
        trial.set_user_attr("successful_days", successful_days)
        trial.set_user_attr("total_days", total_days)
        trial.set_user_attr("completion_rate", completion_rate)
        trial.set_user_attr("avg_bars_per_day", float(avg_bars_per_day))

        # Umbral mínimo absoluto: menos del 60% es inaceptable
        if completion_rate < 0.60:
            completion_pct = completion_rate * 100
            print(f"   Trial {trial.number}: ❌ Below minimum ({successful_days}/{total_days} days = {completion_pct:.0f}%) - lambda={params['exp_lambda']:.3f}, T={params['init_exp_T']}")
            return 0.0

        if len(scores) == 0:
            return 0.0

        # === D1: SCORE COMPUESTO (López de Prado IID-aligned) ===
        # 1. Quality (60%): average statistical quality of bars (0-100)
        mean_quality = np.mean(scores)
        quality_component = mean_quality  # ya está en 0-100

        # 2. Coverage (25%): ratio de cobertura con saturación no lineal
        #    cr^0.3 satura rápido: cr=0.80 → 93, cr=0.90 → 97, cr=0.95 → 98
        coverage_component = (completion_rate ** 0.3) * 100

        # 3. Stability (15%): consistencia entre días (bajo std = alta estabilidad)
        std_quality = np.std(scores) if len(scores) > 1 else 0.0
        stability_component = (1.0 / (1.0 + std_quality / 20.0)) * 100

        # Composite: 60% Quality + 25% Coverage + 15% Stability
        # (granularity term removed — the quality score already rewards IID properties:
        #  low ACF, stationarity, Levene homoscedasticity, low CV of ticks, normality)
        composite_score = (
            quality_component * 0.60 +
            coverage_component * 0.25 +
            stability_component * 0.15
        )

        # Guardar componentes para análisis posterior
        trial.set_user_attr("quality_component", float(quality_component))
        trial.set_user_attr("coverage_component", float(coverage_component))
        trial.set_user_attr("stability_component", float(stability_component))
        trial.set_user_attr("std_quality", float(std_quality))
        trial.set_user_attr("avg_bars_per_day", float(avg_bars_per_day))

        completion_pct = completion_rate * 100
        print(f"   Trial {trial.number}: ✅ ({successful_days}/{total_days} = {completion_pct:.0f}%) "
              f"Composite={composite_score:.2f} [Q={quality_component:.1f} C={coverage_component:.1f} S={stability_component:.1f}] "
              f"bars/day={avg_bars_per_day:.0f} "
              f"λ={params['exp_lambda']:.4f}, T={params['init_exp_T']}")

        return composite_score

    def _gc_maybe(self, trial_number: int):
        """GC every 10 trials instead of every day (reduces overhead ~5-8%)"""
        if trial_number % 10 == 0:
            gc.collect()

    def run_single_day_experiment(
        self, day_cache: Dict, params: Dict
    ) -> Dict:
        """
        Executes an experiment on a SINGLE day of data.

        OPTIMIZATION: Uses precomputed arrays (b_t, dollar_values) from cache
        to call _compute_drbs_core() directly, avoiding get_drbs()
        which would copy the DataFrame and recalucate tick directions.

        Also uses compute_quality_score_fast() tracking only the 5 metrics
        needed for scoring, omitting Shapiro-Wilk, max_drawdown, etc.
        """
        try:
            # Extraer arrays precomputados y DataFrame original
            b_t = day_cache["b_t"]
            dollar_values = day_cache["dollar_values"]
            df = day_cache["df"]

            # Llamar al kernel Numba directamente (SKIP: df.copy, get_tick_directions, dollar_value calc)
            bar_indices = _compute_drbs_core(
                b_t, dollar_values,
                params["exp_lambda"], params["init_exp_T"],
                BaseBars._get_min_ticks()
            )

            if len(bar_indices) < 50:
                return None

            # Formar barras directamente desde arrays (sin _form_bar ni iloc)
            # Extraer arrays raw del DataFrame para evitar pandas overhead
            prices = df["price"].values
            quantities = df["quantity"].values
            timestamps = df["timestamp"].values

            bars = []
            for start, end in bar_indices:
                p = prices[start:end]
                q = quantities[start:end]
                bars.append({
                    "open_time": timestamps[start],
                    "close_time": timestamps[end - 1],
                    "open": float(p[0]),
                    "high": float(p.max()),
                    "low": float(p.min()),
                    "close": float(p[-1]),
                    "n_ticks": end - start,
                    "volume": float(q.sum()),
                    "dollar_value": float(dollar_values[start:end].sum()),
                })
            bars_df = pd.DataFrame(bars)

            # Usar fast quality score (solo 5 métricas, ~2-3x más rápido)
            quality_score = self.stats_calculator.compute_quality_score_fast(bars_df)

            result = {
                "symbol": self.symbol,
                "params": params,
                "quality_score": quality_score,
                "n_bars": len(bars_df),
            }

            return result

        except Exception as e:
            print(f"      ❌ Error: {e}")
            return None

    @staticmethod
    def detect_regime_window(daily_data_cache: Dict, study_months: int = 3) -> int:
        """
        D3: Detects the market regime and determines how many months to utilize.
        Calculates volatility and volume ratios between months.

        Returns:
            int: Recommended months (1, 2, or 3)
        """
        if not daily_data_cache:
            return study_months

        # Agrupar días por mes
        monthly_stats = {}
        for date_str, day_cache in daily_data_cache.items():
            month_key = date_str[:7]  # "YYYY-MM"
            if month_key not in monthly_stats:
                monthly_stats[month_key] = {"dollar_volumes": [], "prices": []}

            # Use precomputed float64 array instead of recomputing from float32 df
            dollar_vol = day_cache["dollar_values"].sum()
            monthly_stats[month_key]["dollar_volumes"].append(dollar_vol)
            monthly_stats[month_key]["prices"].append(float(day_cache["df"]["price"].iloc[-1]))

        if len(monthly_stats) < 2:
            return 1  # Solo un mes de datos

        months_sorted = sorted(monthly_stats.keys())

        # Calcular volatilidad realizada y volumen promedio por mes
        vols = []
        avg_volumes = []
        for mk in months_sorted:
            prices = monthly_stats[mk]["prices"]
            returns = np.diff(np.log(prices)) if len(prices) > 1 else [0]
            vols.append(np.std(returns) * np.sqrt(365) if len(returns) > 0 else 0)
            avg_volumes.append(np.mean(monthly_stats[mk]["dollar_volumes"]))

        # Evitar división por cero
        min_vol = max(min(vols), 1e-10)
        min_volume = max(min(avg_volumes), 1e-10)

        vol_ratio = max(vols) / min_vol
        volume_ratio = max(avg_volumes) / min_volume

        print(f"\n   🔍 REGIME DETECTION:")
        print(f"      Months analyzed: {', '.join(months_sorted)}")
        print(f"      Volatility ratio:  {vol_ratio:.2f}x (threshold: 2.0x severe, 1.5x moderate)")
        print(f"      Volume ratio:      {volume_ratio:.2f}x (threshold: 3.0x severe, 2.0x moderate)")

        if vol_ratio > 2.0 or volume_ratio > 3.0:
            recommended = 1
            reason = "SEVERE regime change detected"
        elif vol_ratio > 1.5 or volume_ratio > 2.0:
            recommended = 2
            reason = "MODERATE regime change detected"
        else:
            recommended = min(study_months, len(monthly_stats))
            reason = "Stable regime"

        print(f"      → Recommended window: {recommended} month(s) ({reason})")

        return recommended

    def run_bayesian_optimization(
        self,
        year: int,
        month: int,
        window_months: int = 1,
        n_trials: int = 100,
        auto_window: bool = False,
        sampler_type: SamplerType = "botorch",
        device: DeviceType = "cuda",
        gpu_id: int = 0,
        cutoff_date: datetime | None = None,
    ) -> optuna.Study:
        """
        Ejecuta optimización bayesiana con Optuna.
        Incluye D1 (score compuesto), D2 (rangos adaptativos), D3 (ventana adaptativa).
        """
        # Calcular rango de fechas: cargar los meses PREVIOS al mes objetivo
        # Ejemplo: para optimizar Feb 2020 con ventana=2, cargar Dic 2019 - Ene 2020
        requested_window = window_months
        load_window = 3 if auto_window else window_months

        # end_date = mes ANTERIOR al objetivo (el último mes con datos disponibles)
        # para Feb 2020, el último mes con datos es Ene 2020
        target_month_start = datetime(year, month, 1)
        default_end_date = target_month_start - timedelta(days=1)

        if cutoff_date and cutoff_date >= target_month_start:
            end_date = cutoff_date
            start_date = target_month_start - pd.DateOffset(months=load_window - 1)
            cutoff_note = f" (cutoff: {end_date.date()})"
        else:
            end_date = default_end_date
            start_date = end_date - pd.DateOffset(months=load_window)
            cutoff_note = ""

        print(f"\n{'=' * 80}")
        print(f"🔬 BAYESIAN OPTIMIZATION v2: {self.symbol}")
        print(f"{'=' * 80}")
        print(f"Target Period: {year}-{month:02d}")
        print(f"Requested Window: {requested_window} months {'(AUTO)' if auto_window else ''}")
        print(f"Date Range (loading): {start_date.date()} to {end_date.date()}{cutoff_note}")
        print(f"Trials: {n_trials}")
        print(f"Search Bounds: T=[{self.search_bounds['T_min']}, {self.search_bounds['T_max']}], "
              f"λ=[{self.search_bounds['lambda_min']:.4f}, {self.search_bounds['lambda_max']:.4f}]")
        if self.search_bounds.get('derived_from_study'):
            print(f"   (Derived from previous study, center T={self.search_bounds.get('t_center', '?'):.0f})")
        print(f"{'=' * 80}\n")

        # Cargar todos los días del periodo en cache
        days_loaded = self.load_data_period(start_date, end_date)

        if days_loaded == 0:
            print("❌ No data available for this period")
            return None

        # D3: Detección de régimen para ventana adaptativa
        if auto_window:
            effective_window = self.detect_regime_window(self.daily_data_cache, load_window)

            # Si el régimen recomienda menos meses, filtrar el cache
            if effective_window < load_window:
                cutoff_date = target_month_start - pd.DateOffset(months=effective_window - 1)
                original_count = len(self.daily_data_cache)
                self.daily_data_cache = {
                    k: v for k, v in self.daily_data_cache.items()
                    if datetime.strptime(k, "%Y-%m-%d") >= cutoff_date
                }
                print(f"   📅 Window trimmed: {original_count} → {len(self.daily_data_cache)} days "
                      f"(using last {effective_window} month(s))")
            window_months = effective_window

        # Pre-optimization diagnostic
        self.diagnostics_results = self.diagnose_period_quality()

        # Set adaptive threshold
        self.completion_threshold = self.diagnostics_results['recommended_threshold']

        print(f"\n{'=' * 80}")
        print(f"📋 Optimization Configuration:")
        print(f"   Effective Window: {window_months} month(s) {'(auto-detected)' if auto_window else ''}")
        print(f"   Days in study: {len(self.daily_data_cache)}")
        print(f"   Completion Threshold: {self.completion_threshold:.0%}")
        print(f"   Period Quality: {self.diagnostics_results['quality_assessment']}")
        print(f"   Score: Composite (Quality 60% + Coverage 25% + Stability 15%)")
        print(f"{'=' * 80}\n")

        # Crear estudio Optuna con GPU/CPU support
        StudyFactory.print_config_info(sampler_type, device, gpu_id)

        try:
            # BoTorch: softer pruning (pruned trials damage surrogate model)
            # Allow ~half of n_trials to complete before MedianPruner activates
            pr_start = max(10, n_trials // 2) if sampler_type == "botorch" else 10
            pr_warm = max(5, pr_start // 2) if sampler_type == "botorch" else 5
            study = StudyFactory.create_study(
                study_name=f"{self.symbol}_{year}_{month:02d}",
                sampler_type=sampler_type,
                device=device,
                gpu_id=gpu_id,
                direction="maximize",
                seed=42,
                n_startup_trials=10,
                pruner_n_startup=pr_start,
                pruner_n_warmup=pr_warm,
            )
        except RuntimeError as e:
            print(f"\n❌ {e}")
            print("\n💡 Switching to CPU mode (TPE sampler)...")
            sampler_type = "tpe"
            device = "cpu"
            StudyFactory.print_config_info(sampler_type, device, gpu_id)
            study = StudyFactory.create_study(
                study_name=f"{self.symbol}_{year}_{month:02d}",
                sampler_type=sampler_type,
                device=device,
                gpu_id=gpu_id,
                direction="maximize",
                seed=42,
            )

        print(f"🚀 Starting optimization...\n")

        # Early stopping: pruned streak OR no-improvement streak
        early_stop_callback = EarlyStoppingCallback(n_pruned_consecutive=10, n_no_improvement=15)

        # Optimizar con early stopping
        study.optimize(self.objective, n_trials=n_trials, timeout=None, callbacks=[early_stop_callback])

        # Resultados
        print(f"\n{'=' * 80}")
        print(f"OPTIMIZATION COMPLETED")
        print(f"{'=' * 80}")
        print(f"Total trials: {len(study.trials)}")
        print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

        # NUEVO: Fallback automático (Opción 1)
        if study.best_value == 0.0 and self.completion_threshold >= 0.90:
            print(f"\n{'!' * 80}")
            print(f"⚠️  AUTOMATIC FALLBACK TRIGGERED")
            print(f"{'!' * 80}")
            print(f"No valid configurations found at {self.completion_threshold:.0%} threshold")

            # Bajar threshold a 85%
            original_threshold = self.completion_threshold
            self.completion_threshold = 0.85

            print(f"Lowering threshold: {original_threshold:.0%} → {self.completion_threshold:.0%}")
            print(f"Re-running optimization with {n_trials // 2} additional trials...\n")

            # Crear nuevo estudio para segunda ronda
            study2 = StudyFactory.create_study(
                study_name=f"{self.symbol}_{year}_{month:02d}_fallback",
                sampler_type=sampler_type,
                device=device,
                gpu_id=gpu_id,
                direction="maximize",
                seed=43,  # Different seed
                n_startup_trials=5,
                pruner_n_startup=5,
                pruner_n_warmup=3,
            )

            # Re-optimizar
            study2.optimize(self.objective, n_trials=n_trials // 2, timeout=None)

            # Merge results
            print(f"\n{'=' * 80}")
            print(f"FALLBACK OPTIMIZATION COMPLETED")
            print(f"{'=' * 80}")
            print(f"Additional trials: {len(study2.trials)}")
            print(f"Valid configs found: {len([t for t in study2.trials if t.value > 0])}")

            # Use the better study
            if study2.best_value > 0:
                study = study2
                print(f"✅ Using fallback results")
            else:
                print(f"❌ Fallback also failed - month may be incompatible")

        if study.best_trial and study.best_value > 0:
            bt = study.best_trial
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   exp_lambda: {study.best_params['exp_lambda']:.4f}")
            print(f"   init_exp_T: {study.best_params['init_exp_T']}")
            print(f"   Composite Score: {study.best_value:.2f}")
            print(f"     Quality:   {bt.user_attrs.get('quality_component', 0):.1f}")
            print(f"     Coverage:  {bt.user_attrs.get('coverage_component', 0):.1f}")
            print(f"     Stability: {bt.user_attrs.get('stability_component', 0):.1f}")
            print(f"   Completion: {bt.user_attrs.get('completion_rate', 0):.0%}")
        else:
            print(f"\n⚠️  No valid configurations found")
            print(f"   Recommendation: Use previous month's parameters for production")

        # Guardar estudio local (backup)
        study_path = self.results_dir / f"{self.symbol}_optuna_study.pkl"
        import joblib
        joblib.dump(study, study_path)
        print(f"\n💾 Study saved: {study_path}")

        # ── ESCRIBIR A TIMESCALEDB ──
        try:
            from src.storage.timescale_client import TimescaleDBClient
            db = TimescaleDBClient()
            period = f"{year}-{month:02d}"
            window_name = f"bayesian_{year}_{month:02d}_w{window_months}m"

            bt = study.best_trial
            if bt and bt.value > 0:
                db.upsert_study(
                    symbol=self.symbol,
                    period=period,
                    window_name=window_name,
                    exp_lambda=bt.params['exp_lambda'],
                    init_exp_t=bt.params['init_exp_T'],
                    composite_score=bt.value,
                    quality_component=bt.user_attrs.get('quality_component', 0),
                    coverage_component=bt.user_attrs.get('coverage_component', 0),
                    stability_component=bt.user_attrs.get('stability_component', 0),
                    granularity_component=bt.user_attrs.get('granularity_component', 0.0),
                    total_trials=len(study.trials),
                    completed_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                    pruned_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                    completion_rate=bt.user_attrs.get('completion_rate', 0.0),
                    avg_bars_per_day=bt.user_attrs.get('avg_bars_per_day', 0.0),
                    t_min=self.search_bounds.get('T_min', 1),
                    t_max=self.search_bounds.get('T_max', 10000),
                    lambda_min=self.search_bounds.get('lambda_min', 0.9),
                    lambda_max=self.search_bounds.get('lambda_max', 0.999),
                    sampler_type=sampler_type,
                    device=device,
                    search_bounds_json={k: v for k, v in self.search_bounds.items() if not k.startswith("_")},
                )
                print(f"   📊 Study synced to TimescaleDB: {self.symbol}/{period}")

            # Sync failed days
            failed_report = self.results_dir / f"{self.symbol}_failed_days_analysis.txt"
            if failed_report.exists():
                db.insert_failed_days(self.symbol, period, failed_report)

            db.close()
        except Exception as e:
            print(f"   ⚠️  DB sync failed (study preserved locally): {e}")

        # Guardar search bounds usados (para reproducibilidad y análisis)
        bounds_path = self.results_dir / f"{self.symbol}_search_bounds.yaml"
        with open(bounds_path, "w") as f:
            yaml.dump(self._sanitize_for_yaml({
                "symbol": self.symbol,
                "period": f"{year}-{month:02d}",
                "window_months": window_months,
                "auto_window": auto_window,
                "sampler_type": sampler_type,
                "device": device,
                "gpu_id": gpu_id,
                "search_bounds": {k: v for k, v in self.search_bounds.items()
                                  if not k.startswith("_")},
            }), f, default_flow_style=False, sort_keys=False)

        return study

    def generate_bayesian_report(self, study: optuna.Study, year: int, month: int, sampler_type: str = "botorch", device: str = "cuda"):
        """Genera reporte de optimización bayesiana con score compuesto (D1)"""
        report_path = self.results_dir / f"{self.symbol}_bayesian_report.txt"

        with open(report_path, "w") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{self.symbol} - BAYESIAN OPTIMIZATION REPORT v2\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Period: {year}-{month:02d}\n")
            f.write(f"Days in study: {len(self.daily_data_cache)}\n")
            f.write(f"Total trials: {len(study.trials)}\n")
            f.write(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
            f.write(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
            f.write(f"Sampler: {sampler_type.upper()}\n")
            f.write(f"Device: {device.upper()}\n")
            f.write(f"Score: Composite (Quality 60% + Coverage 25% + Stability 15%)\n\n")

            # Search bounds used
            f.write("=" * 80 + "\n")
            f.write("SEARCH BOUNDS:\n")
            f.write("=" * 80 + "\n")
            b = self.search_bounds
            f.write(f"T range: [{b['T_min']}, {b['T_max']}]\n")
            f.write(f"Lambda range: [{b['lambda_min']:.4f}, {b['lambda_max']:.4f}]\n")
            f.write(f"Source: {'Previous study (adaptive)' if b.get('derived_from_study') else 'Default bounds'}\n\n")

            # Diagnostic information
            if self.diagnostics_results:
                f.write("=" * 80 + "\n")
                f.write("MONTH QUALITY DIAGNOSTICS:\n")
                f.write("=" * 80 + "\n")
                f.write(f"Success Rate (Sample): {self.diagnostics_results['success_rate']:.1%}\n")
                f.write(f"Quality Assessment: {self.diagnostics_results['quality_assessment']}\n")
                f.write(f"Completion Threshold Used: {self.completion_threshold:.0%}\n")

                if self.diagnostics_results['quality_assessment'] in ["POOR", "FAIR"]:
                    f.write(f"\n⚠️  WARNING: Month has {self.diagnostics_results['quality_assessment'].lower()} data quality.\n")
                    f.write(f"Consider using previous month's parameters for production.\n")
                f.write("\n")

            if study.best_trial and study.best_value > 0:
                bt = study.best_trial
                f.write("=" * 80 + "\n")
                f.write("BEST CONFIGURATION:\n")
                f.write("=" * 80 + "\n")
                f.write(f"exp_lambda: {study.best_params['exp_lambda']:.4f}\n")
                f.write(f"init_exp_T: {study.best_params['init_exp_T']}\n")
                f.write(f"Composite Score: {study.best_value:.2f}\n")
                f.write(f"  Quality:   {bt.user_attrs.get('quality_component', 0):.1f}\n")
                f.write(f"  Coverage:  {bt.user_attrs.get('coverage_component', 0):.1f}\n")
                f.write(f"  Stability: {bt.user_attrs.get('stability_component', 0):.1f}\n")
                f.write(f"  Completion Rate: {bt.user_attrs.get('completion_rate', 0):.0%}\n\n")

                f.write("=" * 80 + "\n")
                f.write("TOP 10 CONFIGURATIONS:\n")
                f.write("=" * 80 + "\n\n")

                # Obtener trials completos ordenados
                complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value > 0]
                complete_trials.sort(key=lambda t: t.value, reverse=True)

                for i, trial in enumerate(complete_trials[:10], 1):
                    cr = trial.user_attrs.get('completion_rate', 0)
                    q = trial.user_attrs.get('quality_component', 0)
                    c = trial.user_attrs.get('coverage_component', 0)
                    s = trial.user_attrs.get('stability_component', 0)
                    f.write(f"RANK {i}:\n")
                    f.write(f"  Lambda: {trial.params['exp_lambda']:.4f}\n")
                    f.write(f"  T: {trial.params['init_exp_T']}\n")
                    f.write(f"  Composite: {trial.value:.2f} [Q={q:.1f} C={c:.1f} S={s:.1f}]\n")
                    f.write(f"  Completion: {cr:.0%}\n\n")
            else:
                f.write("\n⚠️  No valid configurations found.\n")
                f.write("All tested configurations failed minimum completion rate (60%).\n")

        print(f"📄 Bayesian report saved: {report_path}")

        # Generar reporte de días fallidos
        self._generate_failed_days_report(study, year, month)

    def _generate_failed_days_report(self, study: optuna.Study, year: int, month: int):
        """Genera reporte detallado de días que fallan consistentemente"""
        report_path = self.results_dir / f"{self.symbol}_failed_days_analysis.txt"

        # Recolectar información de todos los trials
        failed_days_counter = {}

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                failed_days = trial.user_attrs.get("failed_days", [])

                for failure in failed_days:
                    date = failure["date"]
                    reason = failure["reason"]

                    if date not in failed_days_counter:
                        failed_days_counter[date] = {
                            "insufficient_bars": 0,
                            "error": 0,
                            "no_data": 0,
                            "total_trials": 0,
                            "reasons": []
                        }

                    failed_days_counter[date][reason] += 1
                    failed_days_counter[date]["total_trials"] += 1

                    if failure["details"] not in failed_days_counter[date]["reasons"]:
                        failed_days_counter[date]["reasons"].append(failure["details"])

        # Escribir reporte
        with open(report_path, "w") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{self.symbol} - FAILED DAYS ANALYSIS\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Period: {year}-{month:02d}\n")
            f.write(f"Total days in cache: {len(self.daily_data_cache)}\n")
            f.write(f"Days with failures: {len(failed_days_counter)}\n\n")

            if failed_days_counter:
                f.write("=" * 80 + "\n")
                f.write("DAYS THAT CONSISTENTLY FAIL:\n")
                f.write("=" * 80 + "\n\n")

                # Ordenar por número de fallos (descendente)
                sorted_failures = sorted(
                    failed_days_counter.items(),
                    key=lambda x: x[1]["total_trials"],
                    reverse=True
                )

                for date, info in sorted_failures:
                    f.write(f"DATE: {date}\n")
                    f.write(f"  Failed {info['total_trials']} times across all trials\n")
                    f.write(f"  Reasons:\n")
                    f.write(f"    - Insufficient bars: {info['insufficient_bars']}\n")
                    f.write(f"    - Errors: {info['error']}\n")
                    f.write(f"  Details: {', '.join(set(info['reasons']))}\n\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("SUMMARY:\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Most problematic days (top 5):\n")
                for date, info in sorted_failures[:5]:
                    f.write(f"  - {date}: Failed {info['total_trials']} times\n")
            else:
                f.write("\n✅ No consistent failures detected.\n")

        print(f"📊 Failed days analysis saved: {report_path}")


    def run_month_experiment(
        self,
        year: int,
        month: int,
        param_grid: List[Dict],
    ) -> pd.DataFrame:
        """
        Ejecuta grid search para todo el mes
        """
        print(f"\n{'=' * 80}")
        print(f"🔬 Hyperparameter Tuning for {self.symbol}")
        print(f"{'=' * 80}")
        print(f"Period: {year}-{month:02d}")
        print(f"Configurations: {len(param_grid)}")
        print(f"{'=' * 80}\n")

        all_results = []

        # Generar fechas del mes
        from datetime import datetime, timedelta
        start_date = datetime(year, month, 1)

        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)

        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        print(f"📅 Processing {len(dates)} days\n")

        # Procesar cada día
        for day_idx, date_str in enumerate(dates, 1):
            print(f"[{day_idx}/{len(dates)}] {date_str}... ", end="", flush=True)

            # Descargar raw data
            try:
                df_raw = self.downloader.download_day(date_str)

                if df_raw is None or len(df_raw) == 0:
                    print("⚠️  No data")
                    continue

                # LIMPIEZA DE DATOS
                initial_len = len(df_raw)
                df_raw = self.normalizer.clean_raw_ticks(df_raw)

                if len(df_raw) == 0:
                    print("⚠️  No valid data after cleaning")
                    continue

                # Columnas mínimas
                needed_cols = ['timestamp', 'price', 'quantity']
                df_raw = df_raw[needed_cols].copy()

                # Precomputar arrays para el nuevo formato de cache
                base_bars = BaseBars()
                b_t_arr = base_bars.get_tick_directions(df_raw).values.astype(np.float64)
                dollar_values_arr = (df_raw["price"].values * df_raw["quantity"].values).astype(np.float64)
                df_raw["price"] = df_raw["price"].astype(np.float32)
                df_raw["quantity"] = df_raw["quantity"].astype(np.float32)
                day_cache = {
                    "df": df_raw,
                    "b_t": b_t_arr,
                    "dollar_values": dollar_values_arr,
                }

                print(f"✅ {len(df_raw):,} ticks", end=" | ")

                # Probar cada configuración
                day_results = []
                for params in param_grid:
                    result = self.run_single_day_experiment(day_cache, params)
                    if result is not None:
                        result["day"] = date_str
                        day_results.append(result)

                all_results.extend(day_results)
                print(f"{len(day_results)} configs OK")

                # Liberar memoria
                del df_raw, day_cache
                gc.collect()

                # Checkpoint cada 5 días
                if day_idx % 5 == 0:
                    self._save_checkpoint(all_results, day_idx)

            except Exception as e:
                print(f"❌ Error: {e}")
                continue

        # Checkpoint final
        self._save_checkpoint(all_results, len(dates))

        print(f"\n✅ Month completed: {len(all_results)} total results")

        # Guardar resultados detallados día por día
        self.save_detailed_results(all_results)

        # Agregar resultados
        results_df = self._aggregate_results(all_results, expected_days=len(dates))

        return results_df

    def _aggregate_results(self, all_results: List[Dict], expected_days: int) -> pd.DataFrame:
        """Agrega resultados de múltiples días"""

        # Manejar caso de resultados vacíos
        if not all_results:
            print("⚠️  No results to aggregate (no data available for this symbol)")
            return pd.DataFrame()

        # Convertir a DataFrame
        df = pd.DataFrame(
            [
                {
                    "symbol": r["symbol"],
                    "exp_lambda": r["params"]["exp_lambda"],
                    "init_exp_T": r["params"]["init_exp_T"],
                    "day": r["day"],
                    "quality_score": r["quality_score"],
                    **r["stats"],
                }
                for r in all_results
            ]
        )

        # Agrupar por configuración
        grouped = (
            df.groupby(["symbol", "exp_lambda", "init_exp_T"])
            .agg(
                {
                    "quality_score": ["mean", "std", "count"],
                    "abs_acf_lag1": "mean",
                    "cv_ticks": "mean",
                    "n_bars": ["mean", "std"],
                    "is_stationary": lambda x: x.sum() / len(x),
                    "std_return": "mean",
                    "sharpe_ratio": "mean",
                }
            )
            .reset_index()
        )

        # Aplanar columnas
        grouped.columns = [
            "symbol",
            "exp_lambda",
            "init_exp_T",
            "quality_score_mean",
            "quality_score_std",
            "n_days",
            "abs_acf_lag1",
            "cv_ticks",
            "n_bars_mean",
            "n_bars_std",
            "stationary_ratio",
            "std_return",
            "sharpe_ratio",
        ]

        # Calcular ratio de completitud
        grouped["completion_rate"] = grouped["n_days"] / expected_days

        return grouped

    def save_detailed_results(self, all_results: List[Dict]):
        """Guarda resultados detallados día por día"""
        if not all_results:
            return

        # Convertir a DataFrame con todas las métricas por día
        detailed_df = pd.DataFrame([
            {
                "symbol": r["symbol"],
                "day": r["day"],
                "exp_lambda": r["params"]["exp_lambda"],
                "init_exp_T": r["params"]["init_exp_T"],
                "n_bars": r["stats"]["n_bars"],
                "quality_score": r["quality_score"],
                "abs_acf_lag1": r["stats"]["abs_acf_lag1"],
                "cv_ticks": r["stats"]["cv_ticks"],
                "mean_ticks": r["stats"]["mean_ticks"],
                "is_stationary": r["stats"]["is_stationary"],
                "sharpe_ratio": r["stats"]["sharpe_ratio"],
                "total_volume": r["stats"]["total_volume"],
            }
            for r in all_results
        ])

        # Guardar archivo detallado
        detailed_path = self.results_dir / f"{self.symbol}_detailed_results.csv"
        detailed_df.to_csv(detailed_path, index=False)
        print(f"💾 Detailed results saved: {detailed_path}")

    def _save_checkpoint(self, results: List[Dict], day_count: int):
        """Guarda checkpoint"""
        checkpoint_file = self.results_dir / f"checkpoint_day{day_count}.json"

        with open(checkpoint_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"   💾 Checkpoint: day {day_count}")

    def save_results(self, results_df: pd.DataFrame):
        """Guarda resultados finales"""
        csv_path = self.results_dir / f"{self.symbol}_DRBs_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved: {csv_path}")

    def generate_report(self, results_df: pd.DataFrame):
        """Genera reporte final con múltiples lambdas"""
        report_path = self.results_dir / f"{self.symbol}_report.txt"

        with open(report_path, "w") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{self.symbol} - DRBs HYPERPARAMETER TUNING REPORT\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Lambda values tested: {sorted(results_df['exp_lambda'].unique())}\n")
            f.write(f"T values tested: {sorted(results_df['init_exp_T'].unique())}\n")
            f.write(f"Days processed: {results_df['n_days'].iloc[0]:.0f}\n\n")

            # Overall best configuration
            f.write("=" * 80 + "\n")
            f.write("ALL CONFIGURATIONS RANKED (by quality score, prioritized by completion):\n")
            f.write("=" * 80 + "\n\n")

            # Priorizar configuraciones completas (100% de días)
            # Primero ordenar por completion_rate (desc) y luego por quality_score (desc)
            sorted_results = results_df.sort_values(
                ["completion_rate", "quality_score_mean"],
                ascending=[False, False]
            )

            for i, (_, row) in enumerate(sorted_results.iterrows(), 1):
                f.write(f"RANK {i}:\n")
                f.write(f"  Lambda={row['exp_lambda']}, T={row['init_exp_T']}\n")
                f.write(
                    f"  Quality Score: {row['quality_score_mean']:.2f} ± {row['quality_score_std']:.2f}\n"
                )
                f.write(f"  Completion: {row['completion_rate']:.1%} ({row['n_days']}/{results_df['n_days'].max():.0f} days)\n")
                f.write(f"  |ACF(1)|: {row['abs_acf_lag1']:.4f}\n")
                f.write(f"  CV Ticks: {row['cv_ticks']:.4f}\n")
                f.write(f"  Stationary: {row['stationary_ratio']:.0%}\n")
                f.write(f"  Avg Bars/Day: {row['n_bars_mean']:.0f}\n\n")

            # Best configuration per Lambda
            f.write("=" * 80 + "\n")
            f.write("BEST CONFIGURATION PER LAMBDA:\n")
            f.write("=" * 80 + "\n\n")

            for lam in sorted(results_df['exp_lambda'].unique()):
                subset = results_df[results_df['exp_lambda'] == lam].copy()

                # Intentar filtrar solo completas primero
                complete_subset = subset[subset['completion_rate'] == 1.0]
                if not complete_subset.empty:
                    subset = complete_subset

                subset = subset.sort_values('quality_score_mean', ascending=False)
                best = subset.iloc[0]
                f.write(f"Lambda = {lam}:\n")
                f.write(f"  Best T: {best['init_exp_T']}\n")
                f.write(f"  Quality Score: {best['quality_score_mean']:.2f} ± {best['quality_score_std']:.2f}\n")
                f.write(f"  Completion: {best['completion_rate']:.1%}\n")
                f.write(f"  Avg Bars/Day: {best['n_bars_mean']:.0f}\n\n")

        print(f"📄 Report saved: {report_path}")

    def generate_advanced_report(self, results_df: pd.DataFrame):
        """Genera un reporte avanzado enfocado en viabilidad de trading"""
        report_path = self.results_dir / f"{self.symbol}_trading_viability_report.txt"

        # Calcular Trading Score
        # 1. Estabilidad de conteo de barras (CV de n_bars)
        # Rellenar NaN en std con 0 (si solo hay 1 día)
        results_df["n_bars_std"] = results_df["n_bars_std"].fillna(0)
        results_df["bar_cv"] = results_df["n_bars_std"] / results_df["n_bars_mean"]
        results_df["stability_score"] = (1 - results_df["bar_cv"].clip(0, 1)) * 100

        # 2. Puntuación de Volumen (Penalizar < 50 barras, premiar hasta 300)
        # Target ideal: ~300 barras/día (suficiente para intraday, no HFT excesivo)
        results_df["volume_score"] = (results_df["n_bars_mean"] / 300).clip(0, 1) * 100
        results_df.loc[results_df["n_bars_mean"] < 50, "volume_score"] = 0 # Penalización dura por falta de datos

        # 3. Trading Score Ponderado
        # 40% Calidad Estadística (Ruido blanco - lo que ya medíamos)
        # 30% Estabilidad (Consistencia día a día - CRUCIAL para producción)
        # 20% Volumen (Suficiente data para operar)
        # 10% Sharpe (Señal base de rentabilidad/riesgo)
        results_df["trading_score"] = (
            0.40 * results_df["quality_score_mean"] +
            0.30 * results_df["stability_score"] +
            0.20 * results_df["volume_score"] +
            0.10 * (results_df["sharpe_ratio"] * 20).clip(0, 100) # Sharpe 5 = 100 pts
        )

        results_df = results_df.sort_values("trading_score", ascending=False)

        with open(report_path, "w") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{self.symbol} - ADVANCED TRADING VIABILITY REPORT\n")
            f.write(f"{'=' * 80}\n\n")
            f.write("METRICS EXPLAINED:\n")
            f.write("- Trading Score: Weighted average of Statistical Quality, Stability, and Volume.\n")
            f.write("- Stability Score: Consistency of bar counts across days (Higher is better).\n")
            f.write("- Volume Score: Rewards having enough bars (target ~300/day). <50 gets 0.\n")
            f.write("- Stat Quality: The original statistical score (Normality, Stationarity, etc).\n\n")

            f.write(f"{'=' * 80}\n")
            f.write("TOP CONFIGURATIONS FOR TRADING:\n")
            f.write(f"{'=' * 80}\n\n")

            for i, (_, row) in enumerate(results_df.head(15).iterrows(), 1):
                f.write(f"RANK {i}: Lambda={row['exp_lambda']}, T={row['init_exp_T']}\n")
                f.write(f"  🏆 TRADING SCORE:   {row['trading_score']:.2f} / 100\n")
                f.write(f"  --------------------------------------------------\n")
                f.write(f"  📊 Stat Quality:    {row['quality_score_mean']:.2f} (Weight: 40%)\n")
                f.write(f"  ⚖️  Stability:       {row['stability_score']:.2f} (CV: {row['bar_cv']:.2f}) (Weight: 30%)\n")
                f.write(f"  📈 Volume Score:    {row['volume_score']:.2f} (Avg Bars: {row['n_bars_mean']:.0f}) (Weight: 20%)\n")
                f.write(f"  💰 Sharpe Ratio:    {row['sharpe_ratio']:.2f} (Weight: 10%)\n")
                f.write(f"  ✅ Completion:      {row['completion_rate']:.1%}\n\n")

        print(f"📄 Advanced Report saved: {report_path}")


def _process_symbol(symbol, results_dir, prev_study_dir, year, month, window, n_trials, auto_window, sampler_type="botorch", device="cuda", gpu_id=0, source="binance", data_dir="data_raw", cutoff_date: datetime | None = None):
    """Función helper para procesar un símbolo (usada en modo paralelo y secuencial)"""
    print(f"\n{'#' * 80}")
    print(f"# PROCESSING: {symbol}")
    print(f"{'#' * 80}\n")

    experiment = MultiAssetExperiment(
        symbol=symbol,
        results_dir=results_dir,
        prev_study_dir=prev_study_dir,  # D2: adaptive bounds
        source=source,
        data_dir=data_dir,
    )

    # Ejecutar optimización bayesiana
    study = experiment.run_bayesian_optimization(
        year=year,
        month=month,
        window_months=window,
        n_trials=n_trials,
        auto_window=auto_window,  # D3: regime detection
        sampler_type=sampler_type,
        device=device,
        gpu_id=gpu_id,
        cutoff_date=cutoff_date,
    )

    # Generar reporte si hubo resultados válidos
    if study and study.best_trial and study.best_value > 0:
        experiment.generate_bayesian_report(study, year, month, sampler_type, device)
    else:
        print(f"⚠️  No valid configurations found for {symbol}")

    # Liberar memoria
    del experiment
    del study
    gc.collect()

    print(f"\n✅ {symbol} COMPLETED\n")
    return symbol


def main():
    """Script principal con Optimización Bayesiana v2 (D1 + D2 + D3) + Performance Optimizations"""

    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Asset Hyperparameter Tuning v2 - Composite Score + Adaptive Bounds + Regime Detection"
    )
    parser.add_argument("--year", type=int, required=True, help="Year to process (e.g., 2024)")
    parser.add_argument("--month", type=int, required=True, help="Month to process (1-12)")
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of Optuna trials to run (default: 100)"
    )
    parser.add_argument(
        "--window",
        type=str,
        default="2",
        help="Window size in months (1-3) or 'auto' for regime-based detection (default: 2)"
    )
    parser.add_argument(
        "--prev-study-dir",
        type=str,
        default=None,
        help="Path to previous study results dir for adaptive bounds (e.g., experiments/bayesian_2024_12)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of symbols to process in parallel (default: 1 = sequential). "
             "Requires ~1-2GB RAM per symbol. Use 2-4 if RAM allows."
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["tpe", "botorch"],
        default="botorch",
        help="Optimization sampler: 'botorch' (GPU Bayesian Optimization, default) or 'tpe' (CPU Tree-structured Parzen Estimator)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Computation device: 'cuda' (GPU, default) or 'cpu'. Falls back to CPU if GPU unavailable."
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0). Use nvidia-smi to list available GPUs."
    )
    parser.add_argument(
        "--show-gpu-info",
        action="store_true",
        help="Display GPU information and exit (for diagnostics)"
    )
    parser.add_argument(
        "--symbols", type=str, default="BTCUSDT",
        help="Comma-separated symbols to process (e.g., BTCUSDT,ETHUSDT)"
    )
    parser.add_argument(
        "--source", type=str, default="auto", choices=["binance", "local", "auto"],
        help="Data source: binance (download), local (parquet), or auto (local + fallback)"
    )
    parser.add_argument(
        "--data-dir", default="data_raw",
        help="Directory where raw tick parquet files are stored (for --source local)"
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Optional cutoff date (YYYY-MM-DD) to allow partial windows"
    )

    args = parser.parse_args()

    # Mostrar info GPU si se solicitó
    if args.show_gpu_info:
        from src.optimization.samplers.gpu_utils import GPUDetector
        GPUDetector.print_gpu_info(args.gpu_id)
        print("\n" + GPUDetector.get_installation_instructions())
        return

    # Configuración
    YEAR = args.year
    MONTH = args.month
    N_TRIALS = args.trials
    AUTO_WINDOW = args.window.lower() == "auto"
    WINDOW = 3 if AUTO_WINDOW else int(args.window)
    PREV_STUDY_DIR = Path(args.prev_study_dir) if args.prev_study_dir else None
    PARALLEL = max(1, args.parallel)
    SAMPLER_TYPE = args.sampler
    DEVICE = args.device
    GPU_ID = args.gpu_id
    SOURCE = args.source
    DATA_DIR = args.data_dir
    CUTOFF_DATE = datetime.fromisoformat(args.cutoff_date) if args.cutoff_date else None

    # Ajustar nombre de directorio
    window_label = "auto" if AUTO_WINDOW else f"{WINDOW}"
    if WINDOW > 1 or AUTO_WINDOW:
        dir_name = f"bayesian_{YEAR}_{MONTH:02d}_w{window_label}m"
    else:
        dir_name = f"bayesian_{YEAR}_{MONTH:02d}"

    RESULTS_DIR = Path(f"experiments/{dir_name}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Símbolos a procesar
    # SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
    SYMBOLS = args.symbols.split(',') if hasattr(args, 'symbols') and args.symbols else ["BTCUSDT"]

    print("\n" + "🚀" * 40)
    print("BAYESIAN OPTIMIZATION v2 - MULTI-ASSET HYPERPARAMETER TUNING")
    print("🚀" * 40)
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Target Period: {YEAR}-{MONTH:02d}")
    print(f"Window: {window_label} month(s)")
    print(f"Trials per symbol: {N_TRIALS}")
    print(f"Parallel workers: {PARALLEL}")
    print(f"Sampler: {SAMPLER_TYPE.upper()}")
    print(f"Device: {DEVICE.upper()}")
    print(f"Source: {SOURCE}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Output Dir: {RESULTS_DIR}")
    print(f"Previous Study: {PREV_STUDY_DIR or 'None (using default bounds)'}")
    if CUTOFF_DATE:
        print(f"Cutoff Date: {CUTOFF_DATE.date()}")
    print(f"Score: Composite (Quality 60% + Coverage 25% + Stability 15%)")
    print("=" * 80 + "\n")

    if PARALLEL > 1:
        # Procesamiento paralelo con ProcessPoolExecutor
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"🔀 Running {PARALLEL} symbols in parallel...\n")

        with ProcessPoolExecutor(max_workers=PARALLEL) as executor:
            futures = {
                executor.submit(
                    _process_symbol, symbol, RESULTS_DIR, PREV_STUDY_DIR,
                    YEAR, MONTH, WINDOW, N_TRIALS, AUTO_WINDOW, SAMPLER_TYPE, DEVICE, GPU_ID,
                    SOURCE, DATA_DIR, CUTOFF_DATE
                ): symbol
                for symbol in SYMBOLS
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ {symbol} FAILED: {e}")
    else:
        # Procesamiento secuencial (por defecto)
        for symbol in SYMBOLS:
            _process_symbol(
                symbol, RESULTS_DIR, PREV_STUDY_DIR,
                YEAR, MONTH, WINDOW, N_TRIALS, AUTO_WINDOW,
                SAMPLER_TYPE, DEVICE, GPU_ID,
                SOURCE, DATA_DIR, CUTOFF_DATE
            )

    print("\n" + "🎉" * 40)
    print("✅ ALL ASSETS COMPLETED!")
    print("🎉" * 40)


if __name__ == "__main__":
    main()
