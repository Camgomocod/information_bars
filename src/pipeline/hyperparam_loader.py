"""
HyperparamLoader — Walk-Forward Hyperparameter Loader

Logic:
  Experiments are bimonthly: bayesian_YYYY_MM_w2m covers 2 months ENDING at YYYY-MM.
  For a given target (symbol, year, month), we load the best params from the most
  recent experiment whose END period is STRICTLY BEFORE the target month.

  Example:
    bayesian_2022_12_w2m  → covers Nov-Dec 2022 → used for Jan & Feb 2023
    bayesian_2023_01_w2m  → covers Dec 2022 - Jan 2023 → used for Feb & Mar 2023
    bayesian_2023_03_w2m  → covers Feb-Mar 2023 → used for Apr & May 2023
    ...

  The study for a given month M is any study whose period month < target month M.
  We pick the latest one (most recent prior study).
"""

import re
from pathlib import Path
from datetime import date
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import joblib


# Default fallback bounds per symbol (loaded from config/*.yaml)
# Lazily populated on first access to avoid circular imports
_DEFAULT_PARAMS: Dict[str, Dict] = {}
_DEFAULT_PARAMS_LOADED = False


def _ensure_default_params():
    global _DEFAULT_PARAMS, _DEFAULT_PARAMS_LOADED
    if _DEFAULT_PARAMS_LOADED:
        return
    try:
        from src.config import get_default_hyperparams
        _DEFAULT_PARAMS = {
            "BTCUSDT": get_default_hyperparams("BTCUSDT"),
            "ETHUSDT": get_default_hyperparams("ETHUSDT"),
            "BNBUSDT": get_default_hyperparams("BNBUSDT"),
            "SOLUSDT": get_default_hyperparams("SOLUSDT"),
            "_default": get_default_hyperparams(),
        }
    except Exception:
        _DEFAULT_PARAMS = {
            "BTCUSDT": {"exp_lambda": 0.9975, "init_exp_T": 14738},
            "ETHUSDT": {"exp_lambda": 0.9975, "init_exp_T": 5000},
            "BNBUSDT": {"exp_lambda": 0.9975, "init_exp_T": 500},
            "SOLUSDT": {"exp_lambda": 0.9975, "init_exp_T": 2000},
            "_default": {"exp_lambda": 0.9975, "init_exp_T": 2000},
        }
    _DEFAULT_PARAMS_LOADED = True


def _get_default(symbol: str) -> Dict:
    _ensure_default_params()
    return _DEFAULT_PARAMS.get(symbol, _DEFAULT_PARAMS["_default"])

# Pattern: bayesian_YYYY_MM_w2m
_STUDY_DIR_RE = re.compile(r"^bayesian_(\d{4})_(\d{2})_w2m$")


def _parse_study_dirs(experiments_dir: Path) -> list[Tuple[date, Path]]:
    """
    Scans experiments_dir for bayesian_YYYY_MM_w2m folders.
    Returns sorted list of (period_end_date, folder_path).
    period_end is the LAST day of YYYY-MM (represented as the 1st for comparison).
    Returns empty list if directory does not exist.
    """
    if not experiments_dir.exists():
        return []
    entries = []
    for folder in experiments_dir.iterdir():
        if not folder.is_dir():
            continue
        m = _STUDY_DIR_RE.match(folder.name)
        if m:
            year, month = int(m.group(1)), int(m.group(2))
            # Use the 1st of YYYY-MM as the period end marker for comparison
            period_end = date(year, month, 1)
            entries.append((period_end, folder))
    return sorted(entries, key=lambda x: x[0])


@dataclass
class HyperparamsWithMeta:
    """Hyperparams + metadata from the study used."""
    exp_lambda: float
    init_exp_T: int
    study_name: str
    completion_rate: float
    failed_days: List[str]  # List of dates (YYYY-MM-DD) that failed in the study


def _load_failed_days_from_analysis(study_folder: Path, symbol: str) -> Tuple[List[str], float]:
    """
    Parse the failed_days_analysis.txt file to extract failed dates.
    Returns: (list_of_failed_dates, completion_rate)
    """
    analysis_file = study_folder / symbol / f"{symbol}_failed_days_analysis.txt"

    if not analysis_file.exists():
        return [], 1.0

    try:
        content = analysis_file.read_text()
        failed_dates = []
        completion_rate = 1.0

        # Parse "DATE: YYYY-MM-DD" lines
        for line in content.split('\n'):
            if line.startswith('DATE:'):
                date_str = line.split('DATE:')[1].strip()
                failed_dates.append(date_str)

        # Try to parse completion rate from best trial or report
        # This is an approximation - the actual rate is in the study
        return failed_dates, completion_rate

    except Exception:
        return [], 1.0


class HyperparamLoader:
    """
    Walk-forward loader for Optuna-optimized hyperparameters.

    Usage:
        loader = HyperparamLoader("experiments/")
        params = loader.get_params("BTCUSDT", 2023, 3)
        # → {"exp_lambda": ..., "init_exp_T": ...}

        # Or get enriched version with failed days:
        enriched = loader.get_params_with_meta("BTCUSDT", 2023, 3)
        # → HyperparamsWithMeta with failed_days list
    """

    def __init__(self, experiments_dir: str | Path):
        self.experiments_dir = Path(experiments_dir)
        self._study_dirs = _parse_study_dirs(self.experiments_dir)

    def get_params(
        self,
        symbol: str,
        target_year: int,
        target_month: int,
        verbose: bool = True,
    ) -> Dict:
        """
        Returns the best hyperparams for (symbol, target_year, target_month)
        by loading the most recent Optuna study whose period ends BEFORE the target month.

        Returns a dict: {"exp_lambda": float, "init_exp_T": int}
        """
        target_date = date(target_year, target_month, 1)

        # Find the most recent prior study with a valid .pkl for this symbol
        # Walk backwards through sorted study dirs until we find one with data
        best_folder: Optional[Path] = None
        best_period: Optional[date] = None

        for period_end, folder in reversed(self._study_dirs):
            if period_end < target_date:
                study_pkl = folder / symbol / f"{symbol}_optuna_study.pkl"
                if study_pkl.exists():
                    best_folder = folder
                    best_period = period_end
                    break  # Found the most recent valid study

        if best_folder is None:
            params = _get_default(symbol)
            if verbose:
                print(
                    f"   📐 [{symbol}] {target_year}-{target_month:02d}: "
                    f"No prior study found → using defaults: {params}"
                )
            result = dict(params)
            result["study_name"] = "default"
            return result

        # Load Optuna study for this symbol
        study_pkl = best_folder / symbol / f"{symbol}_optuna_study.pkl"

        try:
            study = joblib.load(study_pkl)
            best_trial = study.best_trial
            params = {
                "exp_lambda": best_trial.params["exp_lambda"],
                "init_exp_T": best_trial.params["init_exp_T"],
                "study_name": best_folder.name,
            }
            if verbose:
                print(
                    f"   ✅ [{symbol}] {target_year}-{target_month:02d}: "
                    f"Using {best_folder.name} "
                    f"(λ={params['exp_lambda']:.4f}, T={params['init_exp_T']}) "
                    f"score={best_trial.value:.2f}"
                )
            return params

        except Exception as e:
            params = _get_default(symbol)
            if verbose:
                print(
                    f"   ❌ [{symbol}] {target_year}-{target_month:02d}: "
                    f"Error loading {study_pkl.name}: {e} → using defaults"
                )
            result = dict(params)
            result["study_name"] = "error"
            return result

    def get_params_with_meta(
        self,
        symbol: str,
        target_year: int,
        target_month: int,
        verbose: bool = True,
    ) -> HyperparamsWithMeta:
        """
        Returns hyperparams with metadata including failed days from the study.

        This is the enriched version that includes:
        - exp_lambda, init_exp_T (hyperparams)
        - study_name (e.g., "bayesian_2023_01_w2m")
        - completion_rate (from best trial)
        - failed_days (list of dates that failed in the study)
        """
        target_date = date(target_year, target_month, 1)

        # Find the most recent prior study with a valid .pkl for this symbol
        # Walk backwards through sorted study dirs until we find one with data
        best_folder: Optional[Path] = None
        best_period: Optional[date] = None

        for period_end, folder in reversed(self._study_dirs):
            if period_end < target_date:
                study_pkl = folder / symbol / f"{symbol}_optuna_study.pkl"
                if study_pkl.exists():
                    best_folder = folder
                    best_period = period_end
                    break  # Found the most recent valid study

        if best_folder is None:
            params = _get_default(symbol)
            if verbose:
                print(
                    f"   📐 [{symbol}] {target_year}-{target_month:02d}: "
                    f"No prior study found → using defaults"
                )
            return HyperparamsWithMeta(
                exp_lambda=params["exp_lambda"],
                init_exp_T=params["init_exp_T"],
                study_name="default",
                completion_rate=1.0,
                failed_days=[],
            )

        study_pkl = best_folder / symbol / f"{symbol}_optuna_study.pkl"

        try:
            study = joblib.load(study_pkl)
            best_trial = study.best_trial

            completion_rate = best_trial.user_attrs.get("completion_rate", 1.0)

            failed_days = best_trial.user_attrs.get("failed_days", [])
            failed_dates = [fd.get("date", "") for fd in failed_days if fd.get("date")]

            file_failed_dates, _ = _load_failed_days_from_analysis(best_folder, symbol)
            all_failed_dates = list(set(failed_dates + file_failed_dates))

            if verbose:
                print(
                    f"   ✅ [{symbol}] {target_year}-{target_month:02d}: "
                    f"Using {best_folder.name} "
                    f"(λ={best_trial.params['exp_lambda']:.4f}, T={best_trial.params['init_exp_T']}) "
                    f"score={best_trial.value:.2f} | failed_days={len(all_failed_dates)}"
                )

            return HyperparamsWithMeta(
                exp_lambda=best_trial.params["exp_lambda"],
                init_exp_T=best_trial.params["init_exp_T"],
                study_name=best_folder.name,
                completion_rate=completion_rate,
                failed_days=all_failed_dates,
            )

        except Exception as e:
            params = _get_default(symbol)
            if verbose:
                print(
                    f"   ❌ [{symbol}] {target_year}-{target_month:02d}: "
                    f"Error loading {study_pkl.name}: {e} → using defaults"
                )
            return HyperparamsWithMeta(
                exp_lambda=params["exp_lambda"],
                init_exp_T=params["init_exp_T"],
                study_name="error",
                completion_rate=1.0,
                failed_days=[],
            )
