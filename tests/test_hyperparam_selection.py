"""
Tests for strict walk-forward parameter selection (no-lookahead).

A study bayesian_YYYY_MM_w2m covers up to YYYY-MM and may only be used for
months STRICTLY AFTER it. It must never be used for its own period.
"""

from datetime import date
from pathlib import Path

import joblib
import optuna
import pytest

from src.pipeline.hyperparam_loader import (
    HyperparamLoader,
    _parse_study_dirs,
)


def _fake_study(exp_lambda: float, init_exp_T: int) -> optuna.Study:
    study = optuna.create_study(direction="maximize")
    trial = optuna.create_trial(
        params={"exp_lambda": exp_lambda, "init_exp_T": init_exp_T},
        distributions={
            "exp_lambda": optuna.distributions.FloatDistribution(0.9, 0.999),
            "init_exp_T": optuna.distributions.IntDistribution(100, 30000),
        },
        value=90.0,
    )
    study.add_trial(trial)
    return study


@pytest.fixture
def experiments_dir(tmp_path: Path) -> Path:
    root = tmp_path / "experiments"
    # Study ending at 2023-01 (covers Dec 2022 + Jan 2023)
    s1 = root / "bayesian_2023_01_w2m" / "BTCUSDT"
    s1.mkdir(parents=True)
    joblib.dump(_fake_study(0.9975, 14738), s1 / "BTCUSDT_optuna_study.pkl")

    # Study ending at 2023-03
    s2 = root / "bayesian_2023_03_w2m" / "BTCUSDT"
    s2.mkdir(parents=True)
    joblib.dump(_fake_study(0.9989, 6057), s2 / "BTCUSDT_optuna_study.pkl")
    return root


def test_parse_study_dirs_extracts_period(experiments_dir):
    dirs = _parse_study_dirs(experiments_dir)
    periods = [d[0] for d in dirs]
    assert date(2023, 1, 1) in periods
    assert date(2023, 3, 1) in periods


def test_get_params_uses_most_recent_prior_study(experiments_dir):
    loader = HyperparamLoader(str(experiments_dir))
    # April 2023: most recent prior study is bayesian_2023_03_w2m
    params = loader.get_params("BTCUSDT", 2023, 4, verbose=False)
    assert params["study_name"] == "bayesian_2023_03_w2m"
    assert params["init_exp_T"] == 6057
    assert params["exp_lambda"] == 0.9989


def test_study_not_used_for_its_own_month(experiments_dir):
    loader = HyperparamLoader(str(experiments_dir))
    # March 2023: bayesian_2023_03_w2m is NOT valid for itself.
    params = loader.get_params("BTCUSDT", 2023, 3, verbose=False)
    assert params["study_name"] == "bayesian_2023_01_w2m"


def test_first_month_falls_back_to_defaults(experiments_dir):
    loader = HyperparamLoader(str(experiments_dir))
    # January 2023: no study ends strictly before it → defaults.
    params = loader.get_params("BTCUSDT", 2023, 1, verbose=False)
    assert params["study_name"] == "default"
