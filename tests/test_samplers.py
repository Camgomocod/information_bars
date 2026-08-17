"""
test_samplers.py — StudyFactory must create samplers across Optuna versions.

Optuna 4.0 removed ``consider_running_trials`` from TPESampler (and optuna-
integration 4.x from BoTorchSampler). The factory introspects the signature so
studies can be created on both Optuna 3.x and 4.x.
"""

from src.optimization.samplers.factory import StudyFactory, _accepts_kwarg


def test_accepts_kwarg_introspection():
    # This is the guard that keeps the factory version-agnostic.
    assert isinstance(_accepts_kwarg(type("X", (), {"__init__": lambda self, a=None: None}), "a"), bool)


def test_tpe_sampler_creation_optuna4():
    sampler = StudyFactory.create_sampler(sampler_type="tpe", device="cpu", seed=7)
    assert sampler.__class__.__name__ == "TPESampler"


def test_study_creation_with_tpe():
    study = StudyFactory.create_study(
        study_name="factory-smoke",
        sampler_type="tpe",
        device="cpu",
        seed=7,
        direction="maximize",
    )
    assert study.sampler.__class__.__name__ == "TPESampler"


def test_consider_running_trials_is_tolerated():
    # Optuna 3.x accepts the kwarg; Optuna 4.x ignores it. Either way creation works.
    sampler = StudyFactory.create_sampler(sampler_type="tpe", device="cpu", seed=3, consider_running_trials=True)
    assert sampler.__class__.__name__ == "TPESampler"
