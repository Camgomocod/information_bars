"""
Factory for creating Optuna studies with GPU/CPU support.

This module provides a unified interface for creating Optuna studies
with either CPU (TPE) or GPU (BoTorch) samplers.
"""

from typing import Optional, Literal, Dict, Any
import inspect
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from .gpu_utils import GPUDetector, GPUInfo


SamplerType = Literal["tpe", "botorch"]
DeviceType = Literal["cpu", "cuda"]


def _accepts_kwarg(func, name: str) -> bool:
    """True when ``func`` accepts a keyword argument named ``name``.

    Optuna removed ``consider_running_trials`` from TPESampler in v4.0 and from
    BoTorchSampler in optuna-integration 4.x. Introspecting the signature keeps
    the factory working across Optuna 3.x and 4.x.
    """
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


class StudyFactory:
    """
    Factory for creating Optuna studies with appropriate sampler.

    Supports:
    - TPE sampler (CPU-only, always available)
    - BoTorch sampler (GPU-accelerated Bayesian optimization)

    Usage:
        # GPU mode (default)
        study = StudyFactory.create_study(
            study_name="btc_2025_01",
            sampler_type="botorch",
            device="cuda",
        )

        # CPU mode
        study = StudyFactory.create_study(
            study_name="btc_2025_01",
            sampler_type="tpe",
            device="cpu",
        )
    """

    @classmethod
    def create_sampler(
        cls,
        sampler_type: SamplerType = "botorch",
        device: DeviceType = "cuda",
        gpu_id: int = 0,
        seed: int = 42,
        n_startup_trials: int = 10,
        consider_running_trials: bool = False,
    ):
        """
        Create appropriate sampler based on configuration.

        Args:
            sampler_type: "tpe" or "botorch"
            device: "cpu" or "cuda"
            gpu_id: GPU device ID (for CUDA)
            seed: Random seed
            n_startup_trials: Number of random trials before Bayesian optimization
            consider_running_trials: Whether to consider running trials for parallel optimization

        Returns:
            Optuna sampler instance

        Raises:
            RuntimeError: If GPU mode requested but dependencies not available
        """
        if sampler_type == "tpe":
            kwargs = dict(
                seed=seed,
                n_startup_trials=n_startup_trials,
                multivariate=True,  # Better for 2D spaces (lambda, T)
            )
            if _accepts_kwarg(TPESampler, "consider_running_trials"):
                kwargs["consider_running_trials"] = consider_running_trials
            return TPESampler(**kwargs)

        if sampler_type == "botorch":
            # Import device (will raise if CUDA not available)
            device_obj = GPUDetector.get_device(device, gpu_id)

            # Import BoTorch sampler (will raise if not installed)
            # Try optuna_integration first (Optuna 3.1+), then fallback to optuna.integration (older)
            boTorch_sampler = None
            import_error = None

            try:
                from optuna_integration import BoTorchSampler
                boTorch_sampler = BoTorchSampler
            except ImportError as e1:
                import_error = e1
                try:
                    from optuna.integration import BoTorchSampler
                    boTorch_sampler = BoTorchSampler
                except ImportError as e2:
                    import_error = e2

            if boTorch_sampler is None:
                raise ImportError(
                    "BoTorchSampler not available.\n\n"
                    "To enable GPU acceleration with micromamba:\n"
                    "  micromamba install optuna-integration botorch pytorch -c conda-forge -c pytorch\n\n"
                    "Alternatively, use TPE sampler (CPU-only, no extra deps):\n"
                    "  --sampler tpe --device cpu\n\n"
                    f"Original error: {import_error}"
                )

            # Create BoTorch sampler with GPU support
            kwargs = dict(
                seed=seed,
                n_startup_trials=n_startup_trials,
                device=device_obj,
            )
            if _accepts_kwarg(boTorch_sampler, "consider_running_trials"):
                kwargs["consider_running_trials"] = consider_running_trials
            return boTorch_sampler(**kwargs)

        raise ValueError(f"Unknown sampler type: {sampler_type}. Use 'tpe' or 'botorch'.")

    @classmethod
    def create_study(
        cls,
        study_name: str,
        sampler_type: SamplerType = "botorch",
        device: DeviceType = "cuda",
        gpu_id: int = 0,
        direction: str = "maximize",
        seed: int = 42,
        n_startup_trials: int = 10,
        pruner_n_startup: int = 10,
        pruner_n_warmup: int = 5,
        consider_running_trials: bool = False,
    ) -> optuna.Study:
        """
        Create a complete Optuna study with sampler and pruner.

        Args:
            study_name: Name for the study
            sampler_type: "tpe" or "botorch"
            device: "cpu" or "cuda"
            gpu_id: GPU device ID
            direction: "maximize" or "minimize"
            seed: Random seed
            n_startup_trials: Random trials before Bayesian optimization
            pruner_n_startup: Trials before pruning starts
            pruner_n_warmup: Warmup steps for pruner
            consider_running_trials: For parallel optimization

        Returns:
            Configured optuna.Study instance
        """
        # Create sampler
        sampler = cls.create_sampler(
            sampler_type=sampler_type,
            device=device,
            gpu_id=gpu_id,
            seed=seed,
            n_startup_trials=n_startup_trials,
            consider_running_trials=consider_running_trials,
        )

        # Create pruner (same for both samplers)
        pruner = MedianPruner(
            n_startup_trials=pruner_n_startup,
            n_warmup_steps=pruner_n_warmup,
        )

        # Create and return study
        return optuna.create_study(
            study_name=study_name,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
        )

    @classmethod
    def print_config_info(
        cls,
        sampler_type: SamplerType,
        device: DeviceType,
        gpu_id: int = 0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Print and return configuration information.

        Args:
            sampler_type: Type of sampler
            device: Device type
            gpu_id: GPU device ID
            verbose: Whether to print to console

        Returns:
            Dictionary with configuration details
        """
        info: Dict[str, Any] = {
            "sampler": sampler_type,
            "device": device,
            "gpu_id": gpu_id,
        }

        if verbose:
            print("\n" + "="*70)
            print("⚡ OPTIMIZATION CONFIGURATION")
            print("="*70)
            print(f"   Sampler: {sampler_type.upper()}")
            print(f"   Algorithm: {'TPE (Tree-structured Parzen Estimator)' if sampler_type == 'tpe' else 'BoTorch (Bayesian Optimization)'}")

        if sampler_type == "botorch":
            # Check GPU status
            if GPUDetector.is_cuda_available():
                gpu_info = GPUDetector.get_info(gpu_id)
                if gpu_info:
                    info["gpu"] = gpu_info.to_dict()
                    info["gpu_acceleration"] = True

                    if verbose:
                        print(f"   Device: GPU (CUDA)")
                        print(f"   GPU: {gpu_info.name}")
                        print(f"   VRAM: {gpu_info.free_memory_gb:.1f}GB free / {gpu_info.total_memory_gb:.1f}GB total")
                        print(f"   CUDA: {gpu_info.cuda_version}")
                        print(f"   Status: ✅ GPU Acceleration Enabled")
                else:
                    info["gpu_acceleration"] = False
                    if verbose:
                        print(f"   Device: CPU (GPU info unavailable)")
            else:
                info["gpu_acceleration"] = False
                if verbose:
                    print(f"   Device: CPU (CUDA not available)")
                    print(f"   Note: Install PyTorch+CUDA for GPU acceleration")
        else:
            info["gpu_acceleration"] = False
            if verbose:
                print(f"   Device: CPU (TPE sampler)")
                print(f"   Note: TPE does not support GPU acceleration")

        if verbose:
            print("="*70 + "\n")

        return info

    @classmethod
    def get_available_modes(cls) -> Dict[str, bool]:
        """
        Get availability status of all modes.

        Returns:
            Dictionary with availability flags
        """
        detector = GPUDetector()

        return {
            "cpu_tpe": True,  # Always available
            "cpu_botorch": detector.is_torch_available() and detector.is_botorch_available(),
            "gpu_botorch": detector.is_cuda_available() and detector.is_botorch_available(),
        }

    @classmethod
    def print_available_modes(cls) -> None:
        """Print available optimization modes."""
        modes = cls.get_available_modes()

        print("\n" + "="*70)
        print("🚀 AVAILABLE OPTIMIZATION MODES")
        print("="*70)

        print(f"   CPU + TPE:      {'✅ Available' if modes['cpu_tpe'] else '❌ Unavailable'}")
        print(f"   CPU + BoTorch:  {'✅ Available' if modes['cpu_botorch'] else '❌ Install PyTorch+BoTorch'}")
        print(f"   GPU + BoTorch:  {'✅ Available' if modes['gpu_botorch'] else '❌ Install CUDA+PyTorch+BoTorch'}")

        if not modes['gpu_botorch']:
            print("\n   To enable GPU acceleration:")
            print("     micromamba install pytorch botorch -c pytorch -c conda-forge")

        print("="*70 + "\n")
