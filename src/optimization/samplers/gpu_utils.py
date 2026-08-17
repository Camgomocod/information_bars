"""
GPU detection and configuration utilities for BoTorch optimization.

This module provides utilities to detect GPU availability, check VRAM requirements,
and configure PyTorch devices for GPU-accelerated Bayesian optimization.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class GPUInfo:
    """Container for GPU information."""
    name: str
    total_memory_gb: float
    free_memory_gb: float
    cuda_version: str
    device_id: int
    compute_capability: Tuple[int, int]

    def is_suitable(self, min_vram_gb: float = 2.0) -> bool:
        """Check if GPU has sufficient VRAM."""
        return self.free_memory_gb >= min_vram_gb

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "free_memory_gb": round(self.free_memory_gb, 2),
            "cuda_version": self.cuda_version,
            "device_id": self.device_id,
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
        }


class GPUDetector:
    """Detects and configures GPU for BoTorch optimization."""

    MIN_VRAM_GB = 2.0

    @staticmethod
    def is_torch_available() -> bool:
        """Check if PyTorch is installed."""
        try:
            import torch
            return True
        except ImportError:
            return False

    @staticmethod
    def is_cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def is_botorch_available() -> bool:
        """Check if BoTorch is installed."""
        try:
            import botorch
            return True
        except ImportError:
            return False

    @staticmethod
    def is_optuna_integration_available() -> bool:
        """Check if optuna-integration is installed (needed for BoTorchSampler in Optuna 3.1+)."""
        try:
            from optuna_integration import BoTorchSampler
            return True
        except ImportError:
            # Fallback to old import path
            try:
                from optuna.integration import BoTorchSampler
                return True
            except ImportError:
                return False

    @classmethod
    def check_all_dependencies(cls) -> Tuple[bool, str]:
        """
        Check all required dependencies for GPU mode.

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not cls.is_torch_available():
            return False, "PyTorch not installed"

        if not cls.is_cuda_available():
            return False, "CUDA not available"

        if not cls.is_botorch_available():
            return False, "BoTorch not installed"

        return True, "All dependencies available"

    @classmethod
    def get_info(cls, device_id: int = 0) -> Optional[GPUInfo]:
        """
        Get detailed GPU information.

        Args:
            device_id: GPU device ID to check

        Returns:
            GPUInfo object or None if GPU not available
        """
        if not cls.is_cuda_available():
            return None

        try:
            import torch

            if device_id >= torch.cuda.device_count():
                return None

            props = torch.cuda.get_device_properties(device_id)
            free_memory, total_memory = torch.cuda.mem_get_info(device_id)

            return GPUInfo(
                name=torch.cuda.get_device_name(device_id),
                total_memory_gb=total_memory / 1e9,
                free_memory_gb=free_memory / 1e9,
                cuda_version=torch.version.cuda,
                device_id=device_id,
                compute_capability=(props.major, props.minor),
            )
        except Exception as e:
            warnings.warn(f"Error getting GPU info: {e}")
            return None

    @classmethod
    def get_device(cls, device_str: str = "cuda", gpu_id: int = 0):
        """
        Get PyTorch device based on configuration.

        Args:
            device_str: "cuda" or "cpu"
            gpu_id: GPU device ID

        Returns:
            torch.device object

        Raises:
            RuntimeError: If CUDA requested but not available
        """
        import torch

        if device_str == "cpu":
            return torch.device("cpu")

        if device_str == "cuda":
            # Check all dependencies
            success, msg = cls.check_all_dependencies()
            if not success:
                raise RuntimeError(
                    f"GPU mode requested but: {msg}\n\n"
                    "To enable GPU acceleration with micromamba:\n"
                    "  micromamba install optuna-integration botorch pytorch -c conda-forge -c pytorch\n\n"
                    "To use CPU mode instead:\n"
                    "  --device cpu --sampler tpe"
                )

            # Check GPU availability
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPU mode requested but CUDA is not available.\n"
                    "Possible causes:\n"
                    "  - NVIDIA drivers not installed\n"
                    "  - PyTorch installed without CUDA support\n"
                    "  - No NVIDIA GPU detected\n\n"
                    "To use CPU mode:\n"
                    "  --device cpu"
                )

            # Check VRAM
            info = cls.get_info(gpu_id)
            if info and not info.is_suitable(cls.MIN_VRAM_GB):
                raise RuntimeError(
                    f"GPU detected but insufficient VRAM: {info.free_memory_gb:.1f}GB available, "
                    f"{cls.MIN_VRAM_GB}GB required.\n\n"
                    "To use CPU mode:\n"
                    "  --device cpu"
                )

            return torch.device(f"cuda:{gpu_id}")

        raise ValueError(f"Unknown device string: {device_str}. Use 'cuda' or 'cpu'.")

    @classmethod
    def print_gpu_info(cls, gpu_id: int = 0) -> None:
        """Print formatted GPU information."""
        info = cls.get_info(gpu_id)

        if info:
            print("\n" + "="*60)
            print("🎮 GPU INFORMATION")
            print("="*60)
            print(f"   Device: {info.name}")
            print(f"   CUDA Version: {info.cuda_version}")
            print(f"   Compute Capability: {info.compute_capability[0]}.{info.compute_capability[1]}")
            print(f"   VRAM: {info.free_memory_gb:.1f}GB free / {info.total_memory_gb:.1f}GB total")

            if info.is_suitable(cls.MIN_VRAM_GB):
                print(f"   Status: ✅ Suitable for optimization")
            else:
                print(f"   Status: ⚠️  Insufficient VRAM (need {cls.MIN_VRAM_GB}GB)")
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("🎮 GPU INFORMATION")
            print("="*60)
            print("   No GPU detected")
            print("="*60 + "\n")

    @classmethod
    def get_installation_instructions(cls) -> str:
        """Get installation instructions for GPU dependencies."""
        return """
╔══════════════════════════════════════════════════════════════════════╗
║  GPU SUPPORT INSTALLATION                                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  To enable GPU acceleration with micromamba:                         ║
║                                                                      ║
║    micromamba install optuna-integration botorch pytorch             ║
║                       -c conda-forge -c pytorch                      ║
║                                                                      ║
║  Requirements:                                                       ║
║    - NVIDIA GPU with Compute Capability >= 6.0 (Pascal+)             ║
║    - Minimum 2GB VRAM (4GB+ recommended)                             ║
║    - NVIDIA drivers installed                                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
        """.strip()


# Convenience function for quick checks
def check_gpu_available() -> bool:
    """Quick check if GPU is available for optimization."""
    detector = GPUDetector()
    if not detector.is_cuda_available():
        return False

    info = detector.get_info()
    if info is None:
        return False

    return info.is_suitable(GPUDetector.MIN_VRAM_GB)
