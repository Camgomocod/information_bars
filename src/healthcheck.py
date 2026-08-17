"""Installation and optional network checks for trading-core."""

import argparse
import importlib
from pathlib import Path


def check_dependencies() -> list[str]:
    required = [
        "numpy",
        "pandas",
        "pyarrow",
        "vectorbt",
        "optuna",
        "requests",
        "yaml",
        "scipy",
        "statsmodels",
        "numba",
        "joblib",
    ]
    missing = []
    for package in required:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    return missing


def check_binance_connectivity() -> bool:
    import requests

    try:
        response = requests.head(
            "https://data.binance.vision/data/spot/daily/trades",
            timeout=10,
        )
        return response.status_code in [200, 405]
    except requests.RequestException:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check trading-core installation and services")
    parser.add_argument("--offline", action="store_true", help="Skip network checks")
    args = parser.parse_args()

    print("trading-core healthcheck")
    missing = check_dependencies()
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        return 1
    print("Dependencies: OK")

    checkout_root = Path.cwd()
    if (checkout_root / "src").is_dir():
        required_dirs = ["src", "config", "scripts", "optimization", "tests"]
        missing_dirs = [d for d in required_dirs if not (checkout_root / d).is_dir()]
        if missing_dirs:
            print(f"Missing project directories: {', '.join(missing_dirs)}")
            return 1
        print("Project structure: OK")
    else:
        print("Project structure: installed package (checkout checks skipped)")

    if args.offline:
        print("Network: skipped (offline mode)")
        return 0
    if not check_binance_connectivity():
        print("Network: Binance endpoint unreachable")
        return 2
    print("Network: Binance endpoint reachable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
