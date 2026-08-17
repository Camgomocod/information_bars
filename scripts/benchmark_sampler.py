#!/usr/bin/env python3
"""
GPU vs CPU Benchmark for Bayesian Optimization

This script benchmarks the performance difference between CPU (TPE) and GPU (BoTorch)
samplers for hyperparameter optimization. Results are automatically saved for
performance tracking and documentation.

Usage:
    python scripts/benchmark_sampler.py --symbol BTCUSDT --trials 50
    python scripts/benchmark_sampler.py --symbol BTCUSDT --trials 100 --year 2024 --month 1

The script will:
1. Run optimization with TPE sampler (CPU)
2. Run optimization with BoTorch sampler (GPU if available, else CPU)
3. Compare metrics and save results
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.optimization.samplers.factory import StudyFactory, GPUDetector


def run_benchmark(
    symbol: str,
    year: int,
    month: int,
    n_trials: int,
    window: int = 1,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Run benchmark comparing TPE (CPU) vs BoTorch (GPU/CPU).

    Returns:
        Dictionary with benchmark results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "year": year,
        "month": month,
        "n_trials": n_trials,
        "window": window,
        "system_info": {},
        "benchmarks": [],
    }

    # Collect system info
    detector = GPUDetector()
    results["system_info"] = {
        "cuda_available": detector.is_cuda_available(),
        "pytorch_available": detector.is_torch_available(),
        "botorch_available": detector.is_botorch_available(),
    }

    if detector.is_cuda_available():
        gpu_info = detector.get_info(0)
        if gpu_info:
            results["system_info"]["gpu"] = gpu_info.to_dict()

    # Import optimization module
    from optimization.tune_multiasset_hyperparams import MultiAssetExperiment

    # Benchmark configurations
    configs = [
        ("tpe", "cpu", "TPE (CPU)"),
    ]

    # Only add BoTorch if dependencies available
    if detector.is_botorch_available():
        if detector.is_cuda_available():
            configs.append(("botorch", "cuda", "BoTorch (GPU)"))
        configs.append(("botorch", "cpu", "BoTorch (CPU)"))

    print(f"\n{'='*70}")
    print(f"🚀 BENCHMARK: {symbol} {year}-{month:02d} ({n_trials} trials)")
    print(f"{'='*70}\n")

    for sampler_type, device, label in configs:
        print(f"\n{'─'*70}")
        print(f"Running: {label}")
        print(f"{'─'*70}")

        start_time = time.time()

        try:
            # Create experiment
            results_dir = Path(f"experiments/benchmark_{symbol}_{year}_{month:02d}")
            results_dir.mkdir(parents=True, exist_ok=True)

            experiment = MultiAssetExperiment(
                symbol=symbol,
                results_dir=results_dir,
                prev_study_dir=None,
            )

            # Run optimization
            study = experiment.run_bayesian_optimization(
                year=year,
                month=month,
                window_months=window,
                n_trials=n_trials,
                sampler_type=sampler_type,
                device=device,
                gpu_id=0,
            )

            elapsed = time.time() - start_time

            # Collect metrics
            benchmark_result = {
                "sampler": sampler_type,
                "device": device,
                "label": label,
                "total_time_seconds": round(elapsed, 2),
                "time_per_trial": round(elapsed / n_trials, 3) if n_trials > 0 else None,
                "trials_completed": len([t for t in study.trials if t.state.name == "COMPLETE"]),
                "trials_pruned": len([t for t in study.trials if t.state.name == "PRUNED"]),
                "best_score": study.best_value if study.best_value else None,
                "success": True,
            }

            results["benchmarks"].append(benchmark_result)

            print(f"\n✅ {label} completed in {elapsed:.2f}s")
            print(f"   Time per trial: {benchmark_result['time_per_trial']:.3f}s")
            if benchmark_result['best_score']:
                print(f"   Best score: {benchmark_result['best_score']:.2f}")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ {label} failed after {elapsed:.2f}s: {e}")

            results["benchmarks"].append({
                "sampler": sampler_type,
                "device": device,
                "label": label,
                "total_time_seconds": round(elapsed, 2),
                "error": str(e),
                "success": False,
            })

    # Calculate speedups
    if len(results["benchmarks"]) >= 2:
        baseline = results["benchmarks"][0]["total_time_seconds"]
        for bm in results["benchmarks"][1:]:
            if bm.get("success"):
                bm["speedup_vs_baseline"] = round(baseline / bm["total_time_seconds"], 2)

    return results


def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save benchmark results to JSON and CSV."""
    output_dir = Path(output_dir or "benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol = results["symbol"]

    # Save as JSON
    json_path = output_dir / f"benchmark_{symbol}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save as CSV (benchmarks only)
    if results["benchmarks"]:
        df = pd.DataFrame(results["benchmarks"])
        df["symbol"] = results["symbol"]
        df["year"] = results["year"]
        df["month"] = results["month"]
        df["n_trials"] = results["n_trials"]
        df["timestamp"] = results["timestamp"]

        csv_path = output_dir / f"benchmark_{symbol}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        # Also append to global results file
        global_csv = output_dir / "benchmark_results.csv"
        if global_csv.exists():
            df.to_csv(global_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(global_csv, index=False)

        print(f"\n💾 Results saved:")
        print(f"   JSON: {json_path}")
        print(f"   CSV: {csv_path}")
        print(f"   Global: {global_csv}")


def print_summary(results: Dict[str, Any]) -> None:
    """Print formatted benchmark summary."""
    print(f"\n{'='*70}")
    print("📊 BENCHMARK SUMMARY")
    print(f"{'='*70}\n")

    print(f"Symbol: {results['symbol']}")
    print(f"Period: {results['year']}-{results['month']:02d}")
    print(f"Trials: {results['n_trials']}")
    print(f"Timestamp: {results['timestamp']}")

    print(f"\n{'─'*70}")
    print("System Info:")
    print(f"{'─'*70}")
    print(f"  PyTorch: {'✅' if results['system_info']['pytorch_available'] else '❌'}")
    print(f"  BoTorch: {'✅' if results['system_info']['botorch_available'] else '❌'}")
    print(f"  CUDA: {'✅' if results['system_info']['cuda_available'] else '❌'}")

    if results['system_info'].get('gpu'):
        gpu = results['system_info']['gpu']
        print(f"  GPU: {gpu['name']}")
        print(f"  VRAM: {gpu['free_memory_gb']:.1f}GB free / {gpu['total_memory_gb']:.1f}GB total")

    print(f"\n{'─'*70}")
    print("Results:")
    print(f"{'─'*70}")

    for bm in results["benchmarks"]:
        status = "✅" if bm.get("success") else "❌"
        print(f"\n{status} {bm['label']}")
        print(f"   Time: {bm['total_time_seconds']:.2f}s")

        if bm.get("time_per_trial"):
            print(f"   Per trial: {bm['time_per_trial']:.3f}s")

        if bm.get("best_score"):
            print(f"   Best score: {bm['best_score']:.2f}")

        if bm.get("speedup_vs_baseline"):
            print(f"   Speedup: {bm['speedup_vs_baseline']:.2f}x")

        if bm.get("error"):
            print(f"   Error: {bm['error']}")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPU vs CPU for Bayesian Optimization"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Symbol to benchmark (default: BTCUSDT)"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year to test (default: 2024)"
    )
    parser.add_argument(
        "--month",
        type=int,
        default=1,
        help="Month to test (default: 1)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials per benchmark (default: 50)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Window size in months (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks",
        help="Directory to save benchmark results (default: benchmarks)"
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(
        symbol=args.symbol,
        year=args.year,
        month=args.month,
        n_trials=args.trials,
        window=args.window,
    )

    # Save results
    save_results(results, Path(args.output_dir))

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
