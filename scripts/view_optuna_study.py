"""
Script to visualize and analyze Optuna studies saved in .pkl
"""

import joblib
import pandas as pd
from pathlib import Path
import argparse
import sys


def load_study(pkl_path: Path):
    """Loads an Optuna study from a pickle file (using joblib)"""
    study = joblib.load(pkl_path)
    return study


def print_study_summary(study):
    """Prints study summary"""
    print("=" * 80)
    print("OPTUNA STUDY SUMMARY")
    print("=" * 80)
    print(f"\nStudy name: {study.study_name}")
    print(f"Optimization direction: {study.direction.name}")
    print(f"Total number of trials: {len(study.trials)}")
    print(f"Completed trials: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")
    print(f"Failed trials: {len([t for t in study.trials if t.state.name == 'FAIL'])}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state.name == 'PRUNED'])}")

    if study.best_trial:
        print(f"\n{'=' * 80}")
        print("BEST TRIAL")
        print("=" * 80)
        print(f"Trial number: {study.best_trial.number}")
        print(f"Value: {study.best_trial.value:.6f}")
        print(f"\nParameters:")
        for param, value in study.best_trial.params.items():
            print(f"  {param}: {value}")

        if study.best_trial.user_attrs:
            print(f"\nAdditional attributes:")
            for attr, value in study.best_trial.user_attrs.items():
                print(f"  {attr}: {value}")


def print_trials_dataframe(study, top_n: int = 10):
    """Prints DataFrame with the best trials"""
    df = study.trials_dataframe()

    print(f"\n{'=' * 80}")
    print(f"TOP {top_n} TRIALS")
    print("=" * 80)

    # Sort by value (assuming maximization, adjust if minimization)
    if study.direction.name == 'MAXIMIZE':
        df_sorted = df.sort_values('value', ascending=False)
    else:
        df_sorted = df.sort_values('value', ascending=True)

    # Select relevant columns
    cols_to_show = ['number', 'value', 'state']
    param_cols = [col for col in df.columns if col.startswith('params_')]
    user_attr_cols = [col for col in df.columns if col.startswith('user_attrs_')]

    cols_to_show.extend(param_cols)
    cols_to_show.extend(user_attr_cols)

    # Filter only columns that exist
    cols_to_show = [col for col in cols_to_show if col in df.columns]

    print(df_sorted[cols_to_show].head(top_n).to_string())

    return df


def print_parameter_importance(study):
    """Prints parameter importance if available"""
    try:
        import optuna
        print(f"\n{'=' * 80}")
        print("PARAMETER IMPORTANCE")
        print("=" * 80)

        importance = optuna.importance.get_param_importances(study)

        for param, imp in importance.items():
            print(f"  {param}: {imp:.4f}")
    except Exception as e:
        print(f"\nCould not calculate parameter importance: {e}")


def export_to_csv(study, output_path: Path):
    """Exports trials to CSV"""
    df = study.trials_dataframe()
    df.to_csv(output_path, index=False)
    print(f"\n✓ Trials exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualizes Optuna studies saved in .pkl"
    )
    parser.add_argument(
        "pkl_file",
        type=str,
        help="Path to the Optuna study .pkl file"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of best trials to show (default: 10)"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Path to export trials to CSV"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all trials instead of just the best"
    )

    args = parser.parse_args()

    pkl_path = Path(args.pkl_file)

    if not pkl_path.exists():
        print(f"Error: The file {pkl_path} does not exist")
        sys.exit(1)

    print(f"Loading study from: {pkl_path}")
    study = load_study(pkl_path)

    # Show summary
    print_study_summary(study)

    # Show trials
    top_n = len(study.trials) if args.show_all else args.top
    df = print_trials_dataframe(study, top_n=top_n)

    # Show parameter importance
    print_parameter_importance(study)

    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(study, Path(args.export_csv))

    print(f"\n{'=' * 80}")
    print("VALUE STATISTICS")
    print("=" * 80)
    values = [t.value for t in study.trials if t.value is not None]
    if values:
        print(f"Mean: {pd.Series(values).mean():.6f}")
        print(f"Median: {pd.Series(values).median():.6f}")
        print(f"Std: {pd.Series(values).std():.6f}")
        print(f"Min: {min(values):.6f}")
        print(f"Max: {max(values):.6f}")


if __name__ == "__main__":
    main()
