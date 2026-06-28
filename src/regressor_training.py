"""
regressor_training.py

Trains a Random Forest regressor per channel (online / offline) to predict
overall satisfaction from binary aspect-presence (1 / 0) features.

Usage:
    python -m src.regressor_training [--satisfaction PATH] [--top-aspects PATH]
                                     [--reviews PATH] [--output-dir DIR]
"""

import argparse
import pickle
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

_ROOT                 = Path(__file__).resolve().parents[1]
_DEFAULT_SATISFACTION = _ROOT / "outputs" / "overall_satisfaction_combination" / "overall_satisfaction.csv"
_DEFAULT_TOP_ASPECTS  = _ROOT / "outputs" / "aspect_selection" / "top_aspects.csv"
_DEFAULT_REVIEWS      = _ROOT / "outputs" / "aspect_selection" / "top_aspect_reviews.csv"
_DEFAULT_OUTPUT_DIR   = _ROOT / "outputs" / "regressor_training"


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest per channel for aspect contribution analysis.",
    )
    parser.add_argument("--satisfaction",     type=Path, default=_DEFAULT_SATISFACTION)
    parser.add_argument("--top-aspects",      type=Path, default=_DEFAULT_TOP_ASPECTS)
    parser.add_argument("--reviews",          type=Path, default=_DEFAULT_REVIEWS)
    parser.add_argument("--output-dir",       type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-estimators",     type=int,  default=300)
    parser.add_argument("--max-depth",        type=int,  default=8)
    parser.add_argument("--min-samples-leaf", type=int,  default=5)
    parser.add_argument("--random-state",     type=int,  default=42)
    return parser.parse_args(args)


def build_feature_matrix(
    df_sat: pd.DataFrame,
    df_reviews: pd.DataFrame,
    top_aspects: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    y = df_sat.groupby("preprocessed_content")["overall_satisfaction"].mean()

    df_rev = df_reviews[df_reviews["preprocessed_content"].isin(y.index)].copy()
    df_rev = df_rev[df_rev["normalized_aspect"].isin(top_aspects)]
    df_rev = df_rev[["preprocessed_content", "normalized_aspect"]].drop_duplicates()
    df_rev["value"] = 1

    one_hot = df_rev.pivot_table(
        index="preprocessed_content", columns="normalized_aspect",
        values="value", aggfunc="first", fill_value=0,
    )
    for asp in top_aspects:
        if asp not in one_hot.columns:
            one_hot[asp] = 0
    one_hot = one_hot[top_aspects]

    final = one_hot.join(y, how="inner").dropna()
    return final[top_aspects], final["overall_satisfaction"]


def run(args: argparse.Namespace) -> None:
    df_sat      = pd.read_csv(args.satisfaction.resolve())
    top_aspects = pd.read_csv(args.top_aspects.resolve())["normalized_aspect"].tolist()
    df_reviews  = pd.read_csv(args.reviews.resolve())

    model_dir = args.output_dir.resolve() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for channel in ["online", "offline"]:
        print(f"\n[{channel}]")
        df_ch = df_sat[df_sat["channel"] == channel]
        X, y  = build_feature_matrix(df_ch, df_reviews, top_aspects)
        print(f"  Samples: {len(X)}")

        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
        )
        model.fit(X, y)

        model_path = model_dir / f"{channel}_rf.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "X": X, "X_columns": list(X.columns)}, f)
        print(f"  Model saved: {model_path}")

    print("\nDone.")


if __name__ == "__main__":
    run(parse_args())
