"""
overall_satisfaction_combination.py

Computes a review-level overall satisfaction score by combining the star rating
and the sentiment probability derived from comment text (see Eq. 1 in the paper).

    R    = rating - 1       (scales 1–5 → 0–4)
    Rmax = 4

    Overall Satisfaction = R            if sentiment == negative
                         = Rmax + p_pos if sentiment == positive

Usage:
    python -m src.overall_satisfaction_combination [--input PATH] [--output-dir DIR]
"""

import argparse
from pathlib import Path

import pandas as pd

_ROOT               = Path(__file__).resolve().parents[1]
_DEFAULT_INPUT      = _ROOT / "outputs" / "sentiment_analysis" / "sentiment_analysis.csv"
_DEFAULT_OUTPUT_DIR = _ROOT / "outputs" / "overall_satisfaction_combination"

_RMAX = 4


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine rating and sentiment probability into an overall satisfaction score.",
    )
    parser.add_argument("--input",      type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    return parser.parse_args(args)


def compute_overall_satisfaction(df: pd.DataFrame) -> pd.Series:
    R = df["rating"] - 1
    return R.where(df["sentiment"] == "negative", _RMAX + df["positivity_probability"]).round(4)


def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input.resolve())
    df["overall_satisfaction"] = compute_overall_satisfaction(df)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "overall_satisfaction.csv", index=False, encoding="utf-8-sig")

    print(f"Overall satisfaction score computed for {len(df)} reviews.")
    print(f"Saved: {output_dir / 'overall_satisfaction.csv'}")


if __name__ == "__main__":
    run(parse_args())
