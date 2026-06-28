"""
aspect_selection.py

Selects the top-k most frequent normalized aspects and saves:
  - top_aspects.csv        : ranked list with occurrence counts
  - top_aspect_reviews.csv : review rows whose normalized_aspect is in the top-k set

Usage:
    python -m src.aspect_selection [--input PATH] [--output-dir DIR] [--top-k N]
"""

import argparse
from pathlib import Path

import pandas as pd

_ROOT               = Path(__file__).resolve().parents[1]
_DEFAULT_INPUT      = _ROOT / "outputs" / "aspect_normalization"
_DEFAULT_OUTPUT_DIR = _ROOT / "outputs" / "aspect_selection"


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select top-k normalized aspects by frequency.",
    )
    parser.add_argument("--input",      type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k",      type=int,  default=10)
    return parser.parse_args(args)


def resolve_input(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "normalization.csv"
        if not candidate.exists():
            raise FileNotFoundError(f"normalization.csv not found in {path}")
        return candidate
    return path


def run(args: argparse.Namespace) -> None:
    input_path = resolve_input(args.input.resolve())
    df         = pd.read_csv(input_path)

    valid  = df[df["normalized_aspect"].notna() & (df["normalized_aspect"] != "n/a")].copy()
    counts = valid["normalized_aspect"].value_counts()

    top_aspects = counts.head(args.top_k).reset_index()
    top_aspects.columns = ["normalized_aspect", "count"]

    top_set     = set(top_aspects["normalized_aspect"])
    top_reviews = valid[valid["normalized_aspect"].isin(top_set)].reset_index(drop=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    top_aspects.to_csv(output_dir / "top_aspects.csv",       index=False, encoding="utf-8-sig")
    top_reviews.to_csv(output_dir / "top_aspect_reviews.csv", index=False, encoding="utf-8-sig")

    print(f"Selected top {len(top_aspects)} aspects from {counts.shape[0]} unique normalized aspects.")
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    run(parse_args())
