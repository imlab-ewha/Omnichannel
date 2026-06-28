"""
scenario_assignment.py

Assigns an omnichannel strategy scenario (S1–S4) to each aspect based on:
  - Aspect type   (search / experience)
  - Dominant channel (online / offline, by mean SHAP value)

Scenario matrix:
  S1 — Search     x Online dominant
  S2 — Search     x Offline dominant
  S3 — Experience x Online dominant
  S4 — Experience x Offline dominant

Usage:
    python -m src.scenario_assignment [--shap PATH] [--types PATH]
                                      [--output-dir DIR] [--epsilon FLOAT]
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

_ROOT               = Path(__file__).resolve().parents[1]
_DEFAULT_SHAP       = _ROOT / "outputs" / "contribution_calculation" / "shap.csv"
_DEFAULT_TYPES      = _ROOT / "outputs" / "type_determination" / "aspect_types.csv"
_DEFAULT_OUTPUT_DIR = _ROOT / "outputs" / "scenario_assignment"
_EPSILON            = 0.0

_SCENARIO_MAP = {
    ("search",     "online"):  "S1",
    ("search",     "offline"): "S2",
    ("experience", "online"):  "S3",
    ("experience", "offline"): "S4",
}


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign omnichannel strategy scenarios (S1-S4) to aspects.",
    )
    parser.add_argument("--shap",       type=Path,  default=_DEFAULT_SHAP)
    parser.add_argument("--types",      type=Path,  default=_DEFAULT_TYPES)
    parser.add_argument("--output-dir", type=Path,  default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epsilon",    type=float, default=_EPSILON)
    return parser.parse_args(args)


def assign_scenario(row: pd.Series, epsilon: float) -> str:
    diff = abs(row["online_mean_shap"] - row["offline_mean_shap"])
    if diff < epsilon:
        return "N/A"
    dom    = "offline" if row["offline_mean_shap"] > row["online_mean_shap"] else "online"
    a_type = str(row["type"]).strip().lower()
    return _SCENARIO_MAP.get((a_type, dom), "Unknown")


def run(args: argparse.Namespace) -> None:
    df_shap  = pd.read_csv(args.shap.resolve())
    df_types = pd.read_csv(args.types.resolve())

    df = df_shap.merge(df_types[["aspect", "type"]], on="aspect", how="left")
    df["scenario"] = df.apply(assign_scenario, axis=1, epsilon=args.epsilon)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scenario_assignment.csv"
    df[["aspect", "scenario"]].to_csv(output_path, index=False, encoding="utf-8-sig")

    print(df["scenario"].value_counts().to_string())
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    run(parse_args())
