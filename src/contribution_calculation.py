"""
contribution_calculation.py

Computes SHAP values from trained Random Forest models (one per channel)
and outputs per-aspect mean absolute SHAP and mean SHAP for online and offline.

Usage:
    python -m src.contribution_calculation [--model-dir DIR] [--output-dir DIR]
"""

import argparse
import pickle
import warnings
from pathlib import Path

import pandas as pd
import shap

warnings.filterwarnings("ignore")

_ROOT               = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_DIR  = _ROOT / "outputs" / "regressor_training" / "models"
_DEFAULT_OUTPUT_DIR = _ROOT / "outputs" / "contribution_calculation"


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-channel SHAP values from trained Random Forest models.",
    )
    parser.add_argument("--model-dir",  type=Path, default=_DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    return parser.parse_args(args)


def compute_channel_shap(model_path: Path) -> pd.DataFrame:
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    explainer   = shap.TreeExplainer(saved["model"])
    shap_values = explainer.shap_values(saved["X"])
    return pd.DataFrame(shap_values, columns=saved["X"].columns)


def run(args: argparse.Namespace) -> None:
    model_dir  = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_dfs = {}
    for channel in ["online", "offline"]:
        model_path = model_dir / f"{channel}_rf.pkl"
        if not model_path.exists():
            print(f"Model not found: {model_path} — skipping.")
            continue
        print(f"Computing SHAP for {channel}...")
        shap_df = compute_channel_shap(model_path)
        channel_dfs[channel] = pd.DataFrame({
            f"{channel}_mean_abs_shap": shap_df.abs().mean().round(4),
            f"{channel}_mean_shap":     shap_df.mean().round(4),
        })

    if not channel_dfs:
        print("No models found. Run regressor_training.py first.")
        return

    result = pd.concat(channel_dfs.values(), axis=1)
    result.index.name = "aspect"
    result = result.reset_index()
    result.to_csv(output_dir / "shap.csv", index=False, encoding="utf-8-sig")
    print(f"Saved: {output_dir / 'shap.csv'}")


if __name__ == "__main__":
    run(parse_args())
