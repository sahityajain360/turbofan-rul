"""Report Python + key-package versions for the active interpreter.

Run:  conda run -n <env> python scripts/check_env.py
"""
from __future__ import annotations

import importlib.metadata as md
import sys

REQUIRED = [
    "numpy", "pandas", "scipy", "scikit-learn", "pyarrow",
    "lightgbm", "xgboost", "matplotlib", "seaborn",
    "torch", "fastapi", "uvicorn", "streamlit",
    "shap", "optuna", "mlflow", "pyyaml", "tqdm", "ipykernel", "pytest",
]


def main() -> None:
    print("python", sys.version.split()[0])
    print("executable", sys.executable)
    missing = []
    for pkg in REQUIRED:
        try:
            print(f"  {pkg:<14} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:<14} MISSING")
            missing.append(pkg)
    try:
        import torch

        print("torch.cuda.is_available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("  device:", torch.cuda.get_device_name(0))
    except Exception as e:  # noqa: BLE001
        print("torch check skipped:", type(e).__name__, e)
    print("\nMISSING:", " ".join(missing) if missing else "(none)")


if __name__ == "__main__":
    main()
