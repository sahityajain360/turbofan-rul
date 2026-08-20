"""Pytest session setup.

Allow the duplicate OpenMP runtime that occurs when LightGBM/SHAP/MKL (libiomp) and
PyTorch (libomp) are imported in the same process. On Windows, without this, importing
torch *after* lightgbm/shap fails to load ``fbgemm.dll`` (OMP init clash) and breaks
collection of the torch-dependent tests. Set before any test module imports those libs.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import torch FIRST (before any test module pulls in LightGBM/SHAP/MKL). On Windows those
# load a conflicting libiomp, after which torch's fbgemm.dll fails with WinError 127. Loading
# torch first — as the notebooks do — makes it resolve its DLLs cleanly; the rest then coexist.
try:
    import torch  # noqa: F401
except Exception:
    pass
