# Handoff Context — Mamba & Cross-Dataset SSL for N-CMAPSS

This document serves as the project handoff context for Claude. It details the status, environment setup, file creations, test results, and the currently running background pretraining job for Milestone 7.6.

---

## 1. Environment & Setup Relocation (Completed)

### WSL2 Distro on Disk D
- **Problem**: Disk C was critically low on space (~50MB free).
- **Solution**:
  - Unregistered the old Ubuntu installation from Disk C (freeing 17+ GB on C:).
  - Downloaded the official Ubuntu 22.04 LTS (Jammy) WSL rootfs tarball and imported it to **Disk D** (`D:\WSL\ubuntu`). The virtual machine virtual disk (`ext4.vhdx`) now resides entirely on Disk D.
  - Set up the Python virtual environment on the D drive at `/mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE/mambaenv`.
  - Used the `--no-cache-dir` flag for all pip installations to prevent temp downloads from writing to Disk C.

### Package & Dependency Version Alignment
- Installed PyTorch `2.4.0+cu121` with verified CUDA GPU acceleration on the RTX 3070 Laptop GPU.
- Installed `lightgbm` and `shap` in the virtualenv to satisfy `pdm` package imports.
- Resolved function signature incompatibilities in `mamba-ssm`'s custom CUDA kernels:
  - Installed `causal-conv1d==1.4.0` (which has the 7-argument signature expected by `mamba-ssm==2.2.1`).
  - Installed `mamba-ssm==2.2.1`.
  - Downgraded `transformers` to `4.39.0` to resolve generation output class import conflicts.
- Verified that imports are successful and CUDA is fully accessible inside WSL2.

---

## 2. Files Created & Modified

We created/modified the following core modules, tests, and experiment scripts:

1. **Model Backbone**: [mamba.py](file:///D:/INTERNSHIP/PREDICTIVE_MAINTENANCE/src/pdm/models/mamba.py)
   - `MambaBlock`: Wraps the selective state space model, supporting both unidirectional (causal) and bidirectional sweeps. Bidirectional sweeps process the sequence in both forward and backward directions, concatenate the output features, and project them back to `d_model` via a linear layer.
   - `MambaTelemetryEncoder`: Stack of `MambaBlock` layers that acts as a drop-in replacement for the Transformer `TelemetryEncoder`.
   - `NextStepForecastingModel`: An autoregressive pretraining head that takes the encoder representations and predicts the next time step's telemetry `[B, L-1, F]`.

2. **Data Pipeline**: [cross_dataset.py](file:///D:/INTERNSHIP/PREDICTIVE_MAINTENANCE/src/pdm/data/cross_dataset.py)
   - Dynamically maps and loads N-CMAPSS files (DS01 to DS08).
   - Sanitizes and normalizes each dataset *individually* (fitting a separate `RegimeNormalizer` and `StandardScaler` on its own dev train units) to prevent any cross-dataset leakage.
   - Slices telemetry into within-flight windows (`subsample=5`, `length=64`, `stride=32`).
   - Implemented error handling to automatically skip corrupted/truncated datasets (like `DS08d`, which was found to have a 32-byte EOF truncation issue) while successfully loading the remaining 8 datasets (~170k windows).

3. **Experiment Script**: [17_ncmapss_mamba_ssl.py](file:///D:/INTERNSHIP/PREDICTIVE_MAINTENANCE/notebooks/17_ncmapss_mamba_ssl.py)
   - Implements the complete training loop: pretrains Mamba on the combined datasets using Next-Step Forecasting, fine-tunes on DS02, and trains Mamba from scratch on DS02 (5 seeds each) to evaluate the SSL pretraining lift.

4. **Unit Tests**: [test_mamba.py](file:///D:/INTERNSHIP/PREDICTIVE_MAINTENANCE/tests/test_mamba.py)
   - Validates shapes, bidirectional parameters, and backward gradient flows for Mamba sequence models on the GPU.

5. **Project Log**: [PROJECT_LOG.md](file:///D:/INTERNSHIP/PREDICTIVE_MAINTENANCE/PROJECT_LOG.md)
   - Updated with Milestone 7.6 logs under the section `## 2026-06-25 — M7.6: Mamba & Cross-Dataset SSL — environment relocated, Mamba backbone verified`.

---

## 3. Test & Profiling Results

### Unit Tests (Passed)
Running pytest inside the new environment passes successfully:
```bash
wsl -d Ubuntu -u root -- env PYTHONPATH=/mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE/src /mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE/mambaenv/bin/pytest /mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE/tests/test_mamba.py
```
Output: **`4 passed`** on CUDA GPU.

### Subsampling Trade-off Profile (RTX 3070 GPU)
We benchmarked subsampling factors 20, 10, 5, and 2 to determine tradeoffs:
- **`subsample=5` (0.2 Hz)** was selected as optimal. It preserves high temporal density (measurements every 5 seconds) and covers a healthy 5.3 minutes of physical flight per window (`LEN=64`).
- VRAM footprint is flat at **~153 MB** due to fixed input sequence lengths.
- RAM footprint is very safe (~664 MB for DS02, scaling to ~6.5 GB for all datasets combined).

---

## 4. Ongoing Background Job Status

The main experiment script is currently running in the background inside WSL2:
- **PID**: `147` (running as root).
- **Execution command**:
  ```bash
  wsl -d Ubuntu -u root -- env PYTHONPATH=/mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE/src /mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE/mambaenv/bin/python /mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE/notebooks/17_ncmapss_mamba_ssl.py
  ```
- **Job Status**: Active (verified running at local time 16:11 PM).
- **GPU Usage**: Verified via `nvidia-smi` showing **99% GPU utility** and **~1.47 GB VRAM** allocation.
- **Output Targets**:
  - Results JSON: `reports/ncmapss_ds02_mamba_pretraining.json`
  - Comparison Plot: `reports/figures/ncmapss/05_mamba_pretraining.png`

Once the background process completes, it will write the final metrics, convergence rates, and baseline comparison charts directly to the files above.
