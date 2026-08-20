# Runbook — LODO long-context Mamba (M7.7)

Everything is prepared; **run this at home**. Code is written and syntax-checked but **not run**.

## What it does
Leave-one-dataset-out (LODO) across the 9 healthy N-CMAPSS subsets. For each held-out subset:
pretrain a **Mamba** encoder (next-step forecasting, long context **LEN=512**) on the other 8
subsets, fine-tune an RUL head on those 8, then evaluate on the held-out subset's test split.
Also trains a from-scratch Mamba per fold to measure the **cross-dataset SSL lift**. This is the
generalization test that gates the freeze (M8).

Script: [`notebooks/18_ncmapss_lodo_mamba.py`](../notebooks/18_ncmapss_lodo_mamba.py).
Knobs (LEN, SEEDS, DO_PRETRAIN, INCLUDE_SCRATCH, BATCH_SIZE, …) are constants at the top of it.

## 0. (optional, ~15 s) sanity-check the data pipeline before the long run
Run in **Windows PowerShell / Terminal**:
```powershell
wsl -d Ubuntu -u root -- bash -lc "cd /mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE && PYTHONPATH=src ./mambaenv/bin/pytest tests/test_ncmapss.py -k prepare_dataset -q"
```
Expect `1 passed`. (Confirms `prepare_dataset` builds windows with correct shapes + no engine leakage.)

## 1. Launch the run (foreground, live progress + log file)
In a PowerShell window you can leave open:
```powershell
wsl -d Ubuntu -u root -- bash -lc "cd /mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE && PYTHONPATH=src ./mambaenv/bin/python -u notebooks/18_ncmapss_lodo_mamba.py 2>&1 | tee reports/lodo_run.log"
```
- `tee` shows progress in the window **and** writes `reports/lodo_run.log`.
- **Keep the laptop awake & plugged in** (Settings → Power, or `powercfg /change standby-timeout-ac 0`). If it sleeps, the run pauses/dies — but it's **resumable** (see below).

## 2. Monitor (in a second PowerShell window)
```powershell
wsl -d Ubuntu -u root -- tail -f /mnt/d/INTERNSHIP/PREDICTIVE_MAINTENANCE/reports/lodo_run.log
wsl -d Ubuntu -u root -- nvidia-smi
```
You'll see per-fold lines like `[DS03] pretrained seed 0: rmse=.. nasa=.. recall=.. (Ns)`.

## 3. Resume if interrupted
Just re-run the **same launch command**. Completed folds are saved to
`reports/ncmapss_lodo_mamba.json` after each fold and are **skipped** on restart.

## Outputs
- `reports/ncmapss_lodo_mamba.json` — per-fold metrics (pretrained vs scratch) + LODO summary.
- `reports/figures/ncmapss/06_lodo_mamba.png` — per-held-out-subset RMSE, pretrained vs scratch.
- `reports/lodo_run.log` — full log.

## Runtime / tuning
- Default (`SEEDS=[0,1]`, pretrain + scratch, LEN=512): **~4–6 h** on the RTX 3070.
- **Faster:** set `SEEDS=[0]` (≈halves it), or `INCLUDE_SCRATCH=False` (drops the scratch arm).
- **Longer context:** raise `LEN` (e.g. 768/1024) — but flights must be ≥ LEN samples at
  `SUBSAMPLE=5`; if too high, short flights get skipped. Lower `BATCH_SIZE` if you hit VRAM OOM.
- **More data/finer signal:** lower `SUBSAMPLE` (e.g. 2 = 0.5 Hz) — more windows, more RAM/time.

## What to send back
The JSON summary + the figure. Key questions they answer: does long-context Mamba **generalize to
unseen subsets** (held-out RMSE/NASA vs the within-DS02 ~7.1), and does **cross-dataset SSL beat
scratch** under true domain shift?
