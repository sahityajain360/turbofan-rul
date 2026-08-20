# Predictive Maintenance — Project Plan (v3, refined)

> **Plan of record.** Supersedes `INITIAL_ROADMAP.md` (kept for history).
> Running journal: [`PROJECT_LOG.md`](PROJECT_LOG.md). Last updated 2026-06-24.

---

## ⚠️ Working agreement (constraints)

1. **UI/dashboard is the LAST milestone.** No FastAPI/Streamlit work until the
   models and features are *finalized and frozen* (M8). Until then, all effort goes
   into improving the modeling.
2. **Keep [`PROJECT_LOG.md`](PROJECT_LOG.md) updated** — one entry per milestone
   (what / why / result).
3. **Final deliverable is a plain-language boss report** (M9) — zero-knowledge
   audience.

---

## 0. Decisions locked

| Decision | Choice | Why |
| :-- | :-- | :-- |
| **Anchor dataset** | NASA **C-MAPSS** (FD001–FD004) → scale to **N-CMAPSS** later | Same turbofan RUL problem, small, most-benchmarked → fast iteration + a literature yardstick. |
| **Problem** | RUL regression **+ a maintenance-decision policy** | "Predict *when* maintenance is needed" is a decision, not just a number. |
| **Timeline** | ~2–3 months | Build the full system, scale up, add a research extension. |
| **Deliverable** | Deployable system (built last) **+** plain-language report | Both — a working demo and a boss-facing writeup. |
| **Compute / env** | RTX 3070 Laptop (8 GB), conda env `rapgen_youtube`, Python 3.10 (torch 2.7+cu118, CUDA ✓) | Reuse the existing fully-provisioned env; only `pip install -e .` needed. |

---

## 1. Problem framing (the part that makes it "real")

We are building **decision support for maintenance planning**, not just a predictor:

```
sensor telemetry → health/degradation estimate → Remaining Useful Life (RUL)
                 → maintenance-decision policy   → alert + recommended action
```

The output that matters is: *"Engine 12 needs maintenance in ~27 cycles — schedule
it,"* with a confidence. **Cost asymmetry is the core constraint:** a missed failure
costs far more than an early replacement, so metrics and thresholds reflect that.

---

## 2. Non-negotiable guardrails

1. **Engine-wise splitting** — split by engine unit, never by random row.
2. **No temporal leakage** — fit scalers/normalizers on train only; windows never
   cross a unit boundary; no future info in features.
3. **Piecewise-linear RUL target** — capped at 125 cycles.
4. **Cost-aware evaluation** — NASA asymmetric score + a $ cost model alongside RMSE/MAE.
5. **Reproducibility** — pinned env, fixed seeds, config-driven runs, tests.
6. **Honest baselines first** — deep learning must beat a tuned LightGBM to ship.

---

## 3. Milestones

Modeling first; presentation layer last (per the working agreement).

| # | Milestone | Status | Artifact / Gate |
| :-- | :-- | :-- | :-- |
| **M0** | Setup & data acquisition | ✅ done | env, parsed + labeled parquet, 24 tests |
| **M1** | EDA | ✅ done | `reports/eda_findings.md` + figures |
| **M2** | Baseline + decision (FD001) | ✅ done | **RMSE 14.4 / NASA 312**, cost-tuned alert policy |
| **M3** | Multi-regime baseline + operating-regime normalization | ✅ done | decision cost −60–70% on FD002/FD004 |
| **M4** | Sequence deep learning (LSTM / GRU / TCN) | ✅ done | GRU best; RNNs win decision cost; ≈ baseline RMSE/NASA → **GRU + LightGBM locked** |
| **M5** | Uncertainty + drift monitoring + robustness | ✅ done | calibrated intervals (53→76%), PSI drift monitor, noise/cross-condition probes |
| **M6** | Explainability + maintenance reasoning | ✅ done | SHAP global+per-engine drivers; per-engine verdict panel fusing RUL+interval+drift+drivers; 37 tests |
| **M7** | **N-CMAPSS scale-up + performance track** (§8) | ✅ done | within-flight Mamba ~7.1; LODO map; Mahalanobis OOD gate; per-fleet fine-tuning validated |
| **M8** | 🔒 **FREEZE** models/features | ✅ **frozen** | models/features locked (see PROJECT_LOG M8). **Deployable demo (FastAPI+Streamlit) deferred** — build only if requested |
| **M9** | **Boss report** (plain language, zero-knowledge audience) | ▶ **in progress** | `reports/boss_report.md` |

---

## 4. Metrics & success criteria

- **RUL accuracy:** RMSE, MAE, and the **NASA score** (primary — cost-asymmetric).
- **Decision quality:** precision/recall for "needs maintenance within H cycles,"
  lead-time, and a **$ cost model** (unplanned failure vs. early service vs. false alarm).
- **Calibration (M5+):** are the 80% intervals actually ~80%?
- **Rough targets (FD001):** baseline RMSE ~16–20 (**achieved 14.4**); deep models
  pushing toward ~13–16 (literature SOTA ~11–13). Decision metrics are the real point.

---

## 5. Stack

Python 3.10 · NumPy/Pandas/PyArrow · scikit-learn · LightGBM (+XGBoost) · PyTorch
(CUDA) · Optuna · SHAP · (serving, built last) FastAPI + Streamlit · (stretch) ONNX.

---

## 6. Repo structure

```
ROADMAP.md / PROJECT_LOG.md   plan of record + running journal
README.md / requirements.txt
configs/        yaml run configs
data/{raw,interim,processed}/
docs/           dataset survey + design notes
notebooks/      EDA + experiment runners
src/pdm/        installable package
  data/         load, split, RUL labeling
  features/     rolling-window features, regime normalization
  models/        baselines + (later) torch models
  evaluation/   NASA score, cost model, decision policy
  utils/        seeds, paths
scripts/        small utilities (env check)
tests/          guardrail tests
reports/        metrics, figures, final report
app/            FastAPI + Streamlit  ← built only at M8
```

---

## 7. Parked — research extensions (only if not pulled into M7 below)

Off the critical path: survival analysis / hazard modeling, physics-informed losses,
edge ONNX/INT8, multimodal (telemetry + logs), graph neural nets.

---

## 8. M7 — N-CMAPSS performance track (active)

**Why now.** On C-MAPSS (small: 100–260 engines) a well-tuned LightGBM matched the deep
models on RMSE/NASA — DL only clearly won on *decision recall* (see PROJECT_LOG M4).
N-CMAPSS is the opposite regime: multi-million points, long 1 Hz flight profiles, real
flight dynamics, and explicit fault-mode labels. The working hypothesis is that **deep
learning should pull ahead here**, because data-hungry models and representation learning
finally have the scale to pay off. M7 tests that hypothesis honestly.

Source ideas curated from [`docs/potential_for_ncmapss.txt`](docs/potential_for_ncmapss.txt)
and the dataset survey. Discipline carries over from M0–M6: **engine-wise splits, no
leakage, baselines-first, cost-aware + calibrated + explainable.** The M2–M6 decision /
uncertainty / drift / SHAP layers are model-agnostic and ride along unchanged.

**Dataset:** N-CMAPSS (NASA PCoE #17, HDF5, DS01–DS08; labels = continuous RUL **+**
fault-mode class HPC/LPC/HPT/LPT). Start with **DS02** (the most-benchmarked subset).
Practical: 1 Hz × 90-min flights is huge → **downsample to ~0.1 Hz** (survey notes ~0.02 Hz
is common locally) to fit the RTX 3070.

| Phase | What | Why it fits us | Priority |
| :-- | :-- | :-- | :-- |
| **7.0 Data + honest baseline** ✅ | HDF5 loader; DS02; regime-normalize; per-flight reduction → LightGBM vs GRU. **Result: GRU RMSE 9.8 / NASA 299 vs LGBM 17.2 / 787 — DL wins decisively (reverses C-MAPSS).** | Baselines-first guardrail; reuses the whole M2–M6 stack. | **done** |
| **7.1 Targets & losses** | **Quantile heads (P10/P50/P90)** for calibrated intervals; **Huber** loss; keep asym-NASA (M4b); add a **degradation-stage / multi-horizon** head. | Quantile > MC-dropout for intervals (extends M5); stage head feeds the M2 decision layer directly. | **high** |
| **7.2 Representations** ✅ | Within-flight {std/min/max} + **flight-to-flight delta** (multi-scale + FFT tried/scoped out). **Result: rep C cuts GRU seed-variance ~12× and NASA 669→276 → GRU 10.2±0.4; LightGBM doesn't benefit.** Chosen rep = C, z-scored. | Cheap; recovers within-flight richness — and mainly *stabilizes* the GRU. | **done** |
| **7.3 Architecture (the test)** ✅ | GRU vs PatchTST vs TCN on rep-C flight-history (5-seed). **Result: GRU wins big (RMSE ~10 / NASA ~290); PatchTST (~19) & TCN (~22) overfit ~300 labels.** Bottleneck = labels, not architecture. PatchTST's real habitat (long within-flight stream) needs pretraining → 7.5. | Tests "DL wins at scale"; controlled architecture swap. | **done** |
| **7.4 Multi-task** | Shared encoder → heads for **{RUL, fault-mode, regime, uncertainty}**. | N-CMAPSS ships fault-mode labels → enables this for free; stabilizes representations. | **medium** |
| **7.5 Self-supervised pretrain** ✅ | Masked-sensor pretraining on within-flight windows → fine-tune RUL; ablation vs scratch. **Result: the within-flight Transformer is the new leader (RMSE ~7.1 / NASA ~183, beats GRU+rep-C ~10/276); but SSL was NEUTRAL vs scratch** — windowing already gives label-abundance and the unlabeled data shares the same flights. | Tests the biggest bet; within-flight windowing was the real unlock (×35 examples). | **done** |
| **7.6 Cross-dataset SSL + Mamba** ✅ | Mamba backbone (WSL2+CUDA); next-step pretrain on 8 subsets (~170k windows) → fine-tune DS02. **Result: Mamba ≈ Transformer (~7.1/180); cross-dataset SSL gives a mild lift (scratch 7.28→pretrained 7.08) — directionally positive, unlike M7.5.** Mamba's long-context edge untested (LEN=64). | SSL's real test (unlabeled ≫ labeled diversity); Mamba scales to long sequences. | **done** |
| **7.7 LODO full-dataset eval** ✅ | LODO over 9 subsets, long-context Mamba (LEN=512), cross-dataset SSL vs scratch. **Result: generalizes to 6/9 unseen fleets (RMSE ~9–12) but fails on DS04 (~33) & DS08a (~20); cross-dataset SSL wins/ties 7/9 (esp. NASA) — keep it.** Long-context Mamba ran the sweep efficiently. | Freeze gate: must generalize on the full dataset; tests Mamba long-context + SSL under domain shift. | **done** |
| **7.6 Transfer & robustness** | Joint train across DS0X / fault modes; **domain adaptation** (adversarial / contrastive) for fault-invariant features. | Attacks the cross-condition failure measured in M5 robustness probes. | **medium** |
| **7.7 Hybrid ensemble** | **DL-encoder embeddings → LightGBM head** (or DL + tree residual correction). | Keeps the interpretable model (M6 SHAP) in the loop; "industrial winners are hybrids." | **stretch** |

**Gate:** ship an N-CMAPSS model that **beats the GRU/LightGBM baseline on NASA score**
on DS02 (the win DL couldn't secure on C-MAPSS), with calibrated intervals and SHAP/driver
explanations intact. Then either freeze (M8) on the strongest C-MAPSS *or* N-CMAPSS model.

---

## Appendix — the four sub-datasets

| Sub-dataset | Operating conditions | Fault modes | Role |
| :-- | :-- | :-- | :-- |
| FD001 | 1 | 1 | easiest — done (M2) |
| FD002 | 6 | 1 | multi-regime (non-stationarity) |
| FD003 | 1 | 2 | multi-fault |
| FD004 | 6 | 2 | hardest — closest to N-CMAPSS in spirit |
