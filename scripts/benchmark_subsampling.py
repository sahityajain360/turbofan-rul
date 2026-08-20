"""
Benchmark different subsampling factors (20, 10, 5, 2) on N-CMAPSS DS02
to evaluate VRAM/RAM, training throughput, and temporal representation tradeoffs.
"""
from __future__ import annotations

import gc
import json
import time
import os
import psutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from pdm import paths
from pdm.data import cmapss, ncmapss
from pdm.features import RegimeNormalizer, make_within_flight_windows
from pdm.models.mamba import MambaTelemetryEncoder
from pdm.models.ssl import MaskedSensorModel, masked_mse, random_mask
from pdm.utils import seed_everything

# Ensure directories exist
paths.ensure_dirs()

SUB_FACTORS = [20, 10, 5, 2]
LEN = 64
STRIDE = 32
BATCH_SIZE = 256
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SENSORS = ncmapss.XS_VARS


def get_ram_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_subsampling(factor: int):
    print(f"\n================ BENCHMARKING SUBSAMPLE = {factor} ================")
    
    # 1. Load and Normalization benchmarking
    gc.collect()
    torch.cuda.empty_cache()
    
    ram_start = get_ram_usage_mb()
    t_start = time.time()
    
    # Load dataset
    dev = ncmapss.load_ncmapss("dev", subsample=factor)
    tr_u, va_u = cmapss.split_units(ncmapss.unit_ids("dev"), val_fraction=0.34, seed=42)
    
    # Normalize
    norm = RegimeNormalizer(n_regimes=20, sensors=SENSORS, op_cols=ncmapss.W_VARS)
    norm.fit(dev[dev.unit.isin(tr_u)])
    dev_n = norm.transform(dev)
    
    scaler = StandardScaler().fit(dev_n[dev_n.unit.isin(tr_u)][SENSORS])
    dev_n[SENSORS] = scaler.transform(dev_n[SENSORS])
    
    # Slice windows
    Xtr, ytr, utr, ctr = make_within_flight_windows(dev_n[dev_n.unit.isin(tr_u)], SENSORS, LEN, STRIDE)
    
    t_data = time.time() - t_start
    ram_end = get_ram_usage_mb()
    ram_added = ram_end - ram_start
    
    n_windows = len(Xtr)
    n_features = len(SENSORS)
    print(f"Data processed in {t_data:.2f}s. Loaded {len(dev_n)} rows, extracted {n_windows} windows. RAM added: {ram_added:.2f} MB")
    
    # 2. Model & Training benchmarking
    seed_everything(0)
    
    # Instantiate Mamba Encoder
    enc = MambaTelemetryEncoder(
        n_features=n_features,
        d_model=64,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        bidirectional=False
    ).to(DEVICE)
    
    model = MaskedSensorModel(enc, n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr)), batch_size=BATCH_SIZE, shuffle=True)
    
    torch.cuda.reset_peak_memory_stats(DEVICE)
    vram_start = torch.cuda.memory_allocated(DEVICE) / (1024 * 1024)
    
    epoch_times = []
    losses = []
    
    t_train_start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        t_ep_start = time.time()
        tot_loss = 0.0
        batches = 0
        
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            xm, mask = random_mask(xb, 0.25)
            loss = masked_mse(model(xm), xb, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            batches += 1
            
        epoch_times.append(time.time() - t_ep_start)
        losses.append(tot_loss / max(1, batches))
        
    t_train = time.time() - t_train_start
    vram_max = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
    vram_used = vram_max - vram_start
    
    avg_epoch_time = np.mean(epoch_times)
    throughput = n_windows / avg_epoch_time if avg_epoch_time > 0 else 0
    
    # Convergence score: loss reduction from epoch 1 to epoch 3
    conv_score = (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0.0
    
    # Effective context retained
    # Original 1Hz resolution means 1 window (len 64) is 64 seconds.
    # Subsampled by `factor` means 1 window (len 64) represents `64 * factor` seconds of physical flight time.
    effective_context_sec = LEN * factor
    
    print(f"Training completed in {t_train:.2f}s. Avg epoch time: {avg_epoch_time:.2f}s.")
    print(f"Max VRAM used: {vram_used:.2f} MB. Throughput: {throughput:.1f} windows/sec.")
    print(f"Losses: {losses} | Convergence: {conv_score:.2%}")
    
    return {
        "subsample_factor": factor,
        "effective_hz": 1.0 / factor,
        "data_loading_time_sec": t_data,
        "ram_added_mb": ram_added,
        "num_rows": len(dev_n),
        "num_windows": n_windows,
        "max_vram_used_mb": vram_used,
        "avg_epoch_time_sec": avg_epoch_time,
        "throughput_windows_per_sec": throughput,
        "losses": losses,
        "convergence_rate": conv_score,
        "effective_context_sec": effective_context_sec
    }


def main():
    results = []
    for factor in SUB_FACTORS:
        try:
            res = benchmark_subsampling(factor)
            results.append(res)
        except Exception as e:
            print(f"Failed benchmark for subsample={factor}: {e}")
            
    # Save results to a json file
    out_path = paths.REPORTS / "benchmark_subsampling_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nProfiling completed! Results saved to {out_path}")
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Subsample':<10}{'Hz':<10}{'Windows':<10}{'RAM (MB)':<10}{'VRAM (MB)':<10}{'Epoch (s)':<12}{'Throughput':<12}{'Context (s)':<12}")
    print("="*80)
    for r in results:
        print(f"{r['subsample_factor']:<10}"
              f"{r['effective_hz']:<10.3f}"
              f"{r['num_windows']:<10}"
              f"{r['ram_added_mb']:<10.1f}"
              f"{r['max_vram_used_mb']:<10.1f}"
              f"{r['avg_epoch_time_sec']:<12.2f}"
              f"{r['throughput_windows_per_sec']:<12.1f}"
              f"{r['effective_context_sec']:<12}")
    print("="*80)


if __name__ == "__main__":
    main()
