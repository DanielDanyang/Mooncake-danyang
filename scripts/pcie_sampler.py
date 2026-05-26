#!/usr/bin/env python3
"""High-frequency GPU PCIe sampler using NVML.

The output is intentionally simple CSV so it can be aligned with Mooncake's
perf_counter_ns timeline after a run.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import pynvml


def parse_gpus(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--interval-ms", type=float, default=20.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    args = parser.parse_args()

    interval_s = args.interval_ms / 1000.0
    pynvml.nvmlInit()
    handles = [(idx, pynvml.nvmlDeviceGetHandleByIndex(idx))
               for idx in parse_gpus(args.gpus)]
    start_ns = time.perf_counter_ns()
    deadline = None if args.max_seconds <= 0 else time.monotonic() + args.max_seconds

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ts_ns",
                "t_rel_s",
                "gpu",
                "pcie_rx_MBps",
                "pcie_tx_MBps",
                "pcie_count_rx_bytes",
                "pcie_count_tx_bytes",
                "pcie_count_rx_MBps",
                "pcie_count_tx_MBps",
                "util_gpu_pct",
                "util_mem_pct",
                "power_w",
            ],
        )
        writer.writeheader()
        last_counts: dict[int, tuple[int, int, int]] = {}
        while True:
            now_ns = time.perf_counter_ns()
            for gpu, handle in handles:
                try:
                    rx_kbs = pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                    tx_kbs = pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                except pynvml.NVMLError:
                    rx_kbs = tx_kbs = -1
                count_tx = count_rx = -1
                count_tx_MBps = count_rx_MBps = -1.0
                try:
                    vals = pynvml.nvmlDeviceGetFieldValues(handle, [
                        pynvml.NVML_FI_DEV_PCIE_COUNT_TX_BYTES,
                        pynvml.NVML_FI_DEV_PCIE_COUNT_RX_BYTES,
                    ])
                    if vals[0].nvmlReturn == 0 and vals[1].nvmlReturn == 0:
                        count_tx = int(vals[0].value.uiVal)
                        count_rx = int(vals[1].value.uiVal)
                        if gpu in last_counts:
                            last_ns, last_tx, last_rx = last_counts[gpu]
                            dt = (now_ns - last_ns) / 1e9
                            if dt > 0:
                                # These counters are exposed as 32-bit values on
                                # this A100 system. Account for wraparound.
                                dtx = (count_tx - last_tx) % (1 << 32)
                                drx = (count_rx - last_rx) % (1 << 32)
                                count_tx_MBps = dtx / dt / (1024 * 1024)
                                count_rx_MBps = drx / dt / (1024 * 1024)
                        last_counts[gpu] = (now_ns, count_tx, count_rx)
                except (pynvml.NVMLError, AttributeError):
                    pass
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_gpu = util.gpu
                    util_mem = util.memory
                except pynvml.NVMLError:
                    util_gpu = util_mem = -1
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except pynvml.NVMLError:
                    power_w = -1
                writer.writerow({
                    "ts_ns": now_ns,
                    "t_rel_s": (now_ns - start_ns) / 1e9,
                    "gpu": gpu,
                    "pcie_rx_MBps": rx_kbs / 1024.0,
                    "pcie_tx_MBps": tx_kbs / 1024.0,
                    "pcie_count_rx_bytes": count_rx,
                    "pcie_count_tx_bytes": count_tx,
                    "pcie_count_rx_MBps": count_rx_MBps,
                    "pcie_count_tx_MBps": count_tx_MBps,
                    "util_gpu_pct": util_gpu,
                    "util_mem_pct": util_mem,
                    "power_w": power_w,
                })
            f.flush()
            if os.path.exists(args.stop_file):
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(interval_s)
    pynvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
