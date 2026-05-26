#!/usr/bin/env python3
"""Analyze best-delay sweep runs and update paper figures."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "case_study" / "artifacts"
RUNS = ART / "runs"
PLOTS = ART / "control_plots"
PAPER_FIG = ROOT / "APNet26___FabricContention" / "figures"


def load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pairs(events: list[dict], begin: str, end: str) -> list[tuple[dict, dict]]:
    starts: list[dict] = []
    out: list[tuple[dict, dict]] = []
    for event in events:
        if event.get("event") == begin:
            starts.append(event)
        elif event.get("event") == end and starts:
            out.append((starts.pop(0), event))
    return out


def interval_ms(interval: tuple[dict, dict]) -> float:
    return (int(interval[1]["ts_ns"]) - int(interval[0]["ts_ns"])) / 1e6


def request_stats(run_dir: Path) -> dict[str, float]:
    data = json.loads((run_dir / "request_results.json").read_text())
    ttft: list[float] = []
    elapsed: list[float] = []
    tpot: list[float] = []
    for request in data:
        for result in request.get("results", []):
            if not result or result.get("name") != "decode":
                continue
            if result.get("ttft_s") is not None:
                ttft.append(float(result["ttft_s"]) * 1000.0)
            if result.get("elapsed_s") is not None:
                elapsed.append(float(result["elapsed_s"]) * 1000.0)
            if result.get("tpot_s") is not None:
                tpot.append(float(result["tpot_s"]) * 1000.0)
    return {
        "ttft_mean_ms": float(np.mean(ttft)) if ttft else np.nan,
        "ttft_p95_ms": float(np.percentile(ttft, 95)) if ttft else np.nan,
        "decode_elapsed_mean_ms": float(np.mean(elapsed)) if elapsed else np.nan,
        "tpot_mean_ms": float(np.mean(tpot)) if tpot else np.nan,
        "request_count": len(ttft),
    }


def pcie_stats(run_dir: Path) -> dict[str, float]:
    path = run_dir / "pcie_nvml.csv"
    if not path.exists():
        return {
            "pcie_tx_peak_gbps": np.nan,
            "pcie_rx_peak_gbps": np.nan,
        }
    df = pd.read_csv(path)
    tx_cols = [col for col in df.columns if col.endswith("_tx_MBps")]
    rx_cols = [col for col in df.columns if col.endswith("_rx_MBps")]
    tx_peak = max(float(df[col].max()) for col in tx_cols) if tx_cols else np.nan
    rx_peak = max(float(df[col].max()) for col in rx_cols) if rx_cols else np.nan
    return {
        "pcie_tx_peak_gbps": tx_peak * 8 / 1000.0,
        "pcie_rx_peak_gbps": rx_peak * 8 / 1000.0,
    }


def summarize(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    events = load_events(run_dir / "timeline.jsonl")
    pd_intervals = pairs(events, "pd_write_begin", "pd_write_end")
    store_intervals = pairs(events, "store_put_begin", "store_put_end")
    wait_intervals = pairs(events, "store_control_wait_begin", "store_control_wait_end")
    pause_intervals = pairs(events, "store_chunk_pause_begin", "store_chunk_pause_end")
    if not pd_intervals:
        raise RuntimeError(f"missing pd_write intervals: {run_dir}")

    pd_total_ms = sum(interval_ms(item) for item in pd_intervals)
    pd_mean_ms = pd_total_ms / len(pd_intervals)
    pd_bytes = sum(
        int(end.get("total_bytes", begin.get("total_bytes", 0)))
        for begin, end in pd_intervals
    )
    bucket = str(cfg["bucket"])
    match = re.match(r"bestdelay_(.+)_(off|d\d+)$", bucket)
    if not match:
        raise RuntimeError(f"unexpected bucket: {bucket}")

    row = {
        "run": run_dir.name,
        "bucket": bucket,
        "scenario": match.group(1),
        "policy_internal": match.group(2),
        "prompt_tokens": int(cfg["prompt_tokens"]),
        "concurrency": int(cfg["concurrency"]),
        "aggregate_prompt_tokens": int(cfg["prompt_tokens"]) * int(cfg["concurrency"]),
        "delay_ms_internal": 0 if match.group(2) == "off" else int(match.group(2)[1:]),
        "pd_count": len(pd_intervals),
        "pd_payload_GB": pd_bytes / 1e9,
        "pd_mean_ms": pd_mean_ms,
        "pd_total_ms": pd_total_ms,
        "pd_GBps": pd_bytes / (pd_total_ms / 1000.0) / 1e9,
        "store_total_ms": sum(interval_ms(item) for item in store_intervals),
        "store_wait_ms": sum(interval_ms(item) for item in wait_intervals),
        "store_pause_ms": sum(interval_ms(item) for item in pause_intervals),
    }
    row.update(request_stats(run_dir))
    row.update(pcie_stats(run_dir))
    return row


def scenario_label(row: pd.Series) -> str:
    tokens_k = int(row["prompt_tokens"]) // 1000
    concurrency = int(row["concurrency"])
    payload = float(row["pd_payload_GB"])
    return f"{tokens_k}Kx{concurrency}\n{payload:.1f}G"


def main() -> int:
    rows: list[dict] = []
    for run_dir in sorted(RUNS.glob("20260520_*_storepd_bestdelay_*")):
        if (run_dir / "config.json").exists() and (run_dir / "timeline.jsonl").exists():
            rows.append(summarize(run_dir))
    ART.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)

    if rows:
        df = pd.DataFrame(rows).sort_values(
            ["aggregate_prompt_tokens", "prompt_tokens", "concurrency", "delay_ms_internal"]
        )
        df.to_csv(ART / "best_delay_sweep_all.csv", index=False)

        best_rows: list[dict] = []
        for scenario, group in df.groupby("scenario", sort=False):
            baseline = group[group["policy_internal"] == "off"].iloc[0]
            candidates = group[group["policy_internal"] != "off"].copy()
            for col in ("ttft_mean_ms", "pd_mean_ms", "pd_GBps", "decode_elapsed_mean_ms"):
                candidates[f"{col}_base"] = baseline[col]
            candidates["ttft_reduction_pct"] = (
                (baseline["ttft_mean_ms"] - candidates["ttft_mean_ms"])
                / baseline["ttft_mean_ms"] * 100.0
            )
            candidates["ttft_reduction_ms"] = baseline["ttft_mean_ms"] - candidates["ttft_mean_ms"]
            candidates["pd_reduction_pct"] = (
                (baseline["pd_mean_ms"] - candidates["pd_mean_ms"])
                / baseline["pd_mean_ms"] * 100.0
            )
            candidates["pd_reduction_ms"] = baseline["pd_mean_ms"] - candidates["pd_mean_ms"]
            candidates["pd_bw_improvement_pct"] = (
                candidates["pd_GBps"] / baseline["pd_GBps"] - 1.0
            ) * 100.0
            candidates["decode_elapsed_reduction_pct"] = (
                (baseline["decode_elapsed_mean_ms"] - candidates["decode_elapsed_mean_ms"])
                / baseline["decode_elapsed_mean_ms"] * 100.0
            )
            candidates["ttft_base_ms"] = baseline["ttft_mean_ms"]
            candidates["pd_base_ms"] = baseline["pd_mean_ms"]
            candidates["pd_base_GBps"] = baseline["pd_GBps"]
            candidates["scenario_label"] = candidates.apply(scenario_label, axis=1)
            candidates = candidates.sort_values(
                ["ttft_reduction_pct", "pd_reduction_pct"], ascending=False
            )
            best_rows.append(candidates.iloc[0].to_dict())

        best = pd.DataFrame(best_rows).sort_values(
            ["aggregate_prompt_tokens", "prompt_tokens", "concurrency"]
        )
        best.to_csv(ART / "best_delay_sweep_selected.csv", index=False)
        paper_best = best[best["scenario"] != "24k_c2"].copy()
        paper_best.to_csv(ART / "best_delay_sweep_paper_selected.csv", index=False)
    else:
        selected_path = ART / "best_delay_sweep_selected.csv"
        paper_path = ART / "best_delay_sweep_paper_selected.csv"
        if not selected_path.exists():
            raise FileNotFoundError(
                f"missing raw runs and cached result CSV: {selected_path}"
            )
        best = pd.read_csv(selected_path)
        if paper_path.exists():
            paper_best = pd.read_csv(paper_path)
        else:
            paper_best = best[best["scenario"] != "24k_c2"].copy()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    blue = "#386CB0"
    green = "#3F8F5F"
    orange = "#D9822B"

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.05), constrained_layout=True)
    x = np.arange(len(paper_best))
    labels = paper_best["scenario_label"].tolist()

    ax = axes[0]
    bars = ax.bar(x, paper_best["pd_reduction_pct"], color=blue, width=0.64,
                  label="PD KV transfer")
    ax.set_xticks(x, labels)
    ax.set_ylabel("PD KV transfer flow reduction (%)")
    ax.set_title("(a) PD KV transfer flow")
    ax.grid(axis="y", alpha=0.24, linewidth=0.6)
    ax.set_ylim(0, max(0.01, paper_best["pd_reduction_pct"].max()) * 1.30)
    for bar, val in zip(bars, paper_best["pd_reduction_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.7,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8.8)
    ax2 = ax.twinx()
    ax2.plot(x, paper_best["pd_bw_improvement_pct"], color=orange, marker="o",
             linewidth=1.5, label="PD KV transfer BW")
    ax2.set_ylabel("PD KV transfer BW improvement (%)", color=orange)
    ax2.tick_params(axis="y", colors=orange)
    ax2.set_ylim(0, max(0.01, paper_best["pd_bw_improvement_pct"].max()) * 1.35)

    ax = axes[1]
    bars = ax.bar(x, paper_best["ttft_reduction_pct"], color=green, width=0.64)
    ax.set_xticks(x, labels)
    ax.set_ylabel("TTFT latency reduction (%)")
    ax.set_title("(b) Serving latency")
    ax.grid(axis="y", alpha=0.24, linewidth=0.6)
    ax.set_ylim(0, max(0.01, paper_best["ttft_reduction_pct"].max()) * 1.28)
    for bar, val in zip(bars, paper_best["ttft_reduction_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.08,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8.8)

    outputs = [
        PAPER_FIG / "pcie_case_study_control.pdf",
        PLOTS / "paper_pcie_case_study_control.pdf",
    ]
    outputs_png = [path.with_suffix(".png") for path in outputs]
    for path in outputs:
        fig.savefig(path)
    for path in outputs_png:
        fig.savefig(path, dpi=240)
    plt.close(fig)

    print(best[[
        "scenario",
        "prompt_tokens",
        "concurrency",
        "pd_payload_GB",
        "delay_ms_internal",
        "ttft_base_ms",
        "ttft_mean_ms",
        "ttft_reduction_pct",
        "pd_base_ms",
        "pd_mean_ms",
        "pd_reduction_pct",
        "pd_bw_improvement_pct",
    ]].to_string(index=False))
    print(ART / "best_delay_sweep_selected.csv")
    print(PAPER_FIG / "pcie_case_study_control.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
