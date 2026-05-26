#!/usr/bin/env python3
"""Run one Mooncake contention case on gpu-danyang.

This script is designed to run as root on the remote host. vLLM processes are
started as user danyang, while root is only used for netns and cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path


BASE = Path("/data/danyang/mooncake-contention/case-study")
ROOT = Path("/data/danyang/mooncake-contention")
VENV = Path("/data/danyang/venvs/vllm")
PY = VENV / "bin/python3"
MODEL = Path("/home/danyang/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
SCRIPT_DIR = ROOT / "scripts"


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def kill_patterns() -> None:
    patterns = [
        "qwen3-8b-case-",
        "mooncake_master.*50151",
    ]
    for pat in patterns:
        run(f"pkill -f {sh_quote(pat)} 2>/dev/null || true", check=False)
    time.sleep(1)
    for pat in patterns:
        run(f"pkill -9 -f {sh_quote(pat)} 2>/dev/null || true", check=False)


def setup_netns() -> None:
    run("ip netns list | grep -q '^mc_prefill' || ip netns add mc_prefill")
    run("if ip link show ens10f0np0 >/dev/null 2>&1; then ip addr flush dev ens10f0np0 || true; ip link set ens10f0np0 netns mc_prefill; fi")
    run("ip netns exec mc_prefill ip link set lo up")
    run("ip netns exec mc_prefill ip link set ens10f0np0 up")
    run("ip netns exec mc_prefill ip addr replace 10.10.10.10/24 dev ens10f0np0")
    run("ip addr replace 10.10.10.11/24 dev ens100f0np0")
    run("ip link set ens100f0np0 up")
    run("iptables -C INPUT -s 10.10.10.10/32 -d 10.10.10.11/32 -p tcp -m comment --comment codex-mooncake-netns -j ACCEPT 2>/dev/null || "
        "iptables -I INPUT 1 -s 10.10.10.10/32 -d 10.10.10.11/32 -p tcp -m comment --comment codex-mooncake-netns -j ACCEPT")


def cleanup_netns() -> None:
    run("for p in $(ip netns pids mc_prefill 2>/dev/null || true); do kill -9 \"$p\" 2>/dev/null || true; done", check=False)
    run("ip netns exec mc_prefill ip link set ens10f0np0 netns 1 2>/dev/null || true", check=False)
    run("ip netns del mc_prefill 2>/dev/null || true", check=False)
    run("ip link set ens10f0np0 up 2>/dev/null || true", check=False)
    run("ip addr flush dev ens10f0np0 2>/dev/null || true", check=False)
    run("iptables -D INPUT -s 10.10.10.10/32 -d 10.10.10.11/32 -p tcp -m comment --comment codex-mooncake-netns -j ACCEPT 2>/dev/null || true", check=False)


def write_configs(local_buffer_gb: int = 8) -> None:
    for gpu, dev in [(0, "mlx5_0"), (1, "mlx5_2")]:
        cfg = {
            "metadata_server": "P2PHANDSHAKE",
            "master_server_address": "127.0.0.1:50151",
            "global_segment_size": "80GB",
            "local_buffer_size": f"{local_buffer_gb}GB",
            "protocol": "rdma",
            "device_name": dev,
        }
        (ROOT / f"mooncake_gpu{gpu}_case.json").write_text(json.dumps(cfg, indent=2))


def make_kv(
    case: str,
    workers: int,
    store_replicas: int = 1,
) -> tuple[dict | None, dict | None]:
    if case == "pdonly":
        return (
            {
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_producer",
                "engine_id": "prefill-case-pdonly",
                "kv_connector_extra_config": {
                    "mooncake_protocol": "rdma",
                    "num_workers": workers,
                    "device_name": "mlx5_0",
                },
            },
            {
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_consumer",
                "engine_id": "decode-case-pdonly",
                "kv_connector_extra_config": {
                    "mooncake_protocol": "rdma",
                    "num_workers": workers,
                    "device_name": "mlx5_2",
                },
            },
        )
    if case == "storeonly":
        return (
            {
                "kv_connector": "MooncakeStoreConnector",
                "kv_role": "kv_producer",
                "engine_id": "prefill-case-storeonly",
                "kv_connector_extra_config": {
                    "load_async": True,
                    "discard_partial_chunks": True,
                    "lookup_rpc_port": 19170,
                },
            },
            None,
        )
    if case == "storepd":
        store_connectors = [
            {
                "kv_connector": "MooncakeStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "load_async": True,
                    "discard_partial_chunks": True,
                    "lookup_rpc_port": 19170,
                },
            }
            for _ in range(store_replicas)
        ]
        return (
            {
                "kv_connector": "MultiConnector",
                "kv_role": "kv_producer",
                "engine_id": "prefill-case-storepd",
                "kv_connector_extra_config": {
                    "connectors": [
                        {
                            "kv_connector": "MooncakeConnector",
                            "kv_role": "kv_producer",
                            "kv_connector_extra_config": {
                                "mooncake_protocol": "rdma",
                                "num_workers": workers,
                                "device_name": "mlx5_0",
                            },
                        },
                    ] + store_connectors
                },
            },
            {
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_consumer",
                "engine_id": "decode-case-storepd",
                "kv_connector_extra_config": {
                    "mooncake_protocol": "rdma",
                    "num_workers": workers,
                    "device_name": "mlx5_2",
                },
            },
        )
    raise ValueError(case)


def wait_ready(url: str, in_netns: bool = False, timeout_s: int = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if in_netns:
            cp = run(f"ip netns exec mc_prefill curl -sf {sh_quote(url)}", check=False)
            if cp.returncode == 0:
                return
        else:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
        time.sleep(2)
    raise RuntimeError(f"server not ready: {url}")


def start_services(args: argparse.Namespace, run_dir: Path, timeline: Path) -> None:
    prefill_kv, decode_kv = make_kv(args.case, args.workers,
                                    args.store_replicas)
    model_name = f"qwen3-8b-case-{args.case}"
    max_num_batched_tokens = args.max_num_batched_tokens or args.max_model_len
    common = (
        f"--host 0.0.0.0 --served-model-name {model_name} "
        f"--max-model-len {args.max_model_len} --gpu-memory-utilization 0.60 "
        f"--enable-prefix-caching --num-gpu-blocks-override {args.num_gpu_blocks} "
        f"--max-num-batched-tokens {max_num_batched_tokens}"
    )
    protect_env = (
        f"MOONCAKE_PD_PROTECT_MODE={args.protect_mode} "
        f"MOONCAKE_PD_PROTECT_GRACE_MS={args.protect_grace_ms} "
        f"MOONCAKE_PD_PROTECT_MAX_WAIT_MS={args.protect_max_wait_ms} "
        f"MOONCAKE_STORE_CHUNK_KEYS={args.store_chunk_keys} "
        f"MOONCAKE_STORE_CHUNK_PAUSE_MS={args.store_chunk_pause_ms} "
        f"MOONCAKE_STORE_CHUNK_ACTIVE_PAUSE_MS={args.store_chunk_active_pause_ms} "
    )
    run("ip netns exec mc_prefill runuser -u danyang -- env HOME=/home/danyang USER=danyang LOGNAME=danyang "
        f"bash -lc {sh_quote(f'cd /home/danyang; source {VENV}/bin/activate; nohup mooncake_master --rpc_port=50151 --metrics_port=9013 --logtostderr > {run_dir}/master.log 2>&1 & echo $! > {run_dir}/master.pid')}")
    time.sleep(2)
    if decode_kv is not None:
        cmd = (
            f"cd /home/danyang; source {VENV}/bin/activate; "
            f"CUDA_VISIBLE_DEVICES=1 VLLM_HOST_IP=10.10.10.11 MC_FORCE_HCA=mlx5_2 MOONCAKE_DEVICE=mlx5_2 "
            f"MOONCAKE_CONFIG_PATH={ROOT}/mooncake_gpu1_case.json MOONCAKE_TIMELINE_PATH={timeline} "
            f"nohup vllm serve {MODEL} {common} --port {args.decode_port} "
            f"--kv-transfer-config {sh_quote(json.dumps(decode_kv))} > {run_dir}/decode.log 2>&1 & echo $! > {run_dir}/decode.pid"
        )
        run("runuser -u danyang -- env HOME=/home/danyang USER=danyang LOGNAME=danyang "
            f"bash -lc {sh_quote(cmd)}")
    cmd = (
        f"cd /home/danyang; source {VENV}/bin/activate; "
        f"CUDA_VISIBLE_DEVICES=0 VLLM_HOST_IP=10.10.10.10 MC_FORCE_HCA=mlx5_0 MOONCAKE_DEVICE=mlx5_0 "
        f"{protect_env}"
        f"MOONCAKE_CONFIG_PATH={ROOT}/mooncake_gpu0_case.json MOONCAKE_TIMELINE_PATH={timeline} "
        f"VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 "
        f"nohup vllm serve {MODEL} {common} --port {args.prefill_port} "
        f"--kv-transfer-config {sh_quote(json.dumps(prefill_kv))} > {run_dir}/prefill.log 2>&1 & echo $! > {run_dir}/prefill.pid"
    )
    run("ip netns exec mc_prefill runuser -u danyang -- env HOME=/home/danyang USER=danyang LOGNAME=danyang "
        f"bash -lc {sh_quote(cmd)}")
    if decode_kv is not None:
        wait_ready(f"http://10.10.10.11:{args.decode_port}/v1/models")
    wait_ready(f"http://10.10.10.10:{args.prefill_port}/v1/models", in_netns=True)


def run_requests(args: argparse.Namespace, run_dir: Path) -> None:
    request_script = run_dir / "request_driver.py"
    model_name = f"qwen3-8b-case-{args.case}"
    request_script.write_text(f"""
import json, threading, time, uuid
from pathlib import Path
import requests

run_dir = Path({str(run_dir)!r})
case = {args.case!r}
model = {model_name!r}
prompt_tokens = {args.prompt_tokens}
concurrency = {args.concurrency}
prefill_port = {args.prefill_port}
decode_port = {args.decode_port}
decode_max_tokens = {args.decode_max_tokens}
stream_decode = {args.stream_decode!r}
unique_prompts = {args.unique_prompts!r}
sentence = 'Mooncake KV cache contention measurement sentence. '
# This sentence is stable with Qwen3's tokenizer in the current setup:
# 1000 repetitions produce an 8000-token prompt. Keep the prompt as repeated
# whole sentences so vLLM reports near-exact target token counts.
base_repetitions = max(1, int(round(prompt_tokens / 8)))
results = []
lock = threading.Lock()

def post_one(i):
    transfer_id = f'case-{{case}}-{{uuid.uuid4().hex[:8]}}'
    if unique_prompts:
        prompt = (
            f'Request {{i}} unique cache-line marker. '
            + sentence * max(1, base_repetitions - 2)
        ).strip()
    else:
        prompt = (sentence * base_repetitions).strip()

    def post(name, url, params, max_tokens, stream=False):
        body = {{'model': model, 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0}}
        if params:
            body['kv_transfer_params'] = params
        if stream:
            body['stream'] = True
        t0 = time.perf_counter()
        try:
            if not stream:
                r = requests.post(url, json=body, timeout=900)
                return {{'name': name, 'status': r.status_code, 'elapsed_s': time.perf_counter()-t0, 'text': r.text[:1200]}}
            r = requests.post(url, json=body, timeout=900, stream=True)
            first_token_s = None
            last_token_s = None
            token_chunks = 0
            text_parts = []
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith('data: '):
                    payload = raw[len('data: '):]
                else:
                    payload = raw
                if payload.strip() == '[DONE]':
                    break
                now = time.perf_counter()
                try:
                    data = json.loads(payload)
                except Exception:
                    continue
                choices = data.get('choices') or []
                if not choices:
                    continue
                text = choices[0].get('text') or ''
                if text:
                    if first_token_s is None:
                        first_token_s = now - t0
                    last_token_s = now - t0
                    token_chunks += 1
                    text_parts.append(text)
            elapsed = time.perf_counter() - t0
            tpot = None
            if first_token_s is not None and last_token_s is not None and token_chunks > 1:
                tpot = (last_token_s - first_token_s) / (token_chunks - 1)
            return {{
                'name': name,
                'status': r.status_code,
                'elapsed_s': elapsed,
                'ttft_s': first_token_s,
                'tpot_s': tpot,
                'token_chunks': token_chunks,
                'text': ''.join(text_parts)[:1200],
            }}
        except Exception as e:
            return {{'name': name, 'error': repr(e), 'elapsed_s': time.perf_counter()-t0}}
    if case == 'storeonly':
        res = [post('prefill', f'http://10.10.10.10:{{prefill_port}}/v1/completions', None, 1)]
    else:
        prefill_params = {{'do_remote_decode': True, 'transfer_id': transfer_id}}
        decode_params = {{'do_remote_prefill': True, 'transfer_id': transfer_id, 'remote_engine_id': 'prefill-case-' + case, 'remote_bootstrap_addr': 'http://10.10.10.10:8998'}}
        out = {{}}
        threads = [
            threading.Thread(target=lambda: out.setdefault('prefill', post('prefill', f'http://10.10.10.10:{{prefill_port}}/v1/completions', prefill_params, 1))),
            threading.Thread(target=lambda: out.setdefault('decode', post('decode', f'http://10.10.10.11:{{decode_port}}/v1/completions', decode_params, decode_max_tokens, stream_decode))),
        ]
        threads[0].start(); time.sleep({args.decode_start_delay_s}); threads[1].start()
        for t in threads: t.join()
        res = [out.get('prefill'), out.get('decode')]
    with lock:
        results.append({{'request_index': i, 'transfer_id': transfer_id, 'prompt_tokens_target': prompt_tokens, 'results': res}})

threads = []
for i in range(concurrency):
    t = threading.Thread(target=post_one, args=(i,))
    threads.append(t)
    t.start()
    time.sleep({args.launch_gap_s})
for t in threads:
    t.join()
(run_dir / 'request_results.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
""")
    cp = run("ip netns exec mc_prefill runuser -u danyang -- env HOME=/home/danyang USER=danyang LOGNAME=danyang "
             f"{PY} {request_script}", check=False)
    (run_dir / "request_driver.stdout").write_text(cp.stdout)
    if cp.returncode != 0:
        raise RuntimeError(f"request driver failed: {cp.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["pdonly", "storeonly", "storepd"], required=True)
    parser.add_argument("--bucket", default="trace_p90_scaled_8k")
    parser.add_argument("--prompt-tokens", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--launch-gap-s", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--store-replicas", type=int, default=1,
                        help="Number of MooncakeStoreConnector replicas in storepd.")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--num-gpu-blocks", type=int, default=1024)
    parser.add_argument("--prefill-port", type=int, default=9010)
    parser.add_argument("--decode-port", type=int, default=9020)
    parser.add_argument("--keep-services", action="store_true")
    parser.add_argument("--cleanup-netns", action="store_true")
    parser.add_argument("--protect-mode", default="off",
                        choices=["off", "delay_store", "chunk_yield", "chunk_pace"])
    parser.add_argument("--protect-grace-ms", type=float, default=10.0)
    parser.add_argument("--protect-max-wait-ms", type=float, default=500.0)
    parser.add_argument("--store-chunk-keys", type=int, default=64)
    parser.add_argument("--store-chunk-pause-ms", type=float, default=0.0)
    parser.add_argument("--store-chunk-active-pause-ms", type=float, default=1.0)
    parser.add_argument("--decode-max-tokens", type=int, default=16)
    parser.add_argument("--decode-start-delay-s", type=float, default=0.2,
                        help="Delay between prefill request start and decode request start.")
    parser.add_argument("--stream-decode", action="store_true")
    parser.add_argument("--unique-prompts", action="store_true",
                        help="Use per-request prompt prefixes to avoid prefix-cache sharing.")
    args = parser.parse_args()

    tag = time.strftime("%Y%m%d_%H%M%S") + f"_{args.case}_{args.bucket}_c{args.concurrency}_w{args.workers}"
    run_dir = BASE / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(run_dir, 0o777)
    timeline = run_dir / "timeline.jsonl"
    stop_file = run_dir / "pcie_sampler.stop"
    sampler_csv = run_dir / "pcie_nvml.csv"
    config = vars(args).copy()
    config.update({"run_dir": str(run_dir), "timeline": str(timeline)})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    sampler = None
    try:
        kill_patterns()
        setup_netns()
        write_configs()
        start_services(args, run_dir, timeline)
        sampler = subprocess.Popen([
            str(PY), str(SCRIPT_DIR / "pcie_sampler.py"),
            "--gpus", "0,1",
            "--interval-ms", "20",
            "--out", str(sampler_csv),
            "--stop-file", str(stop_file),
            "--max-seconds", "900",
        ])
        time.sleep(0.5)
        run_requests(args, run_dir)
        time.sleep(2)
    finally:
        stop_file.write_text("stop\n")
        if sampler is not None:
            try:
                sampler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler.kill()
        if not args.keep_services:
            kill_patterns()
        if args.cleanup_netns:
            cleanup_netns()
    print(f"RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
