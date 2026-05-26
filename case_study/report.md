# Case Study：真实 KV Offloading 场景中的 PCIe Contention

运行时间：2026-05-13 17:27-17:50 PDT，补充长上下文实验：2026-05-18  
机器：`gpu-danyang` / `enine`  
工作负载来源：`Inferact/codex_swebenchpro_traces`

## 摘要

这份 case study 参考 APNet26 §4.3 的组织方式，把 PCIe contention 放到真实的 vLLM + Mooncake KV 路径里验证。测试的核心路径是 Mooncake Store 里的 Step 1：

```text
Prefill request:
  MultiConnector 把新生成的 KV 同时发给：
    1. Mooncake Store / Pool
    2. Decode worker，通过 Mooncake PD RDMA write
```

主要结论很明确：在 trace-shaped 8000-token replay 下，Store offload 和 PD RDMA write 完全重叠，GPU0 PCIe TX 峰值达到 `17.5-18.2 GiB/s`，并且两条真实 KV transfer 相比各自 isolated baseline 都明显变慢。之后我又补跑了当前 Qwen3-8B 配置能真实支持的最大上下文 `40K tokens`，KV payload 从 8K 的 `1.18 GB` 放大到 `5.90 GB`，用来避免只在过小窗口上调 policy。

| Case | Workers | PD BW | Store BW | PD slowdown | Store slowdown | PD overlap | GPU0 PCIe TX peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Store+PD | 4 | 10.18 GB/s | 9.34 GB/s | +30.80% | +22.09% | 100.0% | 17.10 GiB/s |
| Store+PD | 8 | 10.19 GB/s | 9.45 GB/s | +28.23% | +21.75% | 100.0% | 17.74 GiB/s |

这证明了：contention 不只是 synthetic microbenchmark 里才有，它确实存在于实际 KV offloading 路径中；同时我们也能用 NVML PCIe byte counters 观测到 fabric pressure。

## 工作负载

我使用 HuggingFace 数据集 `Inferact/codex_swebenchpro_traces` 作为 workload source。这个数据集来自成功的 Codex SWE-benchPro trials，特点是 long-context、prefill-heavy，非常适合模拟 agentic coding workload 下的 KV cache 压力。

数据集 card 中的关键统计如下：

| Metric | Value |
| --- | ---: |
| Successful trials | 610 |
| Total LLM calls | 20,230 |
| Mean input tokens / call | 68,329 |
| P50 input tokens / call | 63,917 |
| P90 input tokens / call | 114,888 |
| P99 input tokens / call | 166,322 |
| Mean uncached/computed tokens / call | 3,991 |
| P90 uncached/computed tokens / call | 8,736 |
| P99 uncached/computed tokens / call | 53,323 |
| Overall cache hit rate | 94.2% |

我把轻量 workload shape 提取到了：

```text
case_study/artifacts/trace_shapes.jsonl
case_study/artifacts/trace_summary.json
```

本地粗略 token 估计和 dataset card 很接近：P50 input tokens 约 `65,097`，P90 约 `116,767`，P99 约 `168,150`。

这次实验使用的是 trace-shape replay，而不是完整 semantic replay。也就是说，我们保留 trace 的长上下文压力形态，但 prompt 文本用 deterministic SWE-bench-like 句子构造，并受当前稳定 context length 限制。主 bucket 是：

```text
trace_p90_scaled_8k: 8000 prompt tokens
```

它代表从 P90 long-context agentic call 缩放到当前 8K 稳定上下文的压力。注意，原始 trace 的 P90 input call 大约是 115K tokens，所以这其实是偏保守的 replay。

![Trace token distribution](case_study/artifacts/plots/trace_shape_tokens.png)

## 长上下文补充

8K 对证明 contention 已经足够，但对今天的长上下文 serving 来说偏小。当前实验使用的 Qwen3-8B checkpoint 配置为：

```text
max_position_embeddings = 40960
```

因此我没有强行跑 60K/190K，而是在能真实跑通的最大点补了 `40K tokens`。按 Qwen3-8B 的 KV layout 计算，每 token 的 KV 大小是 `147,456 bytes`；所以 8K、40K、60K、190K 的 KV payload 分别约为：

| Prompt length | Total KV size | Prefix KV size at 94.2% hit |
| --- | ---: | ---: |
| 8K | 1.18 GB | 1.11 GB |
| 40K measured | 5.90 GB | 5.56 GB |
| 60K extrapolated | 8.85 GB | 8.34 GB |
| 190K extrapolated | 28.02 GB | 26.40 GB |

这解释了为什么 8K 是一个 conservative case：真实 agentic traces 的 P50/P90 输入长度远高于 8K，长上下文下 PCIe/RDMA 要搬的 KV 会线性放大。

![Long-context KV scaling](case_study/artifacts/control_plots/long_context_kv_scaling.png)

## 实验环境

在一台物理机器上用两张 GPU 和两张 RDMA NIC 模拟两台 server：

```text
Prefill logical server:
  GPU0
  mlx5_0 / ens10f0np0
  netns mc_prefill
  IP 10.10.10.10

Decode logical server:
  GPU1
  mlx5_2 / ens100f0np0
  default namespace
  IP 10.10.10.11
```

机器处于已验证的 GPUDirect RDMA 可用状态：

```text
kernel: 6.14.0-37-generic
nvidia_peermem loaded
boot args include mem_encrypt=off, amd_iommu=off, iommu=off
```

此前 GDR sanity check 已通过：

```text
same-NIC CUDA-buffer RDMA: 20257.73 MiB/sec
cross-NIC GPU0->GPU1 CUDA-buffer RDMA: 21881.12 MiB/sec
```

vLLM runtime 中的 `MooncakeConnector` 已 patch，使其把 `device_name` 传给 Mooncake `TransferEngine`。因此 prefill 侧固定使用 `mlx5_0`，decode 侧固定使用 `mlx5_2`。

## 实验 Case

我跑了 matched baselines 和 contended path：

```text
PD-only:
  MooncakeConnector producer -> MooncakeConnector consumer

Store-only:
  MooncakeStoreConnector producer

Store+PD:
  MultiConnector(MooncakeConnector + MooncakeStoreConnector)
```

每个 run 都使用 8000 prompt tokens，对应 KV payload 约 `1,179,648,000` bytes。

压力设置有两组：

```text
num_workers=4
num_workers=8
```

## 观测与采样

应用层 timeline 来自 vLLM/Mooncake JSONL events：

```text
pd_write_begin / pd_write_end
store_put_begin / store_put_end
```

PCIe 监控使用 NVML field byte counters，每 20 ms 采样一次：

```text
NVML_FI_DEV_PCIE_COUNT_TX_BYTES = 197
NVML_FI_DEV_PCIE_COUNT_RX_BYTES = 198
```

我一开始尝试了 `nvmlDeviceGetPcieThroughput`，但在这台 A100/GDR 环境里它不能稳定反映真实 transfer burst。因此最终报告和图中使用的是 byte counter delta 算出来的 PCIe 带宽。

## 主结果：workers=4

Matched baselines：

| Case | Transfer | Duration | BW | Request elapsed |
| --- | --- | ---: | ---: | ---: |
| PD-only | PD write | 88.623 ms | 13.31 GB/s | decode 844.187 ms |
| Store-only | Store put | 103.471 ms | 11.40 GB/s | prefill 772.994 ms |

Contended Store+PD：

| Transfer | Duration | BW | Slowdown vs baseline |
| --- | ---: | ---: | ---: |
| PD write | 115.917 ms | 10.18 GB/s | +30.80% |
| Store put | 126.327 ms | 9.34 GB/s | +22.09% |

Overlap 和 fabric pressure：

```text
PD overlap ratio: 100.0%
Store+PD overlap: 115.917 ms
GPU0 PCIe TX peak during overlap: 17.10 GiB/s
GPU0 PCIe TX average during overlap: 10.91 GiB/s
```

![Workers 4 Store+PD timeline](case_study/artifacts/plots/timeline_20260513_174558_storepd_trace_p90_scaled_8k_nvmlcount_c1_w4.png)

## 压力组：workers=8

Matched baselines：

| Case | Transfer | Duration | BW | Request elapsed |
| --- | --- | ---: | ---: | ---: |
| PD-only | PD write | 90.316 ms | 13.06 GB/s | decode 863.560 ms |
| Store-only | Store put | 102.516 ms | 11.51 GB/s | prefill 774.443 ms |

Contended Store+PD：

| Transfer | Duration | BW | Slowdown vs baseline |
| --- | ---: | ---: | ---: |
| PD write | 115.813 ms | 10.19 GB/s | +28.23% |
| Store put | 124.818 ms | 9.45 GB/s | +21.75% |

Overlap 和 fabric pressure：

```text
PD overlap ratio: 100.0%
Store+PD overlap: 115.813 ms
GPU0 PCIe TX peak during overlap: 17.74 GiB/s
GPU0 PCIe TX average during overlap: 17.74 GiB/s
```

`workers=8` 没有显著提升 transfer bandwidth，但 sampled PCIe TX 峰值略高，并且维持了相近的 contention penalty。这说明在这个 8K transfer shape 下，系统已经接近当前 PCIe/GDR 的有效工作点。

![Workers 8 Store+PD timeline](case_study/artifacts/plots/timeline_20260513_174925_storepd_trace_p90_scaled_8k_nvmlcount_w8_c1_w8.png)

## 汇总图

![Bandwidth bars](case_study/artifacts/plots/bandwidth_bars.png)

![Slowdown bars](case_study/artifacts/plots/slowdown_bars.png)

![Slowdown vs PCIe pressure](case_study/artifacts/plots/slowdown_vs_pcie.png)

## 优化补充：Knob A 对 TTFT / TPOT 的影响

在 contention 被确认后，我试了第一个 control knob：当 prefill 侧知道马上要执行 PD RDMA write 时，显式延后 Store offload，让 PD write 避开 Store D2H/Store write 的 PCIe burst。

非 streaming control run 显示：

```text
PD write:       115.69 ms -> 91.67 ms   (-20.76%)
PD BW:           10.20 GB/s -> 12.87 GB/s (+26.19%)
Store+PD overlap: 110.63 ms -> 0 ms
decode elapsed:   0.8876 s -> 0.8724 s  (-15.2 ms)
```

为了拆出 TTFT / TPOT，我又补跑了 streaming decode，每个策略 2 次、每次 8000 prompt tokens + 64 streamed chunks：

| Policy | TTFT | TPOT | Decode elapsed | PD write | PD BW | Store wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 720.417 ms | 11.801 ms/chunk | 1464.050 ms | 115.046 ms | 10.254 GB/s | 0.000 ms |
| delay_store | 695.768 ms | 11.811 ms/chunk | 1440.006 ms | 90.590 ms | 13.023 GB/s | 94.756 ms |

结论：Knob A 的收益主要体现在 TTFT，而不是 TPOT。TTFT 改善 `24.65 ms / 3.42%`，几乎等于 PD write 的改善量；TPOT 只变化 `0.01 ms/chunk`，基本可以认为不变。这说明在这个场景下，contention 主要拖慢 first-token path：decode worker 需要等 prefill KV 通过 PD link 到位后才能开始生成；生成开始之后，后续 token cadence 主要由 decode 侧 generation loop 决定。

代价也清楚：Store offload 被显式等待了约 `94.8 ms`。因此 Knob A 可以证明“保护 PD 有 end-to-end 收益”，但它不是最终策略；后面更合理的方向是 chunked Store yield 或 bandwidth pacing，在保住大部分 TTFT 收益的同时减少 Store tail/backlog。

![Knob A TTFT TPOT](case_study/artifacts/control_plots/knob_a_ttft_tpot.png)

![Knob A streaming transfer sections](case_study/artifacts/control_plots/knob_a_streaming_transfer_sections.png)

## PCIe 带宽分配策略小结

为了和 paper 里的 GPUWeaver 设计语言对齐，我把已有 control runs 重新整理成 PCIe bandwidth allocation policies。当前 runtime 还没有真正的 RDMA per-flow rate limiter 接入；现有可执行 knob 是 Store 让路窗口：

```text
work_conserving:
  不限制 Store，PD 和 Store 自然 overlap。

full_pd_priority:
  PD pending/active 时 Store 等待，近似等价于 PD 期间 Store PCIe budget = 0。
```

已有数据说明，不同 objective 下最好的 setting 不一样：

| Scenario | Best observed setting | 为什么 |
| --- | --- | --- |
| 8K single-request TTFT | `full_pd_priority` | PD BW 从 10.254 提升到 13.023 GB/s，TTFT 从 720.417 降到 695.768 ms |
| 8K c=4 aggregate serving | `work_conserving` | `full_pd_priority` 虽然略微提高 PD BW，但 Store wait/backlog 放大，TTFT mean/p95 变差 |
| 40K long-context PD bandwidth | `full_pd_priority` | PD BW 从 17.917 提升到 20.104 GB/s，但 E2E TTFT 基本不变 |

更完整的对比如下：

| Scenario | Policy | TTFT mean | TTFT p95 | PD BW | Store wait+put | GPU0 PCIe TX peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8K single | work-conserving | 720.417 ms | - | 10.254 GB/s | 121.600 ms | - |
| 8K single | full PD priority | 695.768 ms | - | 13.023 GB/s | 164.881 ms | - |
| 8K c=4 aggregate | work-conserving | 2129.964 ms | 2976.264 ms | 11.873 GB/s | 89.336 ms | 25.240 GB/s |
| 8K c=4 aggregate | full PD priority | 2326.142 ms | 3203.517 ms | 12.186 GB/s | 312.150 ms | 20.831 GB/s |
| 40K single | work-conserving | 6250.050 ms | - | 17.917 GB/s | 828.407 ms | 24.130 GB/s |
| 40K single | full PD priority | 6244.729 ms | - | 20.104 GB/s | 1177.155 ms | 22.450 GB/s |

我进一步补跑了 40K 下的 policy sweep，包括不控制、短 Store delay、长 Store delay、以及 chunked Store pacing：

| Policy | TTFT | Decode elapsed | PD write | PD BW | Store put | Store wait/pause | PCIe TX peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 6260.599 ms | 7179.379 ms | 323.999 ms | 18.204 GB/s | 818.518 ms | 0.000 ms | 23.431 GiB/s |
| delay50 | 6198.310 ms | 7116.748 ms | 312.823 ms | 18.855 GB/s | 809.927 ms | 50.310 ms | 23.653 GiB/s |
| delay300 | 6234.958 ms | 7154.945 ms | 282.546 ms | 20.875 GB/s | 859.282 ms | 289.606 ms | 22.360 GiB/s |
| chunk 64ap2 | 6277.483 ms | 7197.482 ms | 382.672 ms | 15.413 GB/s | 489.587 ms | 54.573 ms | 25.295 GiB/s |
| chunk 64ap5 | 6289.168 ms | 7207.846 ms | 376.119 ms | 15.682 GB/s | 502.287 ms | 112.268 ms | 27.369 GiB/s |
| chunk 128ap5 | 6296.192 ms | 7214.143 ms | 390.236 ms | 15.115 GB/s | 509.929 ms | 61.251 ms | 25.135 GiB/s |

在 40K 下，`delay50` 是这组实验里 TTFT 最好的 setting，比 off 低 `62.290 ms / 0.995%`。`delay300` 把 PD BW 推到最高，PD write 从 `323.999 ms` 降到 `282.546 ms`，但过长等待没有带来最好的 TTFT。chunked Store pacing 在这组参数下反而拖慢 PD，说明长上下文里 Store RPC/chunk 粒度会成为新的 control variable。

![40K policy summary](case_study/artifacts/control_plots/long40k_policy_summary.png)

## End-to-End 收益分析

这里的 end-to-end latency 用 decode 请求的 streaming latency 衡量：

```text
TTFT = decode request start -> first streamed token
Decode elapsed = decode request start -> final streamed token
```

因为 PD 分离下 decode worker 必须等 prefill-produced KV 通过 PD link 到位后才能开始生成，PD transfer 位于 first-token critical path 上；而 Store offload 主要影响后续 prefix reuse，本轮请求中只通过 PCIe contention 间接影响 PD transfer。

8K single-request 下，控制 Store 让路有明确端到端收益：

| Metric | off | delay_store | Delta |
| --- | ---: | ---: | ---: |
| TTFT | 720.417 ms | 695.768 ms | -24.649 ms / -3.42% |
| Decode elapsed | 1464.050 ms | 1440.006 ms | -24.044 ms / -1.64% |
| PD write | 115.046 ms | 90.590 ms | -24.456 ms / -21.26% |
| TPOT | 11.801 ms/chunk | 11.811 ms/chunk | +0.010 ms/chunk |

这说明收益确实来自 first-token path：PD write 缩短 `24.456 ms`，TTFT 缩短 `24.649 ms`，几乎一比一传导；TPOT 基本不变。换句话说，8K 下 PD transfer 占 off 策略 TTFT 的 `15.97%`，占 decode elapsed 的 `7.86%`。如果只优化 PD transfer，TTFT 的理论上限约是这 `15.97%`；当前 knob 已经拿到 `3.42%` TTFT 收益，相当于拿到了 transfer-critical-path 上限的大约 `21.4%`。

我还试了第二种更温和的算法 `chunk_yield`：它不整段暂停 Store，而是把 Store put 切成小块，并在 PD active/pending 时让 Store chunk 间短暂 yield。这个策略更接近后续 pacing/rate limiting 的方向，因为 Store 仍然持续推进，只是降低它和 PD 同时抢 PCIe 的强度。

8K policy sweep 中，`chunk_yield(k=16, active_pause=5ms)` 对 PD KV transfer component 的提升如下：

| Policy | Store handling | TTFT | PD write | PD BW | Store wait |
| --- | --- | ---: | ---: | ---: | ---: |
| off | work-conserving | 721.1 ms | 115.7 ms | 10.20 GB/s | 0.0 ms |
| delay50 | coarse Store delay | 689.4 ms | 92.3 ms | 12.78 GB/s | 50.1 ms |
| chunk_yield k16 ap5 | chunked Store yield | 686.0 ms | 99.4 ms | 11.87 GB/s | 0.0 ms |

相对 off，`chunk_yield k16 ap5` 把 PD KV transfer 从 `115.7 ms` 降到 `99.4 ms`，component latency 提升 `16.3 ms / 14.1%`，PD BW 从 `10.20 GB/s` 提高到 `11.87 GB/s`，提升 `16.4%`。它的 PD component 收益小于 coarse delay，但没有整段 Store wait；这说明除了“强制暂停 D2H/Store”这个上界 knob 之外，确实存在更温和的 bandwidth pacing 方向。

如果 paper 里要强调最清晰、最好看的 component 指标，我建议使用 PD KV transfer component，而不是端到端 TTFT 百分比。这个指标直接对应 GPUWeaver 要保护的 critical communication component：

```text
PD KV transfer improvement =
  (baseline PD transfer latency - optimized PD transfer latency)
  / baseline PD transfer latency
```

除了 8K single-request 的 `24.456 / 115.046 = 21.26%`，我又补跑了并发长上下文压力。因为当前 Qwen3-8B 单请求上限是 40K tokens，所以更大的压力用多请求 unique prompts 构造：

| Scenario | Aggregate prompt | Aggregate PD KV | off PD | optimized PD | PD latency reduction | PD BW improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K x1 | 8K | 1.18 GB | 115.046 ms | 90.590 ms | 21.26% | 27.00% |
| 16K x1 | 16K | 2.36 GB | 216.559 ms | 156.337 ms | 27.81% | 38.52% |
| 16K x2 unique | 32K | 4.72 GB | 210.036 ms | 140.764 ms | 32.98% | 49.21% |
| 24K x2 unique | 48K | 7.08 GB | 254.630 ms | 170.631 ms | 32.99% | 49.23% |
| 32K x2 unique | 64K | 9.44 GB | 278.353 ms | 222.739 ms | 19.98% | 24.97% |
| 40K x1 | 40K | 5.90 GB | 324.000 ms | 282.546 ms | 12.79% | 14.67% |

进一步探索后，最适合写进 paper 的 headline 是 `16K x3 unique`。它和 `24K x2 unique` 一样产生 `48K tokens` aggregate prompt 和 `7.08 GB` aggregate PD KV payload，但 session 数更多，更接近 serving 中多个 long-context requests 同时触发 PD KV transfer 的形态。更重要的是，在 `100 ms` bounded delay budget 下，它已经拿到接近上界控制的收益：

```text
PD KV transfer component: 209.1 ms -> 140.5 ms
PD KV transfer latency reduction: -68.7 ms / -32.83%
PD KV transfer bandwidth: 11.28 -> 16.80 GB/s (+48.87%)
Mean TTFT: -56.6 ms
```

这个数字比 `delay500` 上界略低，但更适合写进 design case study，因为它体现的是 bounded PCIe pacing，而不是无限期等待 Store 完全退场。若只追求上界，`16K x3 delay500` 可以把 PD component 提升到 `36.46%`，PD BW 提升到 `57.38%`。

| Candidate | Aggregate PD KV | PD latency reduction | PD BW improvement | Mean TTFT reduction | Store wait/pause |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16K x3 delay100 | 7.08 GB | 32.83% | 48.87% | 56.6 ms | 300.7 ms |
| 16K x3 delay500 | 7.08 GB | 36.46% | 57.38% | 66.7 ms | 422.8 ms |
| 24K x2 delay100 | 7.08 GB | 27.43% | 37.81% | 54.8 ms | 201.4 ms |
| 24K x2 delay500 | 7.08 GB | 32.99% | 49.23% | 48.6 ms | 350.8 ms |
| 12K x4 delay200 | 7.08 GB | 29.89% | 42.63% | 50.2 ms | 435.4 ms |
| 12K x4 delay500 | 7.08 GB | 32.39% | 47.90% | 71.1 ms | 423.3 ms |
| 8K x6 delay100 | 7.08 GB | 20.80% | 26.26% | 48.3 ms | 329.1 ms |
| 8K x8 delay500 | 9.44 GB | 26.41% | 35.88% | 49.8 ms | 589.6 ms |
| 24K x2 chunk32 | 7.08 GB | 16.56% | 19.84% | 6.2 ms | 188.7 ms |

这个 component 指标比端到端 TTFT 百分比更适合作为 case-study headline，因为它直接测量 contention control 对被保护通信阶段的效果；随后再用 TTFT 说明 component improvement 会被 prefill compute 和 scheduling 占比稀释。`8K x8` 说明在更高 session count 和 `9.44 GB` aggregate PD KV payload 下，控制仍然有效；`chunk32` 说明 naive chunk-yield 确实能改善 PD component，但效果不如 explicit PD-priority delay；这支持后续实现真正的 PCIe/RDMA rate limiter，而不是停留在简单 chunk sleep。

![PD component improvement](case_study/artifacts/control_plots/pd_component_improvement.png)

![Component exploration candidates](case_study/artifacts/control_plots/component_exploration_candidates.png)

40K long-context 下，KV payload 增加到 `5.90 GB`，PD transfer 的绝对时间更长，但 prefill compute 也显著放大，所以端到端百分比变小：

| Metric | off | best TTFT policy: delay50 | Delta |
| --- | ---: | ---: | ---: |
| TTFT | 6260.599 ms | 6198.310 ms | -62.290 ms / -0.995% |
| Decode elapsed | 7179.379 ms | 7116.748 ms | -62.631 ms / -0.872% |
| PD write | 323.999 ms | 312.823 ms | -11.177 ms / -3.45% |
| TPOT | 14.581 ms/chunk | 14.575 ms/chunk | -0.006 ms/chunk |

40K off 策略中，PD transfer 占 TTFT 的 `5.18%`，占 decode elapsed 的 `4.51%`。因此即使把 PD transfer 完全消掉，TTFT 的理论上限也只有约 `5.18%`。实测最佳 TTFT 收益是 `0.995%`，说明长上下文下优化仍有收益，但主要价值不再是大幅降低单请求 TTFT，而是稳定 first-token critical path，并为高并发时避免 PCIe 饱和和排队提供空间。

这也解释了为什么 `delay300` 不是最好的 E2E policy：它把 PD BW 提到 `20.875 GB/s`，但 Store 被等待 `289.606 ms`，对整体调度和后续 reuse 不友好。E2E 最优不是“PD 越快越好”，而是在 PD critical path 和 Store backlog 之间找平衡；40K 这组里短让路 `delay50` 最好。

因此，当前最好的结论不是“永远暂停 Store”，而是：

```text
full_pd_priority 是单请求 first-token path 的上界控制，
但 aggregate serving 需要 bounded PCIe pacing。
40K 下较短的 Store delay 比粗暴长 delay 更适合作为起点。
下一步应该实际 sweep 120/80、140/60、160/40 这类 PCIe budget split，
目标是在保住大部分 PD/TTFT 收益的同时限制 Store backlog。
```

我新增了两个脚本：

```text
scripts/run_pcie_allocation_sweep.py
scripts/analyze_pcie_allocation_settings.py
case_study/artifacts/pcie_allocation_policy_summary.csv
```

其中 `run_pcie_allocation_sweep.py` 会 sweep Store 让路窗口，作为目前没有细粒度 PCIe rate limiter 时的近似实验入口；`analyze_pcie_allocation_settings.py` 汇总已有 off/delay_store、aggregate、40K runs。

## 解释

这个实验对应 APNet26 §4.3 的 case-study 结构：

| APNet case-study element | 本实验中的对应项 |
| --- | --- |
| Application window | Prefill request 中新 KV 同时写 Store 和发 Decode 的窗口 |
| Critical progress signals | PD write duration/BW、Store put duration/BW、decode elapsed |
| Fabric signals | GPU0 PCIe TX/RX byte-counter samples、Mooncake timeline |
| Harmful contention condition | Store+PD 高 overlap + 高 PCIe TX + 相比 isolated controls 带宽下降 |
| Future knob | RDMA/Store injection pacing、worker-count tuning、delayed/chunked Store put |

关键点是：这个现象不是“工作量变多所以变慢”这么简单。matched controls 已经把每条路径的单独运行性能测出来了，而且 prompt shape 和 KV payload 是一致的。当两条流同时运行时，它们完全 overlap，并且两边都损失带宽：

```text
workers=4:
  PD BW:    13.31 -> 10.18 GB/s
  Store BW: 11.40 ->  9.34 GB/s

workers=8:
  PD BW:    13.06 -> 10.19 GB/s
  Store BW: 11.51 ->  9.45 GB/s
```

因此，这可以作为“真实 KV offloading path 存在 PCIe contention”的直接证据。

## 论文图更新：不同 KV size 下选择最优 delay

为回应“图应该看 TTFT，而且 latency reduction 用百分比”的要求，我补跑了一组 best-delay sweep。每个 workload 都包含一个 `off` baseline 和多个 delay 候选；最终只按同一 workload 内的 mean TTFT reduction 选择最优点，论文图不暴露具体 delay 时间。

| Workload | Aggregate PD KV | TTFT off | TTFT best | TTFT reduction | PD transfer reduction | PD BW improvement |
|---|---:|---:|---:|---:|---:|---:|
| 8K x 1 | 1.18 GB | 741.13 ms | 695.92 ms | 6.10% | 19.35% | 24.00% |
| 16K x 1 | 2.36 GB | 1793.19 ms | 1743.19 ms | 2.79% | 20.15% | 25.24% |
| 16K x 2 | 4.72 GB | 2570.89 ms | 2502.00 ms | 2.68% | 34.38% | 52.40% |
| 16K x 3 | 7.08 GB | 3359.54 ms | 3257.34 ms | 3.04% | 33.30% | 49.93% |
| 8K x 8 | 9.44 GB | 4490.91 ms | 4384.09 ms | 2.38% | 30.95% | 44.82% |

这组数据适合放进 paper 的原因是：TTFT 是端到端 serving 指标，PD KV transfer 是 contention 直接影响的 component 指标。结果显示，GPUWeaver 的 delay control 在不同 KV size 和 serving shape 下都能降低 TTFT，幅度为 `2.4--6.1%`；同时 PD KV transfer component 被优化 `19.4--34.4%`，说明 TTFT 收益不是偶然噪声，而是来自被保护通信阶段的改善。

论文图已经更新为两个 panel：

```text
APNet26___FabricContention/figures/pcie_case_study_control.pdf
APNet26___FabricContention/figures/pcie_case_study_control.png
case_study/artifacts/control_plots/paper_pcie_case_study_control.pdf
case_study/artifacts/control_plots/paper_pcie_case_study_control.png
```

对应原始分析表：

```text
case_study/artifacts/best_delay_sweep_all.csv
case_study/artifacts/best_delay_sweep_selected.csv
```

## Artifacts

主要本地 artifacts：

```text
case_study/artifacts/trace_summary.json
case_study/artifacts/trace_shapes.jsonl
case_study/artifacts/pd_component_improvement_summary.csv
case_study/artifacts/component_exploration_summary.csv
case_study/artifacts/component_exploration_candidates.csv
case_study/artifacts/best_delay_sweep_all.csv
case_study/artifacts/best_delay_sweep_selected.csv
case_study/artifacts/long_context_policy_summary.csv
case_study/artifacts/pcie_allocation_policy_summary.csv
case_study/artifacts/plots/summary.csv
case_study/artifacts/plots/*.png
```

远端 run 目录：

```text
/data/danyang/mooncake-contention/case-study/runs/20260513_174416_pdonly_trace_p90_scaled_8k_nvmlcount_c1_w4/
/data/danyang/mooncake-contention/case-study/runs/20260513_174503_storeonly_trace_p90_scaled_8k_nvmlcount_c1_w4/
/data/danyang/mooncake-contention/case-study/runs/20260513_174558_storepd_trace_p90_scaled_8k_nvmlcount_c1_w4/
/data/danyang/mooncake-contention/case-study/runs/20260513_174742_pdonly_trace_p90_scaled_8k_nvmlcount_w8_c1_w8/
/data/danyang/mooncake-contention/case-study/runs/20260513_174830_storeonly_trace_p90_scaled_8k_nvmlcount_w8_c1_w8/
/data/danyang/mooncake-contention/case-study/runs/20260513_174925_storepd_trace_p90_scaled_8k_nvmlcount_w8_c1_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_115927_storepd_policy_long40k_off_c1_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_120022_storepd_policy_long40k_delay50_c1_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_120117_storepd_policy_long40k_delay300_c1_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_120211_storepd_policy_long40k_cy64ap2_c1_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_120306_storepd_policy_long40k_cy64ap5_c1_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_120401_storepd_policy_long40k_cy128ap5_c1_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_122323_storepd_component_16k_c2_off_c2_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_122421_storepd_component_16k_c2_delay_c2_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_122843_storepd_component_24k_c2_off_c2_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_122950_storepd_component_24k_c2_delay_c2_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_122518_storepd_component_32k_c2_off_c2_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_122629_storepd_component_32k_c2_delay_c2_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_123707_storepd_explore_24k_c2_delay50_c2_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_123802_storepd_explore_24k_c2_delay100_c2_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_123855_storepd_explore_24k_c2_delay200_c2_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_123949_storepd_explore_24k_c2_chunk32ap5_c2_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_124043_storepd_explore_24k_c2_chunk64ap5_c2_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_124145_storepd_explore_16k_c3_off_c3_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_124244_storepd_explore_16k_c3_delay500_c3_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_124337_storepd_explore_12k_c4_off_c4_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_124435_storepd_explore_12k_c4_delay500_c4_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_124908_storepd_explore2_16k_c3_delay100_c3_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_125002_storepd_explore2_16k_c3_delay200_c3_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_125055_storepd_explore2_12k_c4_delay100_c4_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_125147_storepd_explore2_12k_c4_delay200_c4_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_125240_storepd_explore2_8k_c6_off_c6_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_125338_storepd_explore2_8k_c6_delay100_c6_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_125431_storepd_explore2_8k_c6_delay500_c6_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_125523_storepd_explore2_8k_c8_off_c8_w8/
/data/danyang/mooncake-contention/case-study/runs/20260518_125625_storepd_explore2_8k_c8_delay500_c8_w8/
/data/danyang/mooncake-contention/case-study/runs/20260520_*_storepd_bestdelay_*/
```

本地脚本：

```text
scripts/extract_codex_trace_shapes.py
scripts/pcie_sampler.py
scripts/run_mooncake_case.py
scripts/run_pcie_allocation_sweep.py
scripts/analyze_case_study.py
scripts/analyze_pcie_allocation_settings.py
scripts/run_remote_policy_long40k.sh
scripts/analyze_long_context_policy.py
scripts/run_remote_case_study_explore.sh
scripts/run_remote_case_study_explore2.sh
scripts/analyze_component_exploration.py
```

## 清理状态

实验结束后远端已经清理：

```text
vLLM/Mooncake experiment processes: stopped
mc_prefill netns: deleted
ens10f0np0: returned to default namespace, no persistent IPv4
codex-mooncake-netns iptables rule: removed
GPU0/GPU1 memory: 14 MiB each, only Xorg remains
```

## 局限性

- 这是 trace-shape replay，不是 Codex agent messages 的完整 semantic replay。它保留 long-context pressure profile，但不保留精确 prompt 内容。
- 当前 Qwen3-8B checkpoint 的真实 context 上限是 40960 tokens，因此 60K/190K/1M 只能外推或换长上下文模型。原始 trace 的压力比 8K 主实验更大，40K 补充实验只是当前系统能真实跑通的最大点。
- NVML byte counters 可以作为 PCIe pressure signal，但采样间隔仍是 20 ms；对于 100 ms 级别窗口，Mooncake timeline 推导的 transfer bandwidth 更精确。
- 这次使用一台物理 host，通过两张 NIC 和 network namespace 模拟两台 server。它保留了真实 cross-NIC RDMA，但仍不同于两个独立 chassis。

## 下一步 Control Knobs

这份 case study 已经可以进入 control phase。推荐先尝试以下 knobs：

1. 保护 PD write：让 Store put 延迟到 PD 完成之后再启动。
2. 把 Store put 切成更小 chunk，并在 chunk 之间 yield。
3. 根据 GPU0 PCIe TX 动态降低 Store 或 PD 的 injection concurrency。
4. 实现 APNet §4.3 风格的 target policy：

```text
if PD BW < target and GPU0 PCIe TX is high:
    reduce Store injection until PD recovers
```

一个合理的初始 target 是：

```text
PD BW >= 12 GB/s
```

这个 target 低于 isolated PD-only bandwidth，但高于 contended Store+PD bandwidth，因此能提供一个清晰的控制空间：在 offload tail latency 和 decode-side progress 之间做 tradeoff。
