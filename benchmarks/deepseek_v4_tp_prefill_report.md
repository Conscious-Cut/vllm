# DeepSeek V4 Long-Prefill TP/EP Profiling

Hardware: 4x NVIDIA RTX PRO 6000 Blackwell Server Edition test GPUs,
`CUDA_VISIBLE_DEVICES=4,5,6,7`. The production PP4 service on GPUs 0-3 was
left running.

## Summary

The TP4 bottleneck is communication, not MoE or sparse-indexer compute. A torch
profile of DeepSeek V4 TP4 at 128k context showed `vllm::all_reduce` consuming
61.9% of self CUDA time in a late prefill chunk. These test GPUs are PCIe-only
for 4-GPU custom all-reduce, so vLLM falls back to NCCL and TP4 is structurally
slow on this topology.

EP did not help on this model/topology. It loaded successfully, but 128k prefill
was slower than non-EP TP4.

PP4 remains the best tested topology for long-prefill service on this machine.
The code change in this branch reduces large temporary allocations in the
DeepSeek V4 sparse MLA prefill matmul path by replacing concatenated-score
softmax with in-place sink-aware streaming softmax accumulation. It is a modest
but repeatable improvement and preserves the existing math.

## Key Results

| Run | 8k TTFT | 128k TTFT | 500k TTFT | Notes |
| --- | ---: | ---: | ---: | --- |
| DeepSeek V4 TP4 | 4.86s | 85.55s | >18 min | 500k aborted; abandoned request had to be stopped. |
| gpt-oss-120b TP4 | 1.43s | 19.10s at 131k | N/A | Model max context is ~131k. |
| DeepSeek V4 TP4 + EP | 4.79s | 130.62s | N/A | Worse than TP4. |
| DeepSeek V4 PP2 x TP2 | 2.03s min | 39.99s | >13 min | 500k aborted; still too slow. |
| DeepSeek V4 PP4 baseline | 2.11s min | 28.17s | 576.76s | Same test GPUs, unpatched. |
| DeepSeek V4 PP4 + qchunk=128 | N/A | 30.29s | 575.91s | No useful long-context gain. |
| DeepSeek V4 PP4 + this patch | N/A | 26.76s | 571.10s | Best validated PP4 result. |

Additional rejected experiments:

- `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=8192`: 128k regressed to 125.26s in
  TP4.
- Larger sparse-indexer workspace:
  `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1024` regressed TP4 128k to 130.72s.
- Forcing sparse MLA matmul prefill query/head chunks to 2048/16 regressed TP4
  128k to 159.88s.
- Disabling sparse MLA matmul prefill regressed TP4 128k to 153.51s.
- `--max-num-batched-tokens 32768` with PP4 caused an OOM during 128k prefill
  and is not production-safe at the current memory settings.

## Production Recommendation

Do not move this deployment to TP4 on the current 4-GPU PCIe test topology.
Keep PP4 for long-prefill service. TP4 only becomes worth revisiting on a
topology with fast 4-GPU collectives, such as NVLink/NVSwitch or a working
FlashInfer/custom all-reduce path for this GPU generation.
