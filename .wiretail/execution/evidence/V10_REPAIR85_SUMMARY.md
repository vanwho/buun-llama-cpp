# Repair85 current capability and speed summary

- Result: **current_findings**
- Schema: `repair85-summary-v1`

## Findings

- Small CUDA/native-MTP mechanics are measured; practical speed is a finding, not a threshold pass.
- 256K allocation succeeded, but the actual occupied frontier is 0 and is not inferred from allocation.
- 32K failed at `ggml_cuda_turbo_prefill_attend`; 128K was allocation-only.

## Matched speed ratios

```json
{
  "original_three_prompt": {
    "selected_native_over_all_gpu_native": {
      "fresh_pp": 1.721410869180386,
      "committed_tg": 1.637309784350853
    },
    "selected_native_over_cpu_main_native": {
      "fresh_pp": 3.4251858736059484,
      "committed_tg": 3.9270342513534797
    }
  },
  "context_hot_cold": {
    "value": null,
    "reason": "no matched valid hot control; selected-off did not export stage timings"
  }
}
```

## Cold promotion and limitations

- Controlled rank/copy evidence and organic outcome are separate records.
- See `limitations` and `raw_pointers` in the JSON for severity, minimal reproducers, and immutable source hashes.
