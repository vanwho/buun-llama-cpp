# ACO article figures and data

The [ACO article](../aco-article.md) uses these five figures:

| Figure | Contents |
|---|---|
| [Decode with MTP](figures/decode-mtp-on.png) | GPU, ACO and CPU target KV offload, three prompts |
| [Decode without MTP](figures/decode-mtp-off.png) | GPU, ACO and CPU target KV offload, three prompts |
| [Prefill with MTP](figures/prefill-mtp-on.png) | Fresh repository-content append curves |
| [Prefill without MTP](figures/prefill-mtp-off.png) | Fresh repository-content append curves |
| [CPU MTP placement](figures/cpu-mtp-placement.png) | CPU target KV with GPU MTP compared with master |

Each PNG also has an SVG with the same name. Legends use GPU, ACO and CPU;
the placement comparison names the two CPU configurations separately.

## Data

- [decode.csv](data/decode.csv): maximum measured decode rate per prompt,
  configuration and context checkpoint, with reasoning off. Recorded repeat
  values are retained alongside each maximum.
- [prefill.csv](data/prefill.csv): fresh-content append rates and their new
  token counts. Context positions are cumulative; 1K means 1,024 tokens.
- [cpu-mtp-placement.csv](data/cpu-mtp-placement.csv): maximum measured
  decode rates for CPU target KV with GPU MTP and the master CPU/MTP control.

The November 6 ACO and GPU prefill curves supply every one of their measured
points: 32 ACO appends per MTP setting, eight full-GPU appends with MTP and
16 without it. CPU curves and the decode figures use the recorded comparison
data. The [manifest](data/manifest.json) defines the metrics and records data
and figure hashes.

## Rendering

The renderer reads only the three CSVs in this directory. It writes the PNG
and SVG figures without running a model or accessing benchmark transcripts.

```bash
uv run --no-project --with matplotlib python3 docs/aco-figures/render.py
```

Alternatively, run `python3 docs/aco-figures/render.py` in an environment with
Matplotlib installed.
