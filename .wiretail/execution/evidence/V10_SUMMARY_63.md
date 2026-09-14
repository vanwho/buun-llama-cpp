# V10 phase-63 benchmark summary

- Result: **current findings**
- Revision: `hotpath-v10-20260914`
- Scope: phase-63 receipt and bounded raw manifests only

## Identity, geometry, and placement

The bounded run completed the repaired small coordinate L8192/C1200/H4096/B128/U64
with page size 256. A was not exported by the manifest and remains null. The
observed Qwen3.8-27B UD-IQ4_XS bundle and model identity are retained in the JSON.
Target K/V was Turbo4 on CUDA; native-MTP draft K/V was Turbo4 on GPU with an
8192-row draft capacity.

## Rates and MTP

| row | status | samples | median tok/s | range tok/s |
|---|---|---:|---:|---:|
| cold prefill | measured | 9 | 478.6853 | 475.0088–481.7136 |
| cached append 64 | not_run | 0 | — | no cached segment |
| cached append 256 | not_run | 0 | — | no cached segment |
| committed decode | measured | 9 | 27.0287 | 26.3518–27.0842 |

All nine native-MTP rows have request-scoped 59-draft denominators and 0 accepted
tokens, hence a genuine 0% acceptance result. The nine completed target requests
used the selected packed prefill route. That route observation is not a claim
that a cold page was physically promoted and consumed.

## Requested rows and boundaries

| row | status | reason |
|---|---|---|
| matched original q0/q1/q2 | measured | 3 trials each, fresh cold-prefill |
| controlled T1 physical promotion | not_run | no controlled promotion campaign |
| organic T2/T3 physical promotion | not_run | no organic round trip |
| answer quality | not_run | no quality evaluation |
| L32768 / L128K pilots | not_run | bounded at L8192 |
| full-L 256K allocation | not_run | no allocation attempt |
| occupied C262144 | not_run | no occupied full-context frontier |

Allocation/startup, physical promotion, correctness, and target-route completion
are separate claims. See `V10_SUMMARY_63.json` for every denominator, identity,
raw-manifest checksum, and explicit failure/not-run reason.
