# Post-17 gap index — task 17-16

Decision: **bridge complete**. The 17-15 campaign is preserved as observed; this receipt enters the ordered 18–22 remediation path. It does not claim pager quality, physical transfers, or full-context acceptance.

## Evidence boundary

The source campaign is `/srv/ai/paged-kv/results/17-15-quality-20260905T052900Z/`, with checked-in summary [PHASE17_QUALITY_V2.json](PHASE17_QUALITY_V2.json). It used source commit `23411197aa446e4766ce921160ebe2e33cf53992`, executable SHA256 `404622c75944d27c054c0dcf48919d5af0793c35c95758347c011637f023b790`, Qwen3.8-27B-UD-IQ4_XS model SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`, and corpus hash `37111506c8ddc7f9d04086121797cc9c2b7a2842cc1a56ef4b33f506c2252007`. The four mode command hashes are in the JSON receipt. The campaign root checksum manifest verified 447 files with zero failures.

| Mode | Actual result | Configuration observed/requested | Classification |
| --- | --- | --- | --- |
| dense | 22/24 HTTP 200 answers passed; cases 022/023 returned HTTP 400 | target K/V Turbo4; MTP off | 2 invalid setup rows |
| selected-all | 18 HTTP 200 answer mismatches; cases 018–021 timed out; cases 022/023 HTTP 400 | selective pager, 86 requested hot pages, native Turbo4/GPU MTP, resolved MTP rows 22016 | semantic mismatch plus incomplete runtime rows; “selected-all” coverage unverified |
| exact | 22 HTTP 501 refusals; cases 022/023 HTTP 400 | exact pager, native Turbo4/GPU MTP, resolved MTP rows 22016 | missing exact page-wave implementation plus 2 invalid setup rows |
| selective | same 18 mismatches, 4 timeouts, and 2 HTTP 400 rows as selected-all | selective pager, automatic hot pages, native Turbo4/GPU MTP | semantic mismatch plus incomplete runtime rows |

Resolved context was 22,016 tokens. Actual case sizes were 4,096, 5,120, 6,401, 8,192, 9,473, 11,264, 12,800, 14,592, 16,128, 17,920, 19,968, and 22,016 tokens. The server’s boundary error reports the rendered recent-control prompt as 22,028 tokens, so those rows are invalid sizing—not throughput or quality failures. Native MTP placement is startup-confirmed as Turbo4 K/V on GPU; one response-level sample reports 23 proposed and 19 accepted tokens, but full parity and rollback acceptance remain unmeasured.

The startup gate was healthy for 90.578 seconds across 94 health/identity samples with zero gate restarts. The prior SIGSEGV did not reproduce; `coredumpctl` was unavailable and no backtrace exists. `/metrics` returned HTTP 500 (`json.exception.type_error.302`, boolean where number was expected), leaving pager objects empty. Resident/host page counts, H2D/D2H bytes, faults, evictions, route fractions, attention mass, and parity are therefore `not_measured`, never zero.

## Remediation map

| Gap | Class | Owner | Smallest next experiment |
| --- | --- | --- | --- |
| Runtime identity and prior crash | runtime fault, not reproduced | 18-01 | Freeze executable plus loaded DSOs; repeat one immutable 22016 startup and short generation. |
| Rendered-context overflow | invalid setup | 18-02 | Tokenize the fully rendered request with headroom; rerun only recent-control boundary cases. |
| Selected-all answer mismatch | numerical/semantic mismatch | 18-05 | MTP-off identical-prefix all-selected sentinel; capture first divergent layer/logits and validate Turbo4 K/Q domain. |
| Native-MTP parity | not measured semantic risk | 18-06 | Compare MTP off/on target logits and accepted-prefix traces through reject and restore cases. |
| Four long request timeouts | incomplete runtime observation | 18-03 | One 17,920-token focus-shift case with staged deadlines, progress polling, checkpoint, and slot drain. |
| Metrics contract | missing measurement implementation | 18-04 | Fix `/metrics` types and report actual requested/admitted/valid/selected fields on one short generation. |
| Exact page waves | missing implementation | 19-06 | One 4,096-token exact cold wave with GPU online-softmax merge. |
| Physical authority/admission | missing implementation | 19-01/02 | Two-page poisoned/permuted slot fixture plus pre-allocation byte ledger and Turbo4 byte comparison. |
| Live residency transactions | missing implementation | 19-03/07 | One manually selected cold-page eviction/promotion with host checksum, CUDA event completion, and generation recheck. |
| Query shape and paged performance | performance finding, not yet measured | 19-04/05/09, 20-04 | One verify block and bounded prefill tile; capture direct/tiled/split-KV timing and transfer bytes. |
| Layer-aware retrieval | missing implementation | 20-01/02/03 | Real post-Q per-layer query scores all host pages and promotes one cold fact before retention/lookahead. |

The strategy observations from 17-09/17-10 remain diagnostic only because the final campaign could not measure pager fields. The 17-13 crash is not reproduced, not declared fixed. The 17-15 word-padding concern is independently supported by the 22,028-token boundary error and is owned by 18-02.

`WORK_STATE.json` still points to 18-01 as the next task and 22-01 as the final benchmark-only reviewer; future packet and cluster files were checked on the runner checkpoint path. The last healthy 17-15 runtime was retained; 17-16 performed no lifecycle action.
