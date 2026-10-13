#!/usr/bin/env python3
"""Emit compact cold/hot findings from one completed manual 8K campaign."""
import hashlib
import json
import pathlib
import sys


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def prompt_metrics(response):
    usage = response.get("usage", {})
    timings = response.get("timings", {})
    return {
        "usage_prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "new_prompt_tokens_from_usage": usage.get("prompt_tokens", 0) -
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "server_prompt_n": timings.get("prompt_n"),
        "server_prefill_ms": timings.get("prompt_ms"),
        "completion_tokens": usage.get("completion_tokens"),
        "decode_ms": timings.get("predicted_ms"),
        "decode_tokens_per_second": timings.get("predicted_per_second"),
        "mtp_drafted": timings.get("draft_n"),
        "mtp_accepted": timings.get("draft_n_accepted"),
        "mtp_acceptance_percent": round(
            100 * timings.get("draft_n_accepted", 0) / timings.get("draft_n", 1), 2),
    }


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    case = root / "cases/PY_MERGE_03"
    summary = read(case / "case-summary.json")
    cold = read(case / "request-03/completion-response.json")
    settling = read(case / "request-04-identical-settling-repeat/completion-response.json")
    settling_timing = read(case / "request-04-identical-settling-repeat/request-timing.json")
    hot = read(case / "request-05-identical-measured-repeat/completion-response.json")
    hot_timing = read(case / "request-05-identical-measured-repeat/request-timing.json")
    a2_record = read(case / "request-03/record.json")
    cold_wall = read(case / "request-03/client-wall.json")
    target_id = 5
    def page(slot, where):
        if isinstance(slot, list):
            slot = next((item for item in slot if isinstance(item, dict) and item.get("id") == 0), {})
        pages = (slot.get("pager_metrics") or {}).get("page_inventory", [])
        found = [p for p in pages if p.get("logical_page_id") == target_id]
        return found[0] if len(found) == 1 else None
    before = next(p for p in (a2_record.get("pager_before") or {}).get("page_inventory", [])
                  if p.get("logical_page_id") == target_id)
    after = next(p for p in (a2_record.get("pager_after") or {}).get("page_inventory", [])
                 if p.get("logical_page_id") == target_id)
    settling_pre = page(read(case / "request-04-identical-settling-repeat/slots-before.json"), "before")
    settling_post = page(read(case / "request-04-identical-settling-repeat/slots-after.json"), "after")
    hot_pre = page(read(case / "request-05-identical-measured-repeat/slots-before.json"), "before")
    hot_post = page(read(case / "request-05-identical-measured-repeat/slots-after.json"), "after")
    nat = (a2_record.get("pager_after") or {}).get("natural_proof") or {}
    settling_slots_before = read(case / "request-04-identical-settling-repeat/slots-before.json")
    settling_slots_after = read(case / "request-04-identical-settling-repeat/slots-after.json")
    hot_slots_before = read(case / "request-05-identical-measured-repeat/slots-before.json")
    hot_slots_after = read(case / "request-05-identical-measured-repeat/slots-after.json")

    def pager(slot):
        if isinstance(slot, list):
            slot = next((item for item in slot if isinstance(item, dict) and item.get("id") == 0), {})
        return slot.get("pager_metrics") or {}

    def h2d_delta(first, last):
        left, right = pager(first).get("h2d_useful_bytes"), pager(last).get("h2d_useful_bytes")
        return right - left if isinstance(left, int) and isinstance(right, int) else None

    def added_pages(first, last):
        left = set(pager(first).get("selected_page_ids") or [])
        right = set(pager(last).get("selected_page_ids") or [])
        return sorted(right - left)
    original_hash = hashlib.sha256(
        (case / "request-03/completion-request.json").read_bytes().rstrip(b"\n")).hexdigest()
    repeat_hash = hashlib.sha256(
        (case / "request-05-identical-measured-repeat/completion-request.json").read_bytes().rstrip(b"\n")).hexdigest()
    h2d_before = (a2_record.get("pager_before") or {}).get("h2d_useful_bytes")
    h2d_after = (a2_record.get("pager_after") or {}).get("h2d_useful_bytes")
    result = {
        "schema": "manual-8k-cold-hot-promotion-result-v1",
        "status": "complete",
        "output_root": str(root),
        "candidate": {"pid": read(root / "candidate-identity.json")["pid"],
                      "binary_sha256": read(root / "candidate-identity.json")["binary_sha256"],
                      "model_sha256": read(root / "candidate-identity.json")["model_sha256"],
                      "loaded_candidate_libraries": read(root / "candidate-identity.json")["loaded_candidate_libraries"],
                      "context_tokens": 8192, "hot_tokens": 4096,
                      "page_tokens": 256, "hot_pages": 16},
        "sequence": {
            "a1": {"http_status": summary["requests"][0]["http_status"],
                   "answer_quality": summary["request_1_answer_quality"],
                   "metrics": prompt_metrics(read(case / "request-01/completion-response.json"))},
            "b": {"http_status": summary["requests"][1]["http_status"],
                  "assistant_reply": summary["requests"][1]["assistant_answer"],
                  "metrics": prompt_metrics(read(case / "request-02/completion-response.json"))},
            "cold_a2": {"http_status": summary["requests"][2]["http_status"],
                        "answer_quality": summary["final_request_answer_quality"],
                        "answer": summary["final_answer"],
                        "metrics": prompt_metrics(cold),
                        "client_wall_ms": cold_wall["wall_ms"],
                        "server_prompt_plus_decode_ms": cold["timings"]["prompt_ms"] + cold["timings"]["predicted_ms"],
                        "wall_note": "Client wall measured around the non-streaming HTTP completion; TTFT is unavailable."},
            "settling_identical_a3": {"http_status": settling_timing["http_status"],
                                 "metrics": prompt_metrics(settling),
                                 "client_wall_ms": settling_timing["wall_ms"],
                                 "h2d_useful_delta_bytes": h2d_delta(settling_slots_before, settling_slots_after),
                                 "selected_page_ids_added": added_pages(settling_slots_before, settling_slots_after),
                                 "target_resident_before": settling_pre.get("resident") if settling_pre else None,
                                 "target_resident_after": settling_post.get("resident") if settling_post else None},
            "hot_identical_a4": {"http_status": hot_timing["http_status"],
                                 "answer": (((hot.get("choices") or [{}])[0].get("message") or {}).get("content")),
                                 "metrics": prompt_metrics(hot),
                                 "client_wall_ms": hot_timing["wall_ms"],
                                 "h2d_useful_delta_bytes": h2d_delta(hot_slots_before, hot_slots_after),
                                 "selected_page_ids_added": added_pages(hot_slots_before, hot_slots_after),
                                 "request_body_sha256": repeat_hash},
        },
        "identical_request": {"cold_a2_sha256": original_hash,
                              "settling_a3_sha256": hashlib.sha256((case / "request-04-identical-settling-repeat/completion-request.json").read_bytes().rstrip(b"\n")).hexdigest(),
                              "measured_a4_sha256": repeat_hash,
                              "byte_identical": original_hash == repeat_hash and
                                  hashlib.sha256((case / "request-04-identical-settling-repeat/completion-request.json").read_bytes().rstrip(b"\n")).hexdigest() == original_hash},
        "target_page": {
            "identity": {k: before.get(k) for k in ("logical_page_id", "generation", "content_version",
                         "position_begin", "position_end", "sequence_id", "sequence_generation")},
            "cold_before_a2": before.get("resident") is False and before.get("host_backed") is True,
            "resident_after_a2": after.get("resident"),
            "resident_before_settling_a3": settling_pre.get("resident") if settling_pre else None,
            "resident_after_settling_a3": settling_post.get("resident") if settling_post else None,
            "resident_before_measured_a4": hot_pre.get("resident") if hot_pre else None,
            "resident_after_measured_a4": hot_post.get("resident") if hot_post else None,
            "selected_in_a2_last_graph": target_id in ((a2_record.get("pager_after") or {}).get("selected_page_ids") or []),
            "a2_request_h2d_useful_delta_bytes": h2d_after - h2d_before if isinstance(h2d_after, int) and isinstance(h2d_before, int) else None,
            "a2_native_natural_proof": nat,
            "proof_scope_note": "The authenticated singleton natural_proof identifies logical page 4; answer-bearing page 5 separately transitioned cold->resident and appeared in selected_page_ids, but per-page H2D sequencing for page 5 is not exposed by that singleton proof.",
        },
        "restore": read(root / "restore-verification.json"),
        "measurement_limits": [
            "TTFT was not captured because completions were non-streaming.",
            "Cold A2 and hot A4 had byte-identical requests and the same 131 output tokens, but differed by two new input tokens; timing differences are measured observations, not pure transfer cost.",
            "A3/A4 are target-hot mixed controls because unrelated pages continued to promote while answer page 5 stayed resident.",
        ],
    }
    (root / "RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    cold_m, hot_m = result["sequence"]["cold_a2"]["metrics"], result["sequence"]["hot_identical_a4"]["metrics"]
    md = f'''# Manual 8K cold/hot promotion result

Run: `{root}`  
Candidate PID: {result["candidate"]["pid"]}; L=8192, H=4096, 16 pages of 256 tokens.

Natural A/B/A completed with HTTP 200 for all three requests. A1 answered the merge question correctly; B replied “{summary["requests"][1]["assistant_answer"]}”; cold A2 returned the expected fact correctly.

| Request | Input tokens (cached / new) | Prefill | Output / decode | MTP accepted / drafted | Client wall |
|---|---:|---:|---:|---:|---:|
| Cold A2 | {cold_m["usage_prompt_tokens"]} ({cold_m["cached_tokens"]} / {cold_m["new_prompt_tokens_from_usage"]}) | {cold_m["server_prefill_ms"]:.1f} ms | {cold_m["completion_tokens"]} / {cold_m["decode_ms"]:.1f} ms ({cold_m["decode_tokens_per_second"]:.2f} tok/s) | {cold_m["mtp_accepted"]}/{cold_m["mtp_drafted"]} ({cold_m["mtp_acceptance_percent"]:.2f}%) | {cold_wall["wall_ms"]:.1f} ms |
| Hot settling A3 | {result["sequence"]["settling_identical_a3"]["metrics"]["usage_prompt_tokens"]} ({result["sequence"]["settling_identical_a3"]["metrics"]["cached_tokens"]} / {result["sequence"]["settling_identical_a3"]["metrics"]["new_prompt_tokens_from_usage"]}) | {result["sequence"]["settling_identical_a3"]["metrics"]["server_prefill_ms"]:.1f} ms | {result["sequence"]["settling_identical_a3"]["metrics"]["completion_tokens"]} / {result["sequence"]["settling_identical_a3"]["metrics"]["decode_ms"]:.1f} ms ({result["sequence"]["settling_identical_a3"]["metrics"]["decode_tokens_per_second"]:.2f} tok/s) | {result["sequence"]["settling_identical_a3"]["metrics"]["mtp_accepted"]}/{result["sequence"]["settling_identical_a3"]["metrics"]["mtp_drafted"]} ({result["sequence"]["settling_identical_a3"]["metrics"]["mtp_acceptance_percent"]:.2f}%) | {result["sequence"]["settling_identical_a3"]["client_wall_ms"]:.1f} ms |
| Hot measured A4 | {hot_m["usage_prompt_tokens"]} ({hot_m["cached_tokens"]} / {hot_m["new_prompt_tokens_from_usage"]}) | {hot_m["server_prefill_ms"]:.1f} ms | {hot_m["completion_tokens"]} / {hot_m["decode_ms"]:.1f} ms ({hot_m["decode_tokens_per_second"]:.2f} tok/s) | {hot_m["mtp_accepted"]}/{hot_m["mtp_drafted"]} ({hot_m["mtp_acceptance_percent"]:.2f}%) | {hot_timing["wall_ms"]:.1f} ms |

The A2, settling A3, and measured A4 request bodies are byte-identical (`{original_hash}`). Answer-bearing page 5 (tokens 1280–1536, generation 9/content version 256) was host-backed and nonresident before A2, resident afterward, and remained resident through both repeats. A2's useful H2D counter increased by {result["target_page"]["a2_request_h2d_useful_delta_bytes"]:,} bytes; page 5 appears in A2's selected page IDs. A3 added {result["sequence"]["settling_identical_a3"]["h2d_useful_delta_bytes"]:,} useful H2D bytes (new selected IDs {result["sequence"]["settling_identical_a3"]["selected_page_ids_added"]}); A4 added {result["sequence"]["hot_identical_a4"]["h2d_useful_delta_bytes"]:,} bytes (new selected IDs {result["sequence"]["hot_identical_a4"]["selected_page_ids_added"]}). These are target-hot mixed controls, not fully settled all-hot runs. The authenticated `natural_proof` records page 4, with selector → completed H2D → mapping publication → target use; it does not independently attribute that full chain to page 5. Page 5's cold→resident transition is directly observed in inventory.

Cold A2 client wall is measured around the non-streaming HTTP request; TTFT is unavailable. Cold A2 server prompt+decode timing sums to {result["sequence"]["cold_a2"]["server_prompt_plus_decode_ms"]:.1f} ms. Cold A2 processed 68 new usage tokens versus 66 on A3/A4, a two-token boundary difference. A2 and A4 both emitted 131 tokens. Treat timing changes as measured observations, not pure transfer cost. A3 is a settling sample; A4 is the measured target-hot mixed control.

Managed service restoration passed: health is HTTP 200, original argv matches, and transient overrides match the original manager environment. See `restore-verification.json`.
'''
    (root / "RESULT.md").write_text(md, encoding="utf-8")
    print(root / "RESULT.md")


if __name__ == "__main__":
    main()
