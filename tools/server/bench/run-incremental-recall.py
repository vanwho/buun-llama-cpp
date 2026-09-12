#!/usr/bin/env python3
"""Continue a completed incremental V7 run with one natural cold-topic query."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("run_final_curve", HERE / "run-final-curve.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load V7 request driver")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    key = next(line.strip() for line in args.api_key_file.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#"))
    receipt = json.loads((args.output / "INTERACTIVE27_01_SCALE.json").read_text())
    records = receipt["raw"]["records"]
    prior_request = json.loads((args.output / "request-17.json").read_text())
    messages = list(prior_request["messages"])
    messages.append({"role": "assistant", "content": records[-1]["response"]["content"]})
    messages.append({"role": "user", "content": (
        "Cold-topic shift: answer only this recall question. What exact retrieval "
        "topic was established in the earliest history?" )})
    request_path = args.output / "request-18-cold-topic.json"
    request_path.write_text(json.dumps({**prior_request, "messages": messages}, indent=2) + "\n")
    record = driver.run_request(
        args.endpoint, key, args.model, messages, 128, 32768, "scale", 18, 1,
        24580, 600, args.output / "raw-18-cold-topic.sse",
        cache_condition="live-continuation", mode="selective", prefill_policy="runtime",
        startup_timeout=180, progress_idle_timeout=600, decode_idle_timeout=120,
        total_timeout=1200,
    )
    record["request_path"] = str(request_path)
    (args.output / "recall-18.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record.get("status"), "prompt_tokens": record.get("usage", {}).get("prompt_tokens"),
                      "cached_tokens": record.get("cached_rows"), "output_tokens": record.get("output_tokens"),
                      "decode_tok_s": record.get("server_tg_tok_s"),
                      "movement_delta": record.get("movement_delta")}), flush=True)
    return 0 if record.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
