#!/usr/bin/env python3
"""Join explicitly supplied patched-master GPU/CPU reports without mutating a freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-json", type=Path, required=True)
    parser.add_argument("--cpu-json", type=Path, required=True)
    parser.add_argument("--freeze-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    inputs = {"gpu": load(args.gpu_json), "cpu": load(args.cpu_json),
              "freeze": load(args.freeze_json)}
    hashes = {name: digest(path) for name, path in (
        ("gpu", args.gpu_json), ("cpu", args.cpu_json), ("freeze", args.freeze_json))}
    expected_gpu = {77824, 8192, 16384, 32768, 65536}
    expected_cpu = expected_gpu | {131072}
    for name, expected in (("gpu", expected_gpu), ("cpu", expected_cpu)):
        report = inputs[name]
        if report.get("cohort") != "master106-draftkv-20261013":
            raise SystemExit(f"{name}: wrong replacement cohort")
        rows = report.get("rows", [])
        contexts = {row.get("geometry", {}).get("L_tokens") for row in rows}
        if contexts != expected or len(rows) != len(expected):
            raise SystemExit(f"{name}: incomplete or duplicate context rows: {contexts}")
        for row in rows:
            if row.get("completion", {}).get("unique_cases") != 48:
                raise SystemExit(f"{name}/{row.get('label')}: expected 48 canonical cases")
            if row.get("runtime_identity", {}).get("executable_sha256") != report.get("build", {}).get("executable_sha256"):
                raise SystemExit(f"{name}/{row.get('label')}: executable identity mismatch")
    gpu_id = inputs["gpu"].get("build", {})
    cpu_id = inputs["cpu"].get("build", {})
    if gpu_id != cpu_id:
        raise SystemExit("paired GPU and CPU reports do not name the same frozen build")
    if inputs["freeze"].get("cohort") != "master106-draftkv-20261013":
        raise SystemExit("replacement freeze cohort identity mismatch")
    payload = {"schema_version": 1, "cohort": "master106-draftkv-20261013",
               "base_master_commit": inputs["freeze"]["base_master_commit"],
               "model_sha256": inputs["freeze"]["model_sha256"],
               "build": gpu_id, "inputs": hashes,
               "gpu_rows": inputs["gpu"]["rows"], "cpu_rows": inputs["cpu"]["rows"]}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Patched master 106 paired cohort", "",
             f"- Cohort: `{payload['cohort']}`",
             f"- Base master: `{payload['base_master_commit']}`",
             f"- Model SHA-256: `{payload['model_sha256']}`",
             f"- Frozen executable SHA-256: `{gpu_id['executable_sha256']}`", "",
             "| Placement | L | C | Cases |", "|---|---:|---:|---:|"]
    for placement in ("gpu", "cpu"):
        for row in inputs[placement]["rows"]:
            geo = row["geometry"]
            lines.append(f"| {placement} | {geo['L_tokens']} | {geo.get('C_tokens', 'empty')} | {row['completion']['unique_cases']} |")
    lines += ["", "Input SHA-256:"]
    lines += [f"- {name}: `{value}`" for name, value in hashes.items()]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
