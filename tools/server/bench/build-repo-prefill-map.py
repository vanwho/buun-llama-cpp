#!/usr/bin/env python3
"""Materialize a reusable, model-tokenized, multi-file repository chunk map."""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from repo_tokenizer import NativeTokenizer
from repo_prefill_map import build_map


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=pathlib.Path, required=True)
    p.add_argument("--payload-root", type=pathlib.Path, required=True)
    p.add_argument("--chunk-tokens", type=int, required=True)
    p.add_argument("--capacity-tokens", type=int, default=262144)
    p.add_argument("--model", type=pathlib.Path, default=pathlib.Path("/srv/ai/models/text/current.gguf"))
    p.add_argument("--library-dir", type=pathlib.Path, required=True)
    p.add_argument("--helper", type=pathlib.Path)
    p.add_argument("--manifest", type=pathlib.Path)
    args = p.parse_args()
    h = hashlib.sha256()
    with args.model.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    identity = {"path": str(args.model.resolve()), "sha256": h.hexdigest()}
    with NativeTokenizer(args.model, helper_path=args.helper, library_dir=args.library_dir) as tok:
        m = build_map(args.source_root, args.payload_root, tok,
                      chunk_tokens=args.chunk_tokens, capacity_tokens=args.capacity_tokens,
                      model_identity=identity)
    m["tokenizer"] = {"library_dir": str(args.library_dir.resolve()),
                      "parse_special": True,
                      "helper_source_sha256": hashlib.sha256(
                          pathlib.Path(__file__).with_name("repo_tokenizer.cpp").read_bytes()).hexdigest()}
    (args.payload_root / "map.json").write_text(json.dumps(m, indent=2) + "\n")
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.manifest or args.payload_root / "map.json"),
                      "chunk_tokens": m["chunk_tokens"], "chunks": len(m["chunks"]),
                      "mapped_content_tokens": m["mapped_content_tokens"],
                      "files": len(m["inventory"]), "payload_root": m["payload_root"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
