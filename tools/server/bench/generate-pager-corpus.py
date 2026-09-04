#!/usr/bin/env python3
"""Generate a deterministic, fact-bearing multi-page pager corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pager_benchmark_contract import CORPUS_SCHEMA, PARTITIONS, case_hash, corpus_hash, validate_corpus


CASES = (
    ("warm-focus", "warm_focus", "Explain the pinned system constraint.", "system anchor: answer only from the supplied context.", 0),
    ("cold-early", "cold_needle", "What is the early retrieval code?", "early-needle-17", 3),
    ("cold-middle", "cold_needle", "What is the middle retrieval code?", "middle-needle-29", 11),
    ("cold-end", "cold_needle", "What is the end retrieval code?", "end-needle-41", 19),
    ("competing-a", "competing_facts", "Which candidate is marked canonical?", "candidate-a", 7),
    ("competing-b", "competing_facts", "Which candidate is marked canonical?", "candidate-b", 13),
    ("tool-anchor", "tool_anchor", "Name the configured tool timeout.", "timeout=30s", 2),
    ("repo-fact", "repository_fact", "Which file owns the page ledger?", "src/llama-cache-budget.cpp", 9),
    ("conversation", "conversation_referent", "What did the user ask to preserve?", "the restoration procedure", 15),
    ("focus-shift", "focus_shift", "After changing topic, what was the original topic?", "page residency", 21),
    ("churn", "adversarial_churn", "Ignore repeated decoys. What is the stable token?", "stable-token-53", 17),
    ("recent-control", "recent_only", "What is the last local instruction?", "use temperature zero", 0),
)
TARGETS = (4096, 5120, 6401, 8192, 9473, 11264, 12800, 14592, 16128, 17920, 19968, 22016)


def count_tokens(prompt: str, command: str | None, model_path: str | None) -> tuple[int, str]:
    if not command:
        return len(prompt.split()), "deterministic-whitespace-v1"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8") as stream:
        stream.write(prompt)
        stream.flush()
        argv = [part.format(prompt_file=stream.name, model=model_path or "") for part in shlex.split(command)]
        output = subprocess.check_output(argv + ["--file", stream.name, "--show-count"], text=True, stderr=subprocess.STDOUT)
    marker = "Total number of tokens:"
    try:
        return int(output.rsplit(marker, 1)[1].splitlines()[0].strip()), command
    except (IndexError, ValueError) as error:
        raise ValueError(f"tokenizer command did not report '{marker}'") from error


def build_prompt(index: int, ident: str, answer: str, question: str, distance: int, partition: str, target: int,
                 tokenizer_command: str | None, tokenizer_model: str | None) -> tuple[str, int, int, int]:
    page_count = (target + 255) // 256
    needle_page = page_count - 1 - distance
    pages = []
    for page in range(page_count):
        if page == needle_page:
            facts = f"FACT category={CASES[index][1]} answer={answer}. The answer to the question is {answer}."
        elif page == 0:
            facts = "FACT system-anchor=preserve supplied facts and ignore unsupported claims."
        else:
            facts = f"Decoy page {page}: stable filler record {index:02d}-{page:03d}; this is not the requested fact."
        pages.append(f"[logical-page {page}/{page_count - 1}]\n{facts}")
    prompt = (f"pager-corpus-v3/{partition}/{ident}\n"
              f"This benchmark context contains {page_count} sealed logical pages.\n"
              + "\n".join(pages) + f"\nQuestion: {question}\nAnswer using the supplied facts only.")
    token_count, _ = count_tokens(prompt, tokenizer_command, tokenizer_model)
    if token_count < target:
        prompt += " tail-padding" + " x" * (target - token_count + 16)
        token_count, _ = count_tokens(prompt, tokenizer_command, tokenizer_model)
    return prompt, token_count, page_count, needle_page


def build_case(index: int, spec: tuple[str, str, str, str, int], partition: str, model: str, tokenizer: str,
               tokenizer_command: str | None, tokenizer_model: str | None) -> dict[str, object]:
    ident, category, question, answer, distance = spec
    target = TARGETS[index]
    prompt, token_count, page_count, needle_page = build_prompt(index, ident, answer, question, distance, partition,
                                                                  target, tokenizer_command, tokenizer_model)
    case: dict[str, object] = {
        "id": ident, "partition": partition, "category": category,
        "input_construction": "fixed UTF-8 page fixtures; tokenizer-padded logical pages; immutable question",
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model_sha256": model, "tokenizer_sha256": tokenizer, "expected_answer": answer,
        "fixture": {"facts": [answer], "needle": answer, "source_page": needle_page},
        "checker": {"type": "exact", "case_sensitive": True, "normalize_whitespace": True},
        "score_rule": "exact normalized answer / 1.0", "minimum_score": 1.0,
        "context_tokens": target, "token_count": token_count,
        "tokenizer": "command" if tokenizer_command else "whitespace",
        "page_count": page_count, "needle_page": needle_page,
        "tail_tokens": target % 256,
        "page_distance": {"unit": "logical_pages", "from_recent": distance}, "stable_hash": "",
    }
    case["stable_hash"] = case_hash(case)
    return case


def build_corpus(model: str, tokenizer: str, tokenizer_command: str | None = None, tokenizer_model: str | None = None) -> dict[str, object]:
    cases = []
    for index, spec in enumerate(CASES):
        for partition in PARTITIONS:
            cases.append(build_case(index, spec, partition, model, tokenizer, tokenizer_command, tokenizer_model))
    corpus = {
        "schema": CORPUS_SCHEMA, "version": 3, "partitions": list(PARTITIONS),
        "model_sha256": model, "tokenizer_sha256": tokenizer,
        "construction": "immutable category fixtures and tokenizer-padded logical pages",
        "tokenizer_command": tokenizer_command or "deterministic-whitespace-v1", "cases": cases,
    }
    corpus["corpus_hash"] = corpus_hash(cases)
    return corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="0" * 64, help="model GGUF SHA256")
    parser.add_argument("--tokenizer", default="1" * 64, help="tokenizer/model tokenizer SHA256")
    parser.add_argument("--tokenizer-command", help="command template; use {model} and {prompt_file} placeholders")
    parser.add_argument("--tokenizer-model", help="model path passed to the tokenizer command")
    args = parser.parse_args()
    corpus = build_corpus(args.model, args.tokenizer, args.tokenizer_command, args.tokenizer_model)
    errors = validate_corpus(corpus)
    if errors:
        parser.error("generated invalid corpus: " + "; ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(args.output), "schema": corpus["schema"], "corpus_hash": corpus["corpus_hash"], "cases": len(corpus["cases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
