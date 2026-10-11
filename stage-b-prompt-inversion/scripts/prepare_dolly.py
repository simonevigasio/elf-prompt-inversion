#!/usr/bin/env python3
"""Prepare reproducible, complete English prompts from the public Dolly dataset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import urllib.request

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer
from prompt_inversion.elf_adapter import read_settings

REPO = "databricks/databricks-dolly-15k"
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
FILENAME = "databricks-dolly-15k.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/dolly-20-tokens-50.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/source"))
    args = parser.parse_args()
    if not 1 <= args.tokens <= 64 or args.count <= 0 or args.seed < 0:
        parser.error("tokens must be 1–64, count positive, and seed nonnegative")
    spec = read_settings()["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"], local_files_only=True)
    source_url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILENAME}"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.cache_dir / f"{REVISION}-{FILENAME}"
    if not source_path.exists():
        print(f"Downloading {REPO} at {REVISION}", flush=True)
        with urllib.request.urlopen(source_url, timeout=60) as response:
            payload = response.read()
        # Validate the download before caching it.
        for line in payload.splitlines():
            json.loads(line)
        source_path.write_bytes(payload)
    source_bytes = source_path.read_bytes()
    eligible, seen = [], set()
    special = set(tokenizer.all_special_ids)
    total = 0
    for row_index, line in enumerate(source_bytes.splitlines()):
        row = json.loads(line)
        total += 1
        text = row["instruction"]
        # Standalone instructions, with no omitted context, no duplicates or specials.
        if row["context"].strip() or text in seen or len(text) > 10000:
            continue
        ids = tokenizer(text, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        if len(ids) != args.tokens or any(t in special or t >= tokenizer.vocab_size for t in ids):
            continue
        seen.add(text)
        eligible.append({"id": f"dolly-{row_index:05d}", "text": text})
    if len(eligible) < args.count:
        raise ValueError(f"Need {args.count} eligible prompts, found {len(eligible)}")
    selected = sorted(random.Random(args.seed).sample(eligible, args.count), key=lambda row: row["id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode("utf-8")
    args.output.write_bytes(payload)
    manifest = {
        "dataset": REPO, "revision": REVISION, "source_url": source_url,
        "license": "CC-BY-SA-3.0", "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "field": "instruction", "require_empty_context": True, "exclude_special_tokens": True,
        "deduplicate_text": True, "truncate": False, "add_special_tokens": False,
        "tokenizer": spec, "exact_token_count": args.tokens, "seed": args.seed,
        "sampling": "random.Random(seed).sample in source order; selected rows sorted by ID",
        "source_rows": total, "eligible_rows": len(eligible), "selected_rows": len(selected),
        "selected_ids": [row["id"] for row in selected],
        "output_sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(selected)} unique {args.tokens}-token prompts from {len(eligible)} eligible rows to {args.output}")


if __name__ == "__main__":
    main()
