"""Prepare reproducible, complete English prompts from the public Dolly dataset.

    python -m experiments.prepare_dolly --tokens 20 --count 50

Downloads a pinned dataset revision (cached in data/dolly/source), keeps
standalone instructions of exactly --tokens tokenizer tokens with no special
tokens, samples --count of them with --seed, and writes
data/dolly/dolly-<tokens>-tokens-<count>.jsonl plus a provenance manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import urllib.request

from transformers import AutoTokenizer

from elf_model.adapter import read_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "dolly"
REPO = "databricks/databricks-dolly-15k"
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
FILENAME = "databricks-dolly-15k.jsonl"
SOURCE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILENAME}"


def download_source(cache_dir):
    """Pinned Dolly JSONL, downloaded once and validated before caching."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{REVISION}-{FILENAME}"
    if not path.exists():
        print(f"Downloading {REPO} at {REVISION}", flush=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            payload = response.read()
        for line in payload.splitlines():
            json.loads(line)
        path.write_bytes(payload)
    return path.read_bytes()


def eligible_prompts(source_bytes, tokenizer, tokens):
    """Standalone instructions (no separate context), unique, exactly ``tokens`` long."""
    eligible, seen = [], set()
    special = set(tokenizer.all_special_ids)
    rows = source_bytes.splitlines()
    for row_index, line in enumerate(rows):
        row = json.loads(line)
        text = row["instruction"]
        if row["context"].strip() or text in seen or len(text) > 10000:
            continue
        ids = tokenizer(text, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        if len(ids) != tokens or any(t in special or t >= tokenizer.vocab_size for t in ids):
            continue
        seen.add(text)
        eligible.append({"id": f"dolly-{row_index:05d}", "text": text})
    return eligible, len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a fixed-length prompt dataset from Dolly.")
    parser.add_argument("--tokens", type=int, default=20, help="Exact tokenizer length of each prompt (1-64)")
    parser.add_argument("--count", type=int, default=50, help="Number of prompts to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Default: data/dolly/dolly-<tokens>-tokens-<count>.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=DATA_DIR / "source")
    args = parser.parse_args(argv)
    if not 1 <= args.tokens <= 64 or args.count <= 0 or args.seed < 0:
        parser.error("tokens must be 1-64, count positive, and seed nonnegative")
    output = args.output or DATA_DIR / f"dolly-{args.tokens}-tokens-{args.count}.jsonl"

    spec = read_settings()["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    source_bytes = download_source(args.cache_dir)
    eligible, total = eligible_prompts(source_bytes, tokenizer, args.tokens)
    if len(eligible) < args.count:
        raise ValueError(f"Need {args.count} eligible prompts, found {len(eligible)}")

    selected = sorted(random.Random(args.seed).sample(eligible, args.count), key=lambda row: row["id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode("utf-8")
    output.write_bytes(payload)
    manifest = {
        "dataset": REPO, "revision": REVISION, "source_url": SOURCE_URL,
        "license": "CC-BY-SA-3.0", "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "field": "instruction", "require_empty_context": True, "exclude_special_tokens": True,
        "deduplicate_text": True, "truncate": False, "add_special_tokens": False,
        "tokenizer": spec, "exact_token_count": args.tokens, "seed": args.seed,
        "sampling": "random.Random(seed).sample in source order; selected rows sorted by ID",
        "source_rows": total, "eligible_rows": len(eligible), "selected_rows": len(selected),
        "selected_ids": [row["id"] for row in selected],
        "output_sha256": hashlib.sha256(payload).hexdigest(),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(selected)} unique {args.tokens}-token prompts from {len(eligible)} eligible rows to {output}")


if __name__ == "__main__":
    main()
