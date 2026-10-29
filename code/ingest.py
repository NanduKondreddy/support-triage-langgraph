from __future__ import annotations

import argparse
import json
from pathlib import Path

from corpus_loader import load_corpus
from retriever import tokenize


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a lightweight inspectable corpus index.")
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "data" / "index" / "corpus_index.jsonl"))
    args = parser.parse_args()

    docs = load_corpus(Path(args.data))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for doc in docs:
            record = {
                "domain": doc.domain,
                "product_area": doc.product_area,
                "path": doc.path,
                "title": doc.title,
                "tokens": tokenize(f"{doc.title} {doc.text}")[:800],
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"Wrote {len(docs)} indexed documents to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
