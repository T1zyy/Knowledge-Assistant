"""Run the same synthetic question set against a running assistant."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
MIME_TYPES = {".txt": "text/plain", ".md": "text/markdown", ".pdf": "application/pdf"}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", default=os.getenv("AI_PROVIDER", "demo"))
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    api_key = os.getenv("API_KEY")
    if not api_key:
        parser.error("Set API_KEY in the environment or .env")

    questions = [
        json.loads(line)
        for line in (ROOT / "eval" / "questions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    with httpx.Client(
        base_url=args.base_url,
        timeout=90,
        headers={"X-API-Key": api_key},
    ) as client:
        for path in sorted((ROOT / "examples").iterdir()):
            if path.suffix in MIME_TYPES:
                response = client.post(
                    "/documents",
                    files={"file": (path.name, path.read_bytes(), MIME_TYPES[path.suffix])},
                    data={"allow_external": str(args.allow_external).lower()},
                )
                response.raise_for_status()
        records = []
        for case in questions:
            started = time.monotonic()
            response = client.post(
                "/ask",
                json={"question": case["question"], "allow_external": args.allow_external},
            )
            response.raise_for_status()
            elapsed_ms = round(1000 * (time.monotonic() - started))
            result = response.json()
            sources = [source["filename"] for source in result["sources"]]
            if case["source"] is None:
                correct = result["status"] == "insufficient_context"
            else:
                needle = case["contains"].casefold().replace("\u00a0", "").replace(" ", "")
                answer = result["answer"].casefold().replace("\u00a0", "").replace(" ", "")
                correct = case["source"] in sources and needle in answer
            records.append(
                {
                    "question": case["question"],
                    "expected_source": case["source"],
                    "sources": sources,
                    "answer": result["answer"],
                    "pass": correct,
                    "elapsed_ms": elapsed_ms,
                    "usage": result["usage"],
                    "cache_hit": result["cache_hit"],
                }
            )
    report = {
        "label": args.label,
        "passed": sum(record["pass"] for record in records),
        "total": len(records),
        "median_ms": statistics.median(record["elapsed_ms"] for record in records),
        "cache_hits": sum(record["cache_hit"] for record in records),
        "cases": records,
    }
    formatted = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted + "\n")
    print(formatted)


if __name__ == "__main__":
    main()
