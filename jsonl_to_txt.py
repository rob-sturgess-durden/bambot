import json
from pathlib import Path

INPUT = Path("resources_full.jsonl")
OUTPUT = Path("resources_full.txt")


def main() -> None:
    lines = []
    if not INPUT.exists():
        raise SystemExit(f"Missing input file: {INPUT}")
    with INPUT.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            title = obj.get("title", "").strip()
            url = obj.get("url", "").strip() or obj.get("source", "").strip()
            summary = obj.get("summary", "").strip()
            text = obj.get("text", "").strip()
            block = []
            if title:
                block.append(title)
            if summary:
                block.append(summary)
            if url:
                block.append(f"Source: {url}")
            if text:
                block.append(text)
            lines.append("\n".join(block))
    OUTPUT.write_text("\n\n\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

