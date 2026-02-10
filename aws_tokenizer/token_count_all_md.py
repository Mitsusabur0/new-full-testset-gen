import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import boto3


def read_text(path: Path) -> Tuple[str, str]:
    """
    Returns (text, encoding_used). Falls back to latin-1 if utf-8 fails.
    """
    try:
        return path.read_text(encoding="utf-8"), "utf-8"
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1"), "latin-1"


def iter_md_files(root: Path):
    for path in sorted(root.rglob("*.md")):
        if path.is_file():
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Count input tokens for each .md file using Amazon Titan Text Embeddings V2 "
            "and append results to a CSV."
        )
    )
    parser.add_argument("--root", default="gold_full", help="Root folder to scan for .md files")
    parser.add_argument(
        "--out",
        default="outputs/token_counts.csv",
        help="CSV output path (will be appended to)",
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region for Bedrock Runtime")
    parser.add_argument(
        "--model-id",
        default="amazon.titan-embed-text-v2:0",
        help="Bedrock model ID",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = boto3.client("bedrock-runtime", region_name=args.region)

    write_header = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["token_count", "file_name"])

        for md_path in iter_md_files(root):
            text, encoding_used = read_text(md_path)
            if not text.strip():
                print(f"SKIP empty: {md_path}")
                continue

            native_request = {"inputText": text}
            request = json.dumps(native_request)
            response = client.invoke_model(modelId=args.model_id, body=request)
            model_response = json.loads(response["body"].read())
            input_token_count = model_response.get("inputTextTokenCount")

            if input_token_count is None:
                print(f"WARN no token count: {md_path}")
                continue

            writer.writerow([input_token_count, md_path.as_posix()])
            print(f"OK {md_path} -> {input_token_count} tokens ({encoding_used})")

    print(f"Done. Appended results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
