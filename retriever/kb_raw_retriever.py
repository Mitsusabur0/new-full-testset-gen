import os
import json
import time
import random
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

# -----------------------------
# CONFIGURATION (edit as needed)
# -----------------------------
AWS_PROFILE = "sandbox"
AWS_REGION = "us-east-1"
KB_SERVICE = "bedrock-agent-runtime"
KB_ID = "J7JNHSZPJ3"

# Query and retrieval settings
QUERY = "que es casaverso?"
TOP_K = 3

# Retry settings
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 8.0
BACKOFF_JITTER_SECONDS = 0.3


def get_runtime_client():
    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client(service_name=KB_SERVICE, region_name=AWS_REGION)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def backoff_sleep(attempt):
    base = BACKOFF_BASE_SECONDS * (2 ** attempt)
    sleep_for = min(base, BACKOFF_MAX_SECONDS)
    sleep_for += random.uniform(0, BACKOFF_JITTER_SECONDS)
    time.sleep(sleep_for)


def call_with_retry(fn, operation_name, error_log):
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except ClientError as e:
            last_error = e
        except Exception as e:
            last_error = e

        if attempt < MAX_RETRIES:
            backoff_sleep(attempt)
        else:
            if last_error is not None:
                error_log.append(
                    {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "operation": operation_name,
                        "error": str(last_error),
                    }
                )
            return None


def retrieve_raw_response(query_text, top_k_value, client, error_log):
    def _call():
        return client.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query_text},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": top_k_value}
            },
        )

    return call_with_retry(_call, "retrieve", error_log)


def main():
    if not QUERY.strip():
        print("Set the 'QUERY' variable at the top of this file before running.")
        return

    client = get_runtime_client()
    error_log = []

    output_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(output_dir, "kb_raw_response.json")
    summary_path = os.path.join(output_dir, "retriever_raw_run_summary.json")
    ensure_parent_dir(raw_path)

    print(f"Retrieving from KB for query: {QUERY[:60]}...")
    response = retrieve_raw_response(QUERY, TOP_K, client, error_log)

    with open(raw_path, "w", encoding="utf-8") as raw_file:
        json.dump(
            {"query": QUERY, "top_k": TOP_K, "response": response},
            raw_file,
            ensure_ascii=False,
            indent=2,
        )

    if error_log:
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(
                {"retrieved": 1, "errors": error_log},
                summary_file,
                ensure_ascii=False,
                indent=2,
            )

    print(f"Saved raw response to {raw_path}")


if __name__ == "__main__":
    main()
