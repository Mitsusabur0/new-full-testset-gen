import os
import json
import time
import random
from datetime import datetime

import pandas as pd
import boto3
from botocore.exceptions import ClientError

import config


def get_runtime_client():
    session = boto3.Session(profile_name=config.AWS_PROFILE_SANDBOX)
    return session.client(service_name=config.KB_SERVICE, region_name=config.AWS_REGION)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def backoff_sleep(attempt):
    base = config.BACKOFF_BASE_SECONDS * (2 ** attempt)
    sleep_for = min(base, config.BACKOFF_MAX_SECONDS)
    sleep_for += random.uniform(0, config.BACKOFF_JITTER_SECONDS)
    time.sleep(sleep_for)


def call_with_retry(fn, operation_name, error_log):
    last_error = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            return fn()
        except ClientError as e:
            last_error = e
        except Exception as e:
            last_error = e

        if attempt < config.MAX_RETRIES:
            backoff_sleep(attempt)
        else:
            if last_error is not None:
                error_log.append({
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "operation": operation_name,
                    "error": str(last_error),
                })
            return None


def retrieve_raw_response(query, client, error_log):
    def _call():
        return client.retrieve(
            knowledgeBaseId=config.KB_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": config.TOP_K
                }
            },
        )

    return call_with_retry(_call, "retrieve", error_log)


def main():
    print(f"Loading {config.OUTPUT_TESTSET_CSV}...")
    try:
        df = pd.read_csv(config.OUTPUT_TESTSET_CSV)
    except FileNotFoundError:
        print("Input file not found. Run File 1 first.")
        return

    client = get_runtime_client()
    error_log = []

    output_dir = os.path.dirname(config.OUTPUT_EVALSET_CSV) or "."
    raw_path = os.path.join(output_dir, "kb_raw_responses.jsonl")
    ensure_parent_dir(raw_path)

    print(f"Saving raw KB responses to {raw_path} ...")
    with open(raw_path, "w", encoding="utf-8") as raw_file:
        for index, row in df.iterrows():
            query = row["user_input"]
            print(f"[{index+1}/{len(df)}] Retrieving: {query[:30]}...")

            response = retrieve_raw_response(query, client, error_log)
            raw_file.write(json.dumps({
                "index": index,
                "query": query,
                "response": response,
            }, ensure_ascii=False) + "\n")

    if error_log:
        summary_path = os.path.join(output_dir, "retriever_raw_run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump({
                "retrieved": len(df),
                "errors": error_log,
            }, summary_file, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
