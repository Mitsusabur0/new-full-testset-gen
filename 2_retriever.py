# retriever.py
import os
import json
import pandas as pd
import boto3
import ast
import random
import time
from datetime import datetime
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

def clean_text(text):
    """Helper to clean retrieved text for better comparison."""
    if not text:
        return ""
    # Remove excessive whitespace, newlines, etc.
    return " ".join(text.split())

def extract_s3_uri(uri):
    return uri or ""

def retrieve_contexts(query, client, error_log):
    def _call():
        return client.retrieve(
            knowledgeBaseId=config.KB_ID,
            retrievalQuery={
                'text': query
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': config.TOP_K
                }
            }
        )

    response = call_with_retry(_call, "retrieve", error_log)
    if response is None:
        print(f"Retrieval Error for query '{query}': exhausted retries")
        return []

    results = response.get('retrievalResults', [])
    retrieved_texts = []
    retrieved_files = []
    for res in results:
        retrieved_texts.append(clean_text(res['content']['text']))
        uri = (
            res.get('location', {})
               .get('s3Location', {})
               .get('uri', "")
        )
        retrieved_files.append(extract_s3_uri(uri))
    return retrieved_texts, retrieved_files

def main():
    print(f"Loading {config.PIPELINE_CSV}...")
    try:
        df = pd.read_csv(config.PIPELINE_CSV)
    except FileNotFoundError:
        print("Input file not found. Run File 1 first.")
        return

    # Ensure reference_contexts is read as a list, not a string representation of a list
    df['reference_contexts'] = df['reference_contexts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    client = get_runtime_client()
    error_log = []
    
    print("Starting retrieval process...")
    retrieved_data = []
    retrieved_files_data = []
    
    for index, row in df.iterrows():
        query = row['user_input']
        print(f"[{index+1}/{len(df)}] Retrieving: {query[:30]}...")
        
        contexts, retrieved_files = retrieve_contexts(query, client, error_log)
        retrieved_data.append(contexts)
        retrieved_files_data.append(retrieved_files)

    df['retrieved_contexts'] = retrieved_data
    df['retrieved_file'] = retrieved_files_data
    
    ensure_parent_dir(config.PIPELINE_CSV)
    df.to_csv(config.PIPELINE_CSV, index=False)
    print(f"Retrieval complete. Updated {config.PIPELINE_CSV}")

    if error_log:
        summary_path = os.path.join(
            os.path.dirname(config.PIPELINE_CSV),
            "retriever_run_summary.json"
        )
        ensure_parent_dir(summary_path)
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump({
                "retrieved": len(df),
                "errors": error_log,
            }, summary_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
