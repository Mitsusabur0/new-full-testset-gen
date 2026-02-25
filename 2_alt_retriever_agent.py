import os
import json
import pandas as pd
import boto3
import random
import time
from datetime import datetime
from botocore.exceptions import ClientError, BotoCoreError
import config


AGENT_ID = os.getenv("AGENT_ID", "UKQEMRZQUS")
AGENT_ALIAS_ID = os.getenv("AGENT_ALIAS_ID", "JPSUY1DN1P")
ENABLE_TRACE = os.getenv("AGENT_ENABLE_TRACE", "false").strip().lower() == "true"


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
        except (ClientError, BotoCoreError) as e:
            last_error = e
        except Exception as e:
            last_error = e

        if attempt < config.MAX_RETRIES:
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


def clean_text(text):
    if not text:
        return ""
    return " ".join(str(text).split())


def extract_s3_uri(reference):
    location = reference.get("location", {})
    uri = location.get("s3Location", {}).get("uri", "")
    if uri:
        return uri
    metadata = reference.get("metadata", {})
    return metadata.get("x-amz-bedrock-kb-source-uri", "")


def build_session_id(index):
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"retriever-agent-{now}-{index + 1}"


def parse_retrieved_references(response):
    retrieved_texts = []
    retrieved_files = []
    seen = set()

    for event in response.get("completion", []):
        chunk = event.get("chunk", {})
        attribution = chunk.get("attribution", {})
        citations = attribution.get("citations", [])
        for citation in citations:
            for reference in citation.get("retrievedReferences", []):
                text = clean_text(reference.get("content", {}).get("text", ""))
                uri = extract_s3_uri(reference)
                key = (text, uri)
                if key in seen:
                    continue
                seen.add(key)
                retrieved_texts.append(text)
                retrieved_files.append(uri)

    return retrieved_texts, retrieved_files


def retrieve_contexts_from_agent(query, index, client, error_log):
    def _call():
        return client.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=build_session_id(index),
            inputText=query,
            enableTrace=ENABLE_TRACE,
        )

    response = call_with_retry(_call, "invoke_agent", error_log)
    if response is None:
        print(f"Retrieval Error for query '{query}': exhausted retries")
        return [], []

    return parse_retrieved_references(response)


def main():
    print(f"Loading {config.PIPELINE_CSV}...")
    try:
        df = pd.read_csv(config.PIPELINE_CSV)
    except FileNotFoundError:
        print("Input file not found. Run File 1 first.")
        return

    if "user_input" not in df.columns:
        print("Missing required column 'user_input'. Run File 1 first.")
        return

    client = get_runtime_client()
    error_log = []

    print("Starting retrieval process via agent...")
    retrieved_data = []
    retrieved_files_data = []

    for index, row in df.iterrows():
        query = "" if pd.isna(row["user_input"]) else str(row["user_input"])
        print(f"[{index + 1}/{len(df)}] Retrieving: {query[:30]}...")

        contexts, retrieved_files = retrieve_contexts_from_agent(query, index, client, error_log)
        retrieved_data.append(contexts)
        retrieved_files_data.append(retrieved_files)

    df["retrieved_contexts"] = retrieved_data
    df["retrieved_file"] = retrieved_files_data

    ensure_parent_dir(config.PIPELINE_CSV)
    df.to_csv(config.PIPELINE_CSV, index=False)
    print(f"Retrieval complete. Updated {config.PIPELINE_CSV}")

    if error_log:
        summary_path = os.path.join(
            os.path.dirname(config.PIPELINE_CSV),
            "retriever_run_summary.json",
        )
        ensure_parent_dir(summary_path)
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(
                {
                    "retrieved": len(df),
                    "errors": error_log,
                },
                summary_file,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
