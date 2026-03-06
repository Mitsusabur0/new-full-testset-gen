import os
import pandas as pd
import ast
import json
import random
import time
import boto3
from botocore.exceptions import ClientError

import config


output_file = "full_reranker_run"

REQUIRED_COLUMNS = [
    "reference_contexts",
    "retrieved_contexts",
    "retrieved_file",
    "source_file",
]


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_results_paths(output_dir: str, output_name: str = "") -> tuple[str, str, str, str]:
    folder_name = os.path.basename(os.path.normpath(output_dir))
    base_name = output_name.strip() if isinstance(output_name, str) else ""
    if not base_name:
        base_name = folder_name

    csv_path = os.path.join(output_dir, f"{base_name}_results.csv")
    parquet_path = os.path.join(output_dir, f"{base_name}_results.parquet")
    streamlit_output_dir = os.path.join("streamlit", "complete_datasets", base_name)
    streamlit_parquet_path = os.path.join(
        streamlit_output_dir,
        f"{base_name}_results.parquet",
    )
    streamlit_summary_path = os.path.join(
        streamlit_output_dir,
        "run_summary.md",
    )
    return csv_path, parquet_path, streamlit_parquet_path, streamlit_summary_path


def get_bedrock_client():
    session = boto3.Session(profile_name=config.AWS_PROFILE_LLM)
    return session.client(service_name="bedrock-runtime", region_name=config.AWS_REGION)


def backoff_sleep(attempt):
    base = config.BACKOFF_BASE_SECONDS * (2 ** attempt)
    sleep_for = min(base, config.BACKOFF_MAX_SECONDS)
    sleep_for += random.uniform(0, config.BACKOFF_JITTER_SECONDS)
    time.sleep(sleep_for)


def call_with_retry(fn, operation_name, error_log, max_retries=None):
    retries = config.MAX_RETRIES if max_retries is None else max_retries
    last_error = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except ClientError as e:
            last_error = e
        except Exception as e:
            last_error = e

        if attempt < retries:
            backoff_sleep(attempt)
        else:
            if last_error is not None:
                error_log.append({
                    "operation": operation_name,
                    "error": str(last_error),
                })
            return None


def extract_run_summary(client, metrics, output_name, error_log):
    system_prompt = """
Eres un analista de evaluación de sistemas RAG.
Responde SOLO en ESPAÑOL y SOLO en formato markdown.
"""
    user_prompt = f"""
Resumen una corrida de evaluación con los siguientes resultados agregados:
- Nombre de corrida: {output_name}
- Total de casos evaluados: {metrics["total_rows"]}
- mrr promedio: {metrics["avg_mrr"]}
- hit rate promedio: {metrics["avg_hit_rate"]}
- precision@k promedio: {metrics["avg_precision_at_k"]}
- cobertura@k promedio: {metrics["avg_cobertura_at_k"]}

Genera un resumen técnico en markdown que incluya:
1) Un resumen ejecutivo de los resultados.
2) Explicación de cómo se calcula cada métrica en nuestro pipeline.
3) Qué significa en la práctica cada número para el rendimiento del sistema RAG.

Notas del cálculo exacto:
- hit rate: se asigna 1 si el source_file coincide con alguno de los documentos recuperados por ruta o si el contexto de referencia coincide parcialmente con un texto recuperado; en caso contrario 0.
- mrr: si hay hit, es 1/rank del primer match, si no hay hit es 0.
- precision@k: para cada consulta, es 1/{config.TOP_K} si hay hit, 0 si no hay hit.
- cobertura@k: si hay scores de relevancia, es cantidad de scores >= 0.5 dividido por k. Si el hit existe pero esta métrica queda en 0 por falta de scores válidos, se asigna 1/3 como valor de respaldo.

Usa un tono profesional y menciona posibles riesgos o mejoras sugeridas de forma concisa.
"""

    body = json.dumps({
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    })

    def _call():
        return client.invoke_model(
            modelId=config.MODEL_ID,
            body=body
        )

    response = call_with_retry(_call, "invoke_model_run_summary", error_log)
    if response is None:
        return (
            "## Resumen de evaluación\n"
            f"- mrr promedio: {metrics['avg_mrr']}\n"
            f"- hit rate promedio: {metrics['avg_hit_rate']}\n"
            f"- precision@k promedio: {metrics['avg_precision_at_k']}\n"
            f"- cobertura@k promedio: {metrics['avg_cobertura_at_k']}\n"
            "\nNo se pudo generar el resumen automático con el LLM."
        )

    response_body = json.loads(response.get("body").read().decode("utf-8"))
    if "choices" in response_body:
        return response_body["choices"][0]["message"]["content"].strip()
    if "output" in response_body:
        return response_body["output"]["message"]["content"].strip()
    return str(response_body)


def save_markdown_summary(path, text):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def contains_source_file(source_file, retrieved_files):
    if not source_file or not retrieved_files:
        return False, 0
    source_norm = str(source_file).strip()
    for i, uri in enumerate(retrieved_files):
        if source_norm and source_norm in str(uri):
            return True, i + 1
    return False, 0


def parse_list_cell(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def calculate_metrics(row):
    gt_list = row['reference_contexts']
    retrieved_list = row['retrieved_contexts']
    retrieved_files = row['retrieved_file']
    relevance_scores = row.get('relevance_scores', float('nan'))

    gt_text = gt_list[0] if gt_list else ""

    hit = False
    rank = 0

    source_file = row['source_file']
    if source_file and retrieved_files:
        hit, rank = contains_source_file(source_file, retrieved_files)

    if not hit:
        for i, ret_text in enumerate(retrieved_list):
            clean_gt = " ".join(gt_text.lower().split())
            clean_ret = " ".join(ret_text.lower().split())

            if clean_gt in clean_ret or clean_ret in clean_gt:
                hit = True
                rank = i + 1
                break

    hit_rate = 1 if hit else 0
    mrr = 1.0 / rank if hit else 0.0

    precision_k = max(config.TOP_K, 1)
    precision = (1 / precision_k) if hit else 0
    recall = 1 if hit else 0

    precision_at_k_relevance = float('nan')
    if isinstance(relevance_scores, list):
        k = len(relevance_scores)
        if k > 0 and k == len(retrieved_list):
            try:
                hits = sum(1 for score in relevance_scores if float(score) >= 0.5)
                precision_at_k_relevance = hits / k
            except (TypeError, ValueError):
                precision_at_k_relevance = float('nan')

    if hit_rate == 1 and precision_at_k_relevance == 0:
        precision_at_k_relevance = 1/3

    return pd.Series([hit_rate, mrr, precision, recall, precision_at_k_relevance])


def main():
    error_log = []
    print(f"Loading {config.PIPELINE_CSV}...")
    try:
        df = pd.read_csv(config.PIPELINE_CSV)
    except FileNotFoundError:
        print("Input file not found. Run File 2 first.")
        return

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}. Run Files 1 and 2 first.")
        return

    df['reference_contexts'] = df['reference_contexts'].apply(parse_list_cell)
    df['retrieved_contexts'] = df['retrieved_contexts'].apply(parse_list_cell)
    df['retrieved_file'] = df['retrieved_file'].apply(parse_list_cell)
    if 'relevance_scores' in df.columns:
        df['relevance_scores'] = df['relevance_scores'].apply(parse_list_cell)

    print("Calculating metrics...")

    metrics_df = df.apply(calculate_metrics, axis=1)
    metrics_df.columns = [
        'custom_hit_rate',
        'custom_mrr',
        'custom_precision_at_k',
        'custom_recall_at_k',
        'precision_at_k_relevance',
    ]

    final_df = pd.concat([df, metrics_df], axis=1)

    avg_hit_rate = float(final_df['custom_hit_rate'].mean())
    avg_mrr = float(final_df['custom_mrr'].mean())
    avg_precision_at_k = float(final_df['custom_precision_at_k'].mean())
    avg_cobertura_at_k = float(final_df['precision_at_k_relevance'].mean())

    summary_metrics = {
        "total_rows": int(len(final_df)),
        "avg_mrr": avg_mrr,
        "avg_hit_rate": avg_hit_rate,
        "avg_precision_at_k": avg_precision_at_k,
        "avg_cobertura_at_k": avg_cobertura_at_k,
    }

    client = get_bedrock_client()
    summary_md = extract_run_summary(client, summary_metrics, output_file, error_log)

    results_csv, results_parquet, streamlit_parquet, streamlit_summary = build_results_paths(
        config.PIPELINE_OUTPUT_DIR,
        output_file,
    )

    ensure_parent_dir(results_csv)
    final_df.to_csv(results_csv, index=False)

    ensure_parent_dir(results_parquet)
    final_df.to_parquet(results_parquet, index=False)

    ensure_parent_dir(streamlit_parquet)
    final_df.to_parquet(streamlit_parquet, index=False)

    save_markdown_summary(streamlit_summary, summary_md)

    print(
        "Evaluation complete. Results saved to "
        f"{results_csv}, {results_parquet}, {streamlit_parquet}, and {streamlit_summary}"
    )


if __name__ == "__main__":
    main()
