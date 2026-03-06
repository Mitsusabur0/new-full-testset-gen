import os
import pandas as pd
import ast
import json
import random
import re
import time
import boto3
from botocore.exceptions import ClientError

import config


output_file = "full_reranker_0.1"

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


def clean_reasoning(text: str) -> str:
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def extract_run_summary(client, metrics, output_name, error_log):
    system_prompt = """
Eres un analista de evaluación de sistemas RAG.
Responde SOLO en ESPAÑOL y SOLO en markdown, con formato estricto y consistente.
"""
    user_prompt = f"""
Resumen una corrida de evaluación con los siguientes resultados agregados:
- Nombre de corrida: {output_name}
- Total de casos evaluados: {metrics["total_rows"]}
- hit rate promedio: {metrics["avg_hit_rate"]}
- mrr promedio: {metrics["avg_mrr"]}
- precision@k promedio: {metrics["avg_precision_at_k"]}
- recall@k promedio: {metrics["avg_recall_at_k"]}

Genera un resumen técnico en markdown con EXACTAMENTE 1 sección y sin texto fuera de markdown:
## Interpretación de resultados

Reglas de salida:
- Solo esa sección: “## Interpretación de los resultados”.
- La sección debe ser una tabla markdown con EXACTAMENTE 4 filas, con columnas: `Métrica`, `Cálculo` y `Interpretación`.
- No agregues texto fuera de la tabla dentro de la sección.
- Mantén siempre el orden: hit rate, mrr, precision@k, recall@k.
- Para cada métrica:
  - Indica primero cómo se calcula (en 1 línea).
  - Luego interpreta qué significa el valor en rendimiento del RAG (1 línea).
- Usa texto breve pero con un poco más de contexto que antes (sin ser excesivamente extenso).
- Incluye explícitamente esta limitación en una línea: recall@k = hit rate, dado que los casos sintéticos usan un solo archivo fuente por consulta y solo se garantiza si ese archivo fue recuperado.
- Explica que usamos un modelo que asigna `relevance_score` a cada contexto recuperado en función de su similitud con el `user_input`, y que aplicamos el umbral 0.5 para calcular precision@k. 
  Esto permite medir la relevancia real de cada contexto recuperado dentro de los k documentos, en lugar de depender solo de coincidencias de ruta.
  Si solo se usara coincidencia de `source_file`, precision@k quedaría acotada a un máximo teórico de 1/k aunque los contextos recuperados fueran relevantes.

Notas de cálculo:
- hit rate: se asigna 1 si el source_file coincide con alguno de los documentos recuperados por ruta o si el contexto de referencia coincide parcialmente con un texto recuperado; en caso contrario 0.
- mrr: si hay hit, es 1/rank del primer match, si no hay hit es 0.
- precision@k: se calcula como ratio de contextos con `relevance_score` >= 0.5 entre k, usando el score asignado por el modelo entre cada contexto y el `user_input`. Si hay hit pero no hay scores válidos y la métrica queda en 0, usar 1/3 como valor de respaldo.
- recall@k: se reporta igual que hit rate en este pipeline.

Usa un tono profesional y breve.
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
            "## Interpretación de los resultados\n"
            "| Métrica | Cálculo | Interpretación |\n"
            "| --- | --- | --- |\n"
            f"| hit rate | 1 si el source_file coincide con alguno de los documentos recuperados; 0 si no. | La tasa de {metrics['avg_hit_rate']:.4f} significa que ese porcentaje de casos recuperó la fuente esperada y su consulta fue satisfecha por el documento correcto. |\n"
            f"| mrr | 1/rank del primer hit cuando existe; 0 si no hay hit. | Un valor de {metrics['avg_mrr']:.4f} indica qué tan pronto se encontró el primer contexto correcto: cuanto más cercano a 1, mejor, porque el acierto ocurrió más arriba en el ranking. |\n"
            f"| precision@k | Proporción de contextos en top-k con `relevance_score >= 0.5`, donde el score lo asigna un modelo comparando el contexto recuperado con el `user_input`; si hay hit sin scores válidos se usa respaldo 1/3. | Un valor de {metrics['avg_precision_at_k']:.4f} muestra qué fracción de los k documentos recuperados son realmente relevantes para la pregunta del usuario, y ayuda a detectar ruido en el ranking. |\n"
            f"| recall@k | Igual que hit rate por diseño del dataset sintético (un archivo fuente por consulta). | El valor de {metrics['avg_recall_at_k']:.4f} coincide con hit rate y no puede interpretarse como recall clásico sobre múltiples relevantes por consulta, debido a la limitación de generación sintética. |\n"
        )

    response_body = json.loads(response.get("body").read().decode("utf-8"))
    if "choices" in response_body:
        return clean_reasoning(response_body["choices"][0]["message"]["content"])
    if "output" in response_body:
        return clean_reasoning(response_body["output"]["message"]["content"])
    return clean_reasoning(str(response_body))


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
                hits = sum(1 for score in relevance_scores if float(score) >= 0.9)
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
    avg_precision_at_k = float(final_df['precision_at_k_relevance'].mean())
    avg_recall_at_k = float(final_df['custom_recall_at_k'].mean())

    summary_metrics = {
        "total_rows": int(len(final_df)),
        "avg_mrr": avg_mrr,
        "avg_hit_rate": avg_hit_rate,
        "avg_precision_at_k": avg_precision_at_k,
        "avg_recall_at_k": avg_recall_at_k,
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
