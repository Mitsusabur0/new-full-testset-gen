# evaluator.py
import os
import pandas as pd
import ast
import config


output_file = "mini_test_reranker_precision"

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

def build_results_paths(output_dir: str, output_name: str = "") -> tuple[str, str, str]:
    folder_name = os.path.basename(os.path.normpath(output_dir))
    base_name = output_name.strip() if isinstance(output_name, str) else ""
    if not base_name:
        base_name = folder_name

    csv_path = os.path.join(output_dir, f"{base_name}_results.csv")
    parquet_path = os.path.join(output_dir, f"{base_name}_results.parquet")
    streamlit_parquet_path = os.path.join(
        "streamlit",
        "complete_datasets",
        f"{base_name}_results.parquet",
    )
    return csv_path, parquet_path, streamlit_parquet_path

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
                hits = sum(1 for score in relevance_scores if float(score) >= 0.7)
                precision_at_k_relevance = hits / k
            except (TypeError, ValueError):
                precision_at_k_relevance = float('nan')

    return pd.Series([hit_rate, mrr, precision, recall, precision_at_k_relevance])

def main():
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

    results_csv, results_parquet, streamlit_parquet = build_results_paths(
        config.PIPELINE_OUTPUT_DIR,
        output_file,
    )

    ensure_parent_dir(results_csv)
    final_df.to_csv(results_csv, index=False)

    ensure_parent_dir(results_parquet)
    final_df.to_parquet(results_parquet, index=False)

    ensure_parent_dir(streamlit_parquet)
    final_df.to_parquet(streamlit_parquet, index=False)
    print(
        "Evaluation complete. Results saved to "
        f"{results_csv}, {results_parquet}, and {streamlit_parquet}"
    )

if __name__ == "__main__":
    main()
