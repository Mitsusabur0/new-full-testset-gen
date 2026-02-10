# evaluator.py
import pandas as pd
import numpy as np
import ast
import config

def contains_source_file(source_file, retrieved_files):
    if not source_file or not retrieved_files:
        return False, 0
    source_norm = str(source_file).strip()
    for i, uri in enumerate(retrieved_files):
        if source_norm and source_norm in str(uri):
            return True, i + 1
    return False, 0

def calculate_metrics(row):
    # Load lists (handling string conversion from CSV)
    gt_list = row['reference_contexts']
    retrieved_list = row['retrieved_contexts']
    retrieved_files = row.get('retrieved_file', [])
    
    if isinstance(gt_list, str): gt_list = ast.literal_eval(gt_list)
    if isinstance(retrieved_list, str): retrieved_list = ast.literal_eval(retrieved_list)
    if isinstance(retrieved_files, str): retrieved_files = ast.literal_eval(retrieved_files)
    
    # We assume 1 ground truth chunk for this pipeline
    gt_text = gt_list[0] if gt_list else ""
    
    hit = False
    rank = 0

    source_file = row.get('source_file', "")
    if source_file and retrieved_files:
        hit, rank = contains_source_file(source_file, retrieved_files)
    
    # Check for containment
    # Logic: Is the ground truth substring roughly contained in the retrieved chunk?
    # Or is the retrieved chunk contained in the ground truth (if chunks are small)?
    if not hit:
        for i, ret_text in enumerate(retrieved_list):
            # Normalize for comparison
            clean_gt = " ".join(gt_text.lower().split())
            clean_ret = " ".join(ret_text.lower().split())
            
            if clean_gt in clean_ret or clean_ret in clean_gt:
                hit = True
                rank = i + 1
                break
    
    # Metrics
    hit_rate = 1 if hit else 0
    mrr = 1.0 / rank if hit else 0.0
    
    # Precision@K: (Relevant Items in Top K) / K
    precision = (1 / config.EVAL_K) if hit else 0
    
    # Recall@K: (Relevant Items in Top K) / Total Relevant Items
    # Total Relevant is 1 in this synthetic setup
    recall = 1 if hit else 0
    
    return pd.Series([hit_rate, mrr, precision, recall])

def main():
    print(f"Loading {config.OUTPUT_RAGAS_DEEP_EVALSET_CSV}...")
    try:
        df = pd.read_csv(config.OUTPUT_RAGAS_DEEP_EVALSET_CSV)
    except FileNotFoundError:
        print("Input file not found. Run File 2 first.")
        return

    print("Calculating metrics...")
    
    metrics_df = df.apply(calculate_metrics, axis=1)
    metrics_df.columns = [
        'custom_hit_rate',
        'custom_mrr',
        'custom_precision_at_k',
        'custom_recall_at_k',
    ]
    
    final_df = pd.concat([df, metrics_df], axis=1)
    
    # Parse lists back to actual python objects for Parquet saving
    # (Parquet handles lists natively, unlike CSV)
    final_df['reference_contexts'] = final_df['reference_contexts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    final_df['retrieved_contexts'] = final_df['retrieved_contexts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    if 'retrieved_file' in final_df.columns:
        final_df['retrieved_file'] = final_df['retrieved_file'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    # Persist the augmented eval set with custom metrics (new file)
    final_df.to_csv(config.OUTPUT_FULL_EVALSET_CSV, index=False)

    # Save a Parquet version for downstream apps (handles lists natively)
    final_df.to_parquet(config.OUTPUT_RESULTS_PARQUET, index=False)
    print(
        "Evaluation complete. Results saved to "
        f"{config.OUTPUT_FULL_EVALSET_CSV} and {config.OUTPUT_RESULTS_PARQUET}"
    )

if __name__ == "__main__":
    main()
