import ast

import pandas as pd
import config

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
REQUIRED_COLUMNS = ["user_input", "retrieved_contexts"]
SCORE_BATCH_SIZE = 64
TARGET_PROGRESS_UPDATES = 20


def progress_iter(iterable, total, desc, unit):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc, unit=unit)
    except Exception:
        print(f"{desc}...")
        return iterable


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


def is_empty_text(value):
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def clamp_score(score):
    numeric = float(score)
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def get_reranker():
    try:
        import torch  # type: ignore

        use_fp16 = torch.cuda.is_available()
    except Exception:
        use_fp16 = False

    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        details = str(exc)
        if "is_torch_fx_available" in details:
            print(
                "Incompatible package versions: FlagEmbedding currently expects "
                "transformers 4.x, but transformers 5.x is installed.\n"
                "Fix by running: `pip install \"transformers<5\"` "
                "(or a specific 4.x version) in this virtual environment."
            )
            return None
        print(
            "Missing dependency: FlagEmbedding. Install it with "
            "`pip install FlagEmbedding` and run again."
        )
        return None

    return FlagReranker(MODEL_NAME, use_fp16=use_fp16)


def compute_relevance_scores(df, reranker):
    all_pairs = []
    row_chunk_counts = []

    row_iter = progress_iter(
        df.iterrows(),
        total=len(df),
        desc="Preparing reranker inputs",
        unit="row",
    )
    for _, row in row_iter:
        query = "" if is_empty_text(row["user_input"]) else str(row["user_input"])
        chunks = [str(chunk) for chunk in parse_list_cell(row["retrieved_contexts"])]
        row_chunk_counts.append(len(chunks))

        for chunk in chunks:
            all_pairs.append([query, chunk])

    if not all_pairs:
        return [[] for _ in range(len(df))]

    # Keep normal throughput on large runs while ensuring visible progress on small runs.
    dynamic_batch_size = min(
        SCORE_BATCH_SIZE,
        max(1, len(all_pairs) // TARGET_PROGRESS_UPDATES),
    )
    total_batches = (len(all_pairs) + dynamic_batch_size - 1) // dynamic_batch_size
    print(
        f"Scoring {len(all_pairs)} query-context pairs in {total_batches} batches "
        f"(batch_size={dynamic_batch_size})..."
    )
    print("Note: the first batch may be slower due to model warm-up.")

    all_scores = []
    batch_iter = progress_iter(
        range(0, len(all_pairs), dynamic_batch_size),
        total=total_batches,
        desc="Scoring pairs",
        unit="batch",
    )
    for start in batch_iter:
        end = start + dynamic_batch_size
        batch_pairs = all_pairs[start:end]
        batch_scores = reranker.compute_score(batch_pairs, normalize=True)
        if isinstance(batch_scores, (float, int)):
            batch_scores = [batch_scores]
        all_scores.extend(clamp_score(score) for score in batch_scores)

    row_scores = []
    cursor = 0
    for chunk_count in row_chunk_counts:
        row_values = all_scores[cursor:cursor + chunk_count]
        row_scores.append(row_values)
        cursor += chunk_count

    return row_scores


def main():
    print(f"Loading {config.PIPELINE_CSV}...")
    try:
        df = pd.read_csv(config.PIPELINE_CSV)
    except FileNotFoundError:
        print("Input file not found. Run File 2 first.")
        return

    missing_cols = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}. Run File 2 first.")
        return

    reranker = get_reranker()
    if reranker is None:
        return

    print("Computing normalized relevance scores...")
    relevance_scores = compute_relevance_scores(df, reranker)

    if "relevance_scores" in df.columns:
        df = df.drop(columns=["relevance_scores"])

    insert_at = df.columns.get_loc("retrieved_contexts") + 1
    df.insert(insert_at, "relevance_scores", relevance_scores)

    df.to_csv(config.PIPELINE_CSV, index=False)
    print(f"Relevance scoring complete. Updated {config.PIPELINE_CSV}")


if __name__ == "__main__":
    main()
