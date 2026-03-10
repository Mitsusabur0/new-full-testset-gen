#!/usr/bin/env python3
"""Validate the integrity of full_run_512_results.csv."""

from __future__ import annotations

import argparse
import ast
import math
import sys
from pathlib import Path

import pandas as pd


DEFAULT_CSV_PATH = Path("outputs/full_run_512/full_run_512_results.csv")
EXPECTED_COLUMNS = [
    "user_input",
    "reference_contexts",
    "query_style",
    "source_file",
    "retrieved_contexts",
    "relevance_scores",
    "retrieved_file",
    "custom_hit_rate",
    "custom_mrr",
    "custom_precision_at_k",
    "custom_recall_at_k",
    "precision_at_k_relevance",
]


def parse_list(value):
    if pd.isna(value):
        return [], None, "empty"
    if isinstance(value, list):
        return value, None, "list"

    if not isinstance(value, str):
        return value, "not_string", "not_string"

    text = value.strip()
    if not text:
        return [], "empty_string", "empty"

    try:
        parsed = ast.literal_eval(text)
    except Exception as exc:
        return value, f"parse_error:{type(exc).__name__}", "parse_error"

    if isinstance(parsed, list):
        return parsed, None, "list"
    return parsed, "not_list", "not_list"


def is_number(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return True
    return False


def check_row(row_num, score_cell, ctx_cell):
    issues = []
    scores, score_err, score_kind = parse_list(score_cell)
    contexts, ctx_err, ctx_kind = parse_list(ctx_cell)

    if score_err:
        issues.append(
            f"relevance_scores[{score_err}]: {score_cell!r}"
        )
    if ctx_err:
        issues.append(
            f"retrieved_contexts[{ctx_err}]: {ctx_cell!r}"
        )

    if score_err is None:
        if score_kind == "list":
            if score_err is None and any(
                not is_number(s) or s < 0 or s > 1 for s in scores
            ):
                bad = [
                    s for s in scores if not is_number(s) or s < 0 or s > 1
                ]
                issues.append(
                    f"relevance_scores out-of-range/non-numeric values {bad!r}"
                )

        if ctx_kind == "list":
            if isinstance(contexts, list) and isinstance(scores, list):
                if scores and not contexts:
                    issues.append(
                        "relevance_scores has values but retrieved_contexts is empty"
                    )
                if contexts and not scores:
                    issues.append("retrieved_contexts has values but relevance_scores is empty")
                if isinstance(contexts, list) and len(contexts) != len(scores):
                    issues.append(
                        "relevance_scores size != retrieved_contexts size "
                        f"({len(scores)} != {len(contexts)})"
                    )
        else:
            if isinstance(scores, list) and scores:
                issues.append("retrieved_contexts could not be parsed as list")

    return row_num, issues


def validate_columns(columns):
    issues = []
    seen = set()
    duplicates = [c for c in columns if c in seen or seen.add(c)]
    if duplicates:
        issues.append(f"duplicate column names: {duplicates}")

    stripped = [c.strip() if isinstance(c, str) else c for c in columns]
    if stripped != columns:
        issues.append("some column names contain leading/trailing whitespace")

    if "relevance_scores" not in columns:
        issues.append("missing required column: relevance_scores")
        return issues

    rel_idx = columns.index("relevance_scores")
    if "retrieved_contexts" in columns and rel_idx < columns.index("retrieved_contexts"):
        issues.append(
            "relevance_scores appears before retrieved_contexts (unexpected order)"
        )
    if "retrieved_file" in columns and rel_idx > columns.index("retrieved_file"):
        issues.append(
            "relevance_scores appears after retrieved_file (unexpected order)"
        )

    for expected in ("relevance_scores", "retrieved_contexts", "retrieved_file"):
        if expected not in columns:
            issues.append(f"missing expected column: {expected}")

    return issues


def run_checks(path: Path):
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"ERROR: failed to read CSV {path}: {exc}")
        return 1

    issues = validate_columns(list(df.columns))
    row_issues = []

    rs_col = df.get("relevance_scores")
    rc_col = df.get("retrieved_contexts")
    if rs_col is None or rc_col is None:
        missing = []
        if rs_col is None:
            missing.append("relevance_scores")
        if rc_col is None:
            missing.append("retrieved_contexts")
        print(f"ERROR: missing columns {missing}")
        return 1

    for idx, (score_cell, ctx_cell) in enumerate(zip(rs_col, rc_col), start=2):
        row_num, row_errs = check_row(idx, score_cell, ctx_cell)
        if row_errs:
            row_issues.append((row_num, row_errs))

    if row_issues:
        for row_num, errs in row_issues[:25]:
            for issue in errs:
                issues.append(f"row {row_num}: {issue}")

    print(f"Rows: {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    if row_issues:
        print(f"FAILED: {len(row_issues)} rows with relevance_scores/retrieved_contexts problems")
        print(f"Showing up to 25 of {len(row_issues)} affected rows:")
        for msg in issues:
            print(f"- {msg}")
        return 2

    if issues:
        print("FAILED: potential structure issues found:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("PASS: file looks structurally consistent.")
    return 0


def build_argparser():
    parser = argparse.ArgumentParser(description="Validate full_run_512_results.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to CSV file (default: outputs/full_run_512/full_run_512_results.csv)",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    return_code = run_checks(args.csv)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
