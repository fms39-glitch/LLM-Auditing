#!/usr/bin/env python3
"""
Step 3 Accuracy: Enhanced comparison with Maintained / Added / Removed columns.

For each concept, for each label type (parents, grandparents, children, siblings),
and for both Set 1 and Set 2, this script computes:

  - Maintained by LLM  = GT ∩ LLM   (terms the LLM correctly kept)
  - Added by LLM       = LLM − GT   (terms the LLM introduced / hallucinated)
  - Removed by LLM     = GT − LLM   (terms the LLM missed / omitted)

Output overwrites the existing comparison CSV with the new columns added.
Only Gemini is considered.
"""

import re
from pathlib import Path

import pandas as pd

# ============================================================
# Paths
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "output"
GT_PATH = OUTPUT_ROOT / "ground_truth" / "ground_truth.csv"

# Gemini run_002
RUN_DIR = OUTPUT_ROOT / "gemini" / "run_002"
SET1_PATH = RUN_DIR / "step2_llm_set1" / "set1_llm_output.csv"
SET2_PATH = RUN_DIR / "step2_llm_set2" / "set2_llm_output.csv"
STEP3_DIR = RUN_DIR / "step3_accuracy"
STEP3_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = STEP3_DIR / "comparison_set1_set2_vs_ground_truth.csv"

# ============================================================
# Helpers
# ============================================================

def strip_semantic_tag(term: str) -> str:
    return re.sub(r"\s*\([^()]*\)\s*$", "", term or "").strip()


def normalize_term(term: str) -> str:
    term = (term or "").lower().strip()
    term = re.sub(r"\s+", " ", term)
    term = strip_semantic_tag(term)
    return term


def pipe_to_set(s: str) -> set:
    """Convert a pipe-delimited string into a normalized set of terms."""
    if not s:
        return set()
    s = str(s).strip()
    if not s or s.upper() == "UNKNOWN":
        return set()
    return {
        normalize_term(x)
        for x in s.split("|")
        if x.strip() and x.upper() != "UNKNOWN"
    }


def pipe_to_list_raw(s: str) -> list:
    """Convert a pipe-delimited string into a list of RAW (non-normalized) terms."""
    if not s:
        return []
    s = str(s).strip()
    if not s or s.upper() == "UNKNOWN":
        return []
    return [x.strip() for x in s.split("|") if x.strip() and x.upper() != "UNKNOWN"]


def set_to_pipe(term_set) -> str:
    """Convert a set/list of terms back to pipe-delimited string."""
    if not term_set:
        return ""
    return "|".join(sorted(term_set))


def compute_maintained_added_removed_raw(gt_pipe: str, llm_pipe: str):
    """
    Compute Maintained / Added / Removed using normalized comparison
    but returning the RAW (original-cased) terms.

    Returns (maintained_pipe, added_pipe, removed_pipe) as pipe-delimited strings.
    """
    gt_raw = pipe_to_list_raw(gt_pipe)
    llm_raw = pipe_to_list_raw(llm_pipe)

    # Build norm -> raw mappings (keep first occurrence)
    gt_norm_to_raw = {}
    for t in gt_raw:
        n = normalize_term(t)
        if n and n not in gt_norm_to_raw:
            gt_norm_to_raw[n] = t

    llm_norm_to_raw = {}
    for t in llm_raw:
        n = normalize_term(t)
        if n and n not in llm_norm_to_raw:
            llm_norm_to_raw[n] = t

    gt_norms = set(gt_norm_to_raw.keys())
    llm_norms = set(llm_norm_to_raw.keys())

    maintained_norms = gt_norms & llm_norms
    added_norms = llm_norms - gt_norms
    removed_norms = gt_norms - llm_norms

    # Map back to raw terms (prefer GT raw for maintained/removed, LLM raw for added)
    maintained_raw = sorted([gt_norm_to_raw[n] for n in maintained_norms])
    added_raw = sorted([llm_norm_to_raw[n] for n in added_norms])
    removed_raw = sorted([gt_norm_to_raw[n] for n in removed_norms])

    return "|".join(maintained_raw), "|".join(added_raw), "|".join(removed_raw)


def accuracy_exact_match(gt_set: set, pred_set: set) -> float:
    if len(gt_set) == 0:
        return 0.0
    return len(gt_set & pred_set) / len(gt_set)


def accuracy_jaccard(gt_set: set, pred_set: set) -> float:
    if len(gt_set) == 0:
        return 0.0
    union = gt_set | pred_set
    if len(union) == 0:
        return 0.0
    return len(gt_set & pred_set) / len(union)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Step 3: Accuracy with Maintained / Added / Removed columns")
    print("=" * 80)

    # Verify prerequisites
    assert GT_PATH.exists(), f"Ground truth not found: {GT_PATH}"
    assert SET1_PATH.exists(), f"Set 1 not found: {SET1_PATH}"
    assert SET2_PATH.exists(), f"Set 2 not found: {SET2_PATH}"

    # Load data
    gt_df = pd.read_csv(GT_PATH, dtype=str).fillna("")
    set1 = pd.read_csv(SET1_PATH, dtype=str).fillna("")
    set2 = pd.read_csv(SET2_PATH, dtype=str).fillna("")

    print(f"Ground truth: {len(gt_df)} concepts")
    print(f"Set 1 predictions: {len(set1)} rows")
    print(f"Set 2 predictions: {len(set2)} rows")

    gt_map = gt_df.set_index("concept_term").to_dict(orient="index")
    concepts = gt_df["concept_term"].tolist()

    set1_idx = set1.set_index("concept_term") if "concept_term" in set1.columns else pd.DataFrame()
    set2_idx = set2.set_index("concept_term") if "concept_term" in set2.columns else pd.DataFrame()

    # Label mappings: (gt_column, set1_column, set2_column, output_label)
    LABELS = [
        ("parents",      "A4_parents",          "B4_immediate_broader",    "parents"),
        ("grandparents", "A5_grandparents",      "B5_grandparents",         "grandparents"),
        ("children",     "A6_children",          "B6_immediate_narrower",   "children"),
        ("siblings",     "A7_siblings",          "B7_peer_terms",           "siblings"),
    ]

    rows = []

    for concept_term in concepts:
        gt_row = gt_map.get(concept_term, {})
        gt_id = str(gt_row.get("snomed_id", ""))
        gt_fsn = str(gt_row.get("fsn", ""))

        row = {
            "concept_name": concept_term,
            "gt_snomed_id": gt_id,
            "gt_fsn": gt_fsn,
        }

        # Ground truth pipe strings
        gt_pipes = {}
        for gt_col, _, _, label in LABELS:
            gt_pipes[label] = str(gt_row.get(gt_col, ""))
            row[f"gt_{label}"] = gt_pipes[label]

        # ---- Set 1 ----
        if concept_term in set1_idx.index:
            s1_row = set1_idx.loc[concept_term]
        else:
            s1_row = {}

        for gt_col, s1_col, _, label in LABELS:
            s1_pipe = str(s1_row.get(s1_col, "")) if isinstance(s1_row, pd.Series) else ""
            row[f"set1_{label}"] = s1_pipe

            # Compute maintained / added / removed
            maintained, added, removed = compute_maintained_added_removed_raw(
                gt_pipes[label], s1_pipe
            )
            row[f"set1_{label}_maintained"] = maintained
            row[f"set1_{label}_added"] = added
            row[f"set1_{label}_removed"] = removed

            # Accuracy metrics
            gt_set = pipe_to_set(gt_pipes[label])
            pred_set = pipe_to_set(s1_pipe)
            row[f"set1_{label}_exact"] = accuracy_exact_match(gt_set, pred_set)
            row[f"set1_{label}_jaccard"] = accuracy_jaccard(gt_set, pred_set)

        # Set 1 concept-level averages
        s1_exact_vals = [row[f"set1_{l}_exact"] for _, _, _, l in LABELS]
        s1_jacc_vals = [row[f"set1_{l}_jaccard"] for _, _, _, l in LABELS]
        row["set1_concept_exact"] = sum(s1_exact_vals) / 4.0
        row["set1_concept_jaccard"] = sum(s1_jacc_vals) / 4.0

        # ---- Set 2 ----
        if concept_term in set2_idx.index:
            s2_row = set2_idx.loc[concept_term]
        else:
            s2_row = {}

        for gt_col, _, s2_col, label in LABELS:
            s2_pipe = str(s2_row.get(s2_col, "")) if isinstance(s2_row, pd.Series) else ""
            row[f"set2_{label}"] = s2_pipe

            # Compute maintained / added / removed
            maintained, added, removed = compute_maintained_added_removed_raw(
                gt_pipes[label], s2_pipe
            )
            row[f"set2_{label}_maintained"] = maintained
            row[f"set2_{label}_added"] = added
            row[f"set2_{label}_removed"] = removed

            # Accuracy metrics
            gt_set = pipe_to_set(gt_pipes[label])
            pred_set = pipe_to_set(s2_pipe)
            row[f"set2_{label}_exact"] = accuracy_exact_match(gt_set, pred_set)
            row[f"set2_{label}_jaccard"] = accuracy_jaccard(gt_set, pred_set)

        # Set 2 concept-level averages
        s2_exact_vals = [row[f"set2_{l}_exact"] for _, _, _, l in LABELS]
        s2_jacc_vals = [row[f"set2_{l}_jaccard"] for _, _, _, l in LABELS]
        row["set2_concept_exact"] = sum(s2_exact_vals) / 4.0
        row["set2_concept_jaccard"] = sum(s2_jacc_vals) / 4.0

        rows.append(row)

    # Build output with desired column order
    col_order = ["concept_name", "gt_snomed_id", "gt_fsn"]
    for label_tuple in LABELS:
        label = label_tuple[3]
        col_order.append(f"gt_{label}")

    # Set 1 columns
    for label_tuple in LABELS:
        label = label_tuple[3]
        col_order.extend([
            f"set1_{label}",
            f"set1_{label}_maintained",
            f"set1_{label}_added",
            f"set1_{label}_removed",
        ])
    for label_tuple in LABELS:
        label = label_tuple[3]
        col_order.extend([
            f"set1_{label}_exact",
            f"set1_{label}_jaccard",
        ])
    col_order.extend(["set1_concept_exact", "set1_concept_jaccard"])

    # Set 2 columns
    for label_tuple in LABELS:
        label = label_tuple[3]
        col_order.extend([
            f"set2_{label}",
            f"set2_{label}_maintained",
            f"set2_{label}_added",
            f"set2_{label}_removed",
        ])
    for label_tuple in LABELS:
        label = label_tuple[3]
        col_order.extend([
            f"set2_{label}_exact",
            f"set2_{label}_jaccard",
        ])
    col_order.extend(["set2_concept_exact", "set2_concept_jaccard"])

    out_df = pd.DataFrame(rows, columns=col_order)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"\nWrote: {OUT_CSV}")
    print(f"Shape: {out_df.shape[0]} rows × {out_df.shape[1]} columns")
    print(f"\nNew columns added:")
    new_cols = [c for c in out_df.columns if "maintained" in c or "added" in c or "removed" in c]
    for c in new_cols:
        print(f"  - {c}")

    # Quick summary
    print(f"\n{'=' * 80}")
    print("ACCURACY SUMMARY (Exact-match recall, %)")
    print(f"{'=' * 80}")
    for label_tuple in LABELS:
        label = label_tuple[3]
        s1 = round(100 * out_df[f"set1_{label}_exact"].mean(), 2)
        s2 = round(100 * out_df[f"set2_{label}_exact"].mean(), 2)
        print(f"  {label:15s}  Set1={s1:6.2f}%   Set2={s2:6.2f}%")

    s1_avg = round(100 * out_df["set1_concept_exact"].mean(), 2)
    s2_avg = round(100 * out_df["set2_concept_exact"].mean(), 2)
    print(f"  {'overall':15s}  Set1={s1_avg:6.2f}%   Set2={s2_avg:6.2f}%")

    print(f"\n{'=' * 80}")
    print("ACCURACY SUMMARY (Jaccard, %)")
    print(f"{'=' * 80}")
    for label_tuple in LABELS:
        label = label_tuple[3]
        s1 = round(100 * out_df[f"set1_{label}_jaccard"].mean(), 2)
        s2 = round(100 * out_df[f"set2_{label}_jaccard"].mean(), 2)
        print(f"  {label:15s}  Set1={s1:6.2f}%   Set2={s2:6.2f}%")

    s1_avg = round(100 * out_df["set1_concept_jaccard"].mean(), 2)
    s2_avg = round(100 * out_df["set2_concept_jaccard"].mean(), 2)
    print(f"  {'overall':15s}  Set1={s1_avg:6.2f}%   Set2={s2_avg:6.2f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
