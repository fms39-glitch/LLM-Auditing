import pandas as pd
import numpy as np
import sys
import re

def count_pipe(s):
    if not s or str(s).strip() == '': return 0
    return len([x for x in str(s).split('|') if x.strip()])

def strip_tag(t):
    return re.sub(r'\s*\([^()]*\)\s*$', '', str(t) or '').strip().lower()

def main():
    try:
        csv_path = r'llm-as-ontology-server\output\gemini\run_002\step3_accuracy\comparison_set1_set2_vs_ground_truth.csv'
        df = pd.read_csv(csv_path, dtype=str).fillna('')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    results = []
    labels = ['parents', 'grandparents', 'children', 'siblings']
    for setname in ['set1', 'set2']:
        for label in labels:
            m_col = f'{setname}_{label}_maintained'
            a_col = f'{setname}_{label}_added'
            r_col = f'{setname}_{label}_removed'
            
            m_counts = df[m_col].apply(count_pipe)
            a_counts = df[a_col].apply(count_pipe)
            r_counts = df[r_col].apply(count_pipe)
            
            # Precision = M / (M + A)
            total_pred = m_counts + a_counts
            p_vals = (m_counts / total_pred).replace([np.inf, -np.inf], np.nan)
            
            # Recall = M / (M + R)
            total_gt = m_counts + r_counts
            r_vals = (m_counts / total_gt).replace([np.inf, -np.inf], np.nan)
            
            results.append({
                'Set': setname.upper(),
                'Label': label.capitalize(),
                'M': int(m_counts.sum()),
                'A': int(a_counts.sum()),
                'R': int(r_counts.sum()),
                'Prec': p_vals.mean(),
                'Rec': r_vals.mean()
            })

    res_df = pd.DataFrame(results)
    print("--- Aggregated Metrics ---")
    print(res_df.to_string(index=False))

    # Error Case Selection
    print("\n--- Diagnostic Case Studies ---")
    
    # 1. High Precision, Low Recall (Omission)
    omission_case = df.assign(r_sum=df['set1_parents_removed'].apply(count_pipe)).sort_values('r_sum', ascending=False).iloc[0]
    print(f"OMISSION_CASE | {omission_case['concept_name']}")
    print(f"  GT: {omission_case['gt_parents']}")
    print(f"  REMOVED: {omission_case['set1_parents_removed']}")
    print(f"  LLM: {omission_case['set1_parents']}")
    print("---")

    # 2. High Hallucination (Added)
    hallucination_case = df.assign(a_sum=df['set1_parents_added'].apply(count_pipe)).sort_values('a_sum', ascending=False).iloc[0]
    print(f"HALLUCINATION_CASE | {hallucination_case['concept_name']}")
    print(f"  ADDED: {hallucination_case['set1_parents_added']}")
    print(f"  GT: {hallucination_case['gt_parents']}")
    print("---")

    # 3. Semantic Similarity Check (terms that might be correct but marked as added/removed)
    # We'll look for synonyms manually in a high-hallucination row
    syn_case = df.iloc[10] # Grab a random one for a closer look
    print(f"SEMANTIC_CHECK | {syn_case['concept_name']}")
    print(f"  GT: {syn_case['gt_parents']}")
    print(f"  LLM: {syn_case['set1_parents']}")
    print(f"  ADDED: {syn_case['set1_parents_added']}")
    print(f"  REMOVED: {syn_case['set1_parents_removed']}")

if __name__ == "__main__":
    main()
