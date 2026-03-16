# LLM as Ontology Server: SNOMED CT Auditing Pipeline

This repository provides a comprehensive three-step pipeline for evaluating Large Language Models (LLMs) on their ability to act as ontology servers, specifically using the **SNOMED CT** terminology.

The pipeline extracts ground truth from the BioPortal API, queries various LLMs with comparative prompt sets, and performs detailed accuracy auditing with term-level diagnostics.

---

## 🚀 Pipeline Overview

### 1. **Step 1: Ground Truth Extraction**
**Location:** `ground_truth/step1_ground_truth.ipynb`  
Validates a list of medical concepts against the BioPortal SNOMED CT API. It ensures every concept exists in the standard, extracts their official hierarchy (parents, grandparents, children, siblings), and saves a shared ground-truth dataset.

### 2. **Step 2: LLM Knowledge Probing**
**Location:** `testing_[model]/step2_llm_queries.ipynb`  
Queries the targeted LLM using two distinct prompt styles:
- **Set 1 (Ontology-style):** Technical prompts using terms like "FSN", "Semantic Tag", and "Definition Status".
- **Set 2 (Natural Language):** Human-like prompts using terms like "Official Name", "Broader categories", and "Peer terms".

### 3. **Step 3: Accuracy Auditing & Analytics**
**Location:** `scripts/step3_accuracy_with_maintained_added_removed.py`  
Executes a deep diagnostic comparison between LLM predictions and BioPortal ground truth. Beyond simple accuracy scores, it breaks down every prediction into:
- **Maintained**: Correct terms identified by the LLM.
- **Added**: Hallucinations or extra terms not in the gold standard.
- **Removed**: Valid SNOMED CT terms that the LLM missed.

---

## 📁 Project Structure

```text
repo/
├── ground_truth/            # Ground Truth harvesting (Step 1)
├── testing_gemini/          # Gemini-specific probes (Step 2)
├── testing_gpt/             # GPT-specific probes
├── testing_claude/          # Claude-specific probes
├── testing_deepseek/        # DeepSeek-specific probes
├── output/                  # Generated Data (Git Ignored)
│   ├── ground_truth/        # BioPortal gold standard CSVs
│   └── [model]/[run]/       # Per-run LLM outputs and analytics
├── scripts/                 # Core automation and auditing logic
│   ├── build_*.py           # Generation scripts for notebooks
│   └── step3_accuracy_...   # Enhanced analytical comparison script
└── README.md                # This file
```

---

## 📊 Analytical Columns (Step 3)

The enhanced auditing process adds **24 analytical columns** to the comparison CSV for each relationship type (parents, grandparents, children, siblings) and both prompt sets:

| Column Suffix | Description | Use Case |
| :--- | :--- | :--- |
| `_maintained` | Intersection of GT and LLM | Measures **correct knowledge retention**. |
| `_added` | LLM terms NOT in GT | Measures **hallucination** or extra context. |
| `_removed` | GT terms NOT in LLM | Measures **knowledge omission**. |
| `_exact` | % of GT terms found | Standard **Recall** metric. |
| `_jaccard` | Intersection over Union | Measures **Similarity** (penalizes hallucinations). |

---

## 🛠 Setup & Configuration

### 1. Environment Variables
You must set the following keys in your environment (or `.env` file):

```bash
# Required for Ground Truth extraction
export BIOPORTAL_API_KEY="your_api_key"

# Required for LLM testing
export GOOGLE_API_KEY="AIzaSy..."   # For Gemini
export OPENAI_API_KEY="sk-..."       # For GPT
export ANTHROPIC_API_KEY="sk-ant-..." # For Claude
```

### 2. Dependencies
Install the required Python packages:
```bash
pip install pandas requests google-generativeai openai anthropic
```

---

## 📈 Analyzing Results in Google Sheets

To perform deep analysis of the results:
1. Locate the enhanced output: `output/gemini/run_002/step3_accuracy/comparison_set1_set2_vs_ground_truth.csv`.
2. Open [Google Sheets](https://sheets.new).
3. Go to **File > Import > Upload** and upload the CSV.
4. **Tip:** Use conditional formatting on the `_maintained`, `_added`, and `_removed` columns to quickly visualize where the LLM is succeeding or failing.

---

## ⚙️ Building / Modifying Notebooks
If you need to change the logic across all testing folders, modify the Python templates in `scripts/` and run the build scripts:
- `python scripts/build_step2_llm_queries.py`
- `python scripts/build_step3_accuracy.py`
