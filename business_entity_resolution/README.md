# Business Entity Resolution — ML Challenge 2026

An end-to-end, high-performance, modular Machine Learning system for large-scale **Business Entity Resolution (ER)** across multi-source datasets from the **US**, **India**, and **France**.

---

## 📌 Project Overview

In commercial platforms, business identity records originate from multiple independent data sources with noisy names, inconsistent address formats, missing PIN/postal codes, and distinct jurisdictional conventions. 

Given:
- **Source 1 ($S_1$):** Deduplicated reference entity pool.
- **Source 2 ($S_2$) & Source 3 ($S_3$):** Noisy candidate target pools.

This solution:
1. **Normalizes** multilingual entity strings and expands localized abbreviations.
2. **Generates compact candidate sets** via multi-tier inverted indexing and character 3-gram TF-IDF retrieval.
3. **Extracts 24 dense features** quantifying string similarity, token overlap, and address component agreement.
4. **Scores matches using LightGBM GBDT** with hard negative mining.
5. **Calibrates decision thresholding** directly optimizing the competition **Macro $F_{0.5}$** metric.
6. **Produces submission-compliant TSVs** (`matching_results.tsv` and `candidate_pairs.tsv`) ensuring $\text{matches} \subseteq \text{candidates}$.

---

## 📁 Repository Structure

```
business_entity_resolution/
│
├── src/
│   ├── __init__.py               # Package initializer
│   ├── config.py                 # Centralized hyperparameters, file paths, constants
│   ├── load_data.py              # TSV file ingestion and ground truth parsing
│   ├── normalization.py          # NFKD Unicode cleaning, legal suffix removal, address expansion
│   ├── preprocessing.py          # Tokenization, 6-digit Indian PIN & 5-digit US/FR postal parsing
│   ├── blocking.py               # Multi-key inverted indexing & character TF-IDF retrieval
│   ├── similarity.py             # RapidFuzz, Jaccard, Jaro-Winkler string similarity metrics
│   ├── features.py               # Dense 24-dimensional continuous feature matrix extraction
│   ├── train.py                  # LightGBM GBDT training with hard negative pair construction
│   ├── evaluate.py               # Macro F0.5 evaluation, Precision, Recall & threshold search
│   ├── predict.py                # Batch scoring on candidate target pairs
│   ├── postprocess.py            # F0.5 thresholding, singleton classification & subset validation
│   ├── submission.py             # Streaming format-compliant TSV export
│   └── run_pipeline.py           # Unified CLI orchestrator
│
├── models/
│   └── business_matcher.pkl      # Trained LightGBM model artifact & optimal threshold
│
├── README.md                     # Comprehensive reproduction and execution guide
└── requirements.txt              # Minimal pinned dependencies
```

---

## 🛠️ Environment & Installation

### Requirements
- **Python:** 3.8+ (Tested on Python 3.11)
- **Model License Compliance:** MIT / Apache 2.0 (LightGBM, Scikit-learn, RapidFuzz)
- **Model Parameter Limit:** $\le$ 8 Billion parameters (GBDT tree ensemble)
- **External Data Policy:** Zero external database lookups or external APIs used.

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

Place your dataset files in the `dataset/` directory:
```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

---

## 🚀 Execution Guide

### 1. Unified End-to-End Pipeline (Recommended)
Train the model, optimize the Macro $F_{0.5}$ threshold, perform streaming test inference, and export final submission TSVs in a single command:

```bash
# Full dataset execution
python -m src.run_pipeline --mode full

# Fast test run with 10,000 sampled records
python -m src.run_pipeline --mode full --sample-size 10000
```

### 2. Step-by-Step Execution

#### Step A: Model Training & Validation Threshold Tuning
```bash
python -m src.train --sample-size 50000
```
- Trains LightGBM classifier with early stopping.
- Scans decision thresholds $\theta \in [0.40, 0.95]$ to maximize Macro $F_{0.5}$.
- Serializes model artifact to `models/business_matcher.pkl`.

#### Step B: Prediction & Scoring on Test Set
```bash
python -m src.predict --sample-size 1000
```

#### Step C: Export & Verify Submission Files
```bash
python -m src.submission
```

---

## 🧪 Testing & Submission Validation

### 1. Run Pre-Submission Validator
Before creating the ZIP archive or uploading to the portal, validate output formatting against all competition rules:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

The script verifies:
- `matching_results.tsv` and `candidate_pairs.tsv` exist and are tab-separated (.tsv).
- Exactly 1 row per test Source 1 entity.
- Only valid `S2-` and `S3-` entity IDs are referenced.
- No duplicate IDs within any ID list.
- Strict candidate subset constraint: $\text{matching\_results} \subseteq \text{candidate\_pairs}$.

### 2. Create Submission Archive
```bash
python package_submission.py
```
This generates `TEAM_NAME_submission.zip` with the exact required root layout:
```
TEAM_NAME_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

## 📊 Technical Highlights & Metric Optimization

- **Multi-Key Inverted Indexing:** Cuts the $\mathcal{O}(N \times M)$ pairwise comparison space to an average of $\sim 10$ candidates per $S_1$ entity while preserving candidate recall.
- **Asymmetric $F_{0.5}$ Tuning:** Because Macro $F_{0.5}$ weights precision $2\times$ over recall, decision thresholds ($\theta^* \approx 0.65 - 0.75$) penalize false merges and preserve singleton accuracy ($>95\%$).
- **Open-Set Generalization:** Handles unseen geographical jurisdictions (e.g., `France` in test data) without hardcoded filtering or schema mismatch.
