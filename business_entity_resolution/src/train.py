"""
Training module for Business Entity Resolution matching model.
Constructs balanced labeled pairs (ground truth positives + hard negatives),
extracts feature vectors, trains LightGBM GBDT classifier, optimizes F0.5 threshold,
and serializes the model artifact.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Set, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.blocking import CandidateBlocker
from src.config import LGBM_PARAMS, MODELS_DIR, MODEL_FILENAME, RANDOM_STATE, TRAIN_DIR
from src.evaluate import evaluate_predictions, optimize_threshold, compute_candidate_recall
from src.features import FEATURE_NAMES, build_feature_matrix
from src.load_data import load_train_data, parse_ground_truth
from src.preprocessing import preprocess_dataframe


def prepare_training_pairs(
    candidate_dict: Dict[str, List[str]],
    gt_dict: Dict[str, Set[str]],
    train_s1_ids: List[str],
    neg_to_pos_ratio: int = 4,
    random_state: int = RANDOM_STATE,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], int]]:
    """
    Build labeled pairs consisting of:
    - Positive pairs: True matches from ground truth.
    - Hard negative pairs: False positive candidates retrieved by blocking.
    """
    random.seed(random_state)
    pair_candidates: Dict[str, List[str]] = defaultdict(list)
    pair_labels: Dict[Tuple[str, str], int] = {}

    for s1_id in train_s1_ids:
        true_matches = gt_dict.get(s1_id, set())
        blocking_cands = candidate_dict.get(s1_id, [])

        # 1. Positives
        for tid in true_matches:
            pair_candidates[s1_id].append(tid)
            pair_labels[(s1_id, tid)] = 1

        # 2. Hard Negatives
        hard_negatives = [c for c in blocking_cands if c not in true_matches]
        target_num_negs = max(len(true_matches) * neg_to_pos_ratio, 2)
        if len(hard_negatives) > target_num_negs:
            sampled_negs = random.sample(hard_negatives, target_num_negs)
        else:
            sampled_negs = hard_negatives

        for tid in sampled_negs:
            pair_candidates[s1_id].append(tid)
            pair_labels[(s1_id, tid)] = 0

    return dict(pair_candidates), pair_labels


def train_matching_model(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    val_size: float = 0.15,
    save_path: Optional[Path] = None,
) -> Tuple[lgb.LGBMClassifier, float, dict]:
    """
    End-to-end model training workflow:
    - S1 Train / Validation Split
    - Multi-key Candidate Blocking
    - Labeled Feature Extraction
    - LightGBM GBDT Training with Early Stopping
    - Decision Threshold Optimization targeting Macro F0.5
    - Model Artifact Serialization
    """
    print("=" * 60)
    print("[Train] Initiating Matching Model Training")
    print("=" * 60)

    # 1. Train / Validation Split
    all_s1_ids = df_s1["entity_id"].tolist()
    train_ids, val_ids = train_test_split(all_s1_ids, test_size=val_size, random_state=RANDOM_STATE)
    print(f"[Train] Entity split: {len(train_ids):,} Train, {len(val_ids):,} Validation")

    # Combine S2 and S3 target pool
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)

    # 2. Fit Blocker on Target Pool
    blocker = CandidateBlocker(enable_tfidf=True)
    blocker.fit(df_targets)

    # 3. Generate candidates for Train and Validation sets
    train_s1_df = df_s1[df_s1["entity_id"].isin(train_ids)]
    val_s1_df = df_s1[df_s1["entity_id"].isin(val_ids)]

    train_cands = blocker.block_candidates(train_s1_df)
    val_cands = blocker.block_candidates(val_s1_df)

    # Candidate Recall Ceiling Check
    val_cand_recall = compute_candidate_recall(gt_dict, val_cands, val_ids)
    print(f"[Validation] Candidate Recall Ceiling: {val_cand_recall * 100:.2f}%")

    # 4. Prepare Training Pairs & Features
    print("[Train] Extracting features for labeled training pairs...")
    train_pair_dict, train_labels = prepare_training_pairs(train_cands, gt_dict, train_ids)
    X_train, train_pair_ids = build_feature_matrix(train_pair_dict, df_s1, df_targets, desc="Train Features")
    y_train = np.array([train_labels[pid] for pid in train_pair_ids], dtype=np.int32)
    print(f"[Train] Feature matrix shape: {X_train.shape}, Labels distribution: {np.bincount(y_train)}")

    # 5. Prepare Validation Pairs & Features
    print("[Validation] Extracting validation features across all candidates...")
    X_val, val_pair_ids = build_feature_matrix(val_cands, df_s1, df_targets, desc="Val Features")
    y_val = np.array([1 if pid[1] in gt_dict.get(pid[0], set()) else 0 for pid in val_pair_ids], dtype=np.int32)
    print(f"[Validation] Validation pairs: {X_val.shape[0]:,}, True matches: {sum(y_val):,}")

    # 6. Fit LightGBM Classifier
    print("[Train] Fitting LightGBM GBDT...")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # 7. Print Feature Importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\n[Train] Feature Importances (Top 10):")
    for idx in sorted_idx[:10]:
        print(f"  - {FEATURE_NAMES[idx]:25s}: {importances[idx]:.1f}")

    # 8. Score Validation Candidates & Optimize F0.5 Threshold
    print("\n[Validation] Scoring validation candidate pool...")
    val_probs = model.predict_proba(X_val)[:, 1]

    val_cand_scores: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (s1_id, target_id), prob in zip(val_pair_ids, val_probs):
        val_cand_scores[s1_id].append((target_id, float(prob)))

    best_thresh, best_f05, _ = optimize_threshold(
        gt_dict=gt_dict,
        candidate_scores=val_cand_scores,
        all_s1_ids=val_ids,
        threshold_range=np.arange(0.40, 0.96, 0.05),
    )

    # Detailed validation summary
    val_preds: Dict[str, Set[str]] = {}
    for s1_id in val_ids:
        scored = val_cand_scores.get(s1_id, [])
        val_preds[s1_id] = {cid for cid, p in scored if p >= best_thresh}

    val_metrics = evaluate_predictions(gt_dict, val_preds, val_ids)
    print("\n" + "=" * 60)
    print(f"[Validation Summary] Optimal Macro F0.5 Threshold: {best_thresh:.3f}")
    print(f"  Macro F_0.5 Score   : {val_metrics['macro_f05']:.4f}")
    print(f"  Macro Precision     : {val_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall        : {val_metrics['macro_recall']:.4f}")
    print(f"  Singleton Accuracy  : {val_metrics['singleton_accuracy']:.4f}")
    print(f"  Candidate Recall    : {val_cand_recall:.4f}")
    print("=" * 60)

    # 9. Save Model Artifact
    if save_path is None:
        save_path = MODELS_DIR / MODEL_FILENAME

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "model": model,
        "optimal_threshold": best_thresh,
        "feature_names": FEATURE_NAMES,
        "val_metrics": val_metrics,
        "val_cand_recall": val_cand_recall,
    }
    joblib.dump(model_artifact, save_path)
    print(f"[Train] Model artifact saved to: {save_path}")

    return model, best_thresh, val_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Business Entity Resolution Matching Model")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR, help="Path to training data directory")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of records to sample for training")
    args = parser.parse_args()

    print(f"[Main] Loading and preprocessing training data ({args.sample_size:,} records)...")
    s1, s2, s3, gt = load_train_data(args.train_dir, sample_size=args.sample_size)
    gt_dict = parse_ground_truth(gt)

    s1_prep = preprocess_dataframe(s1)
    s2_prep = preprocess_dataframe(s2)
    s3_prep = preprocess_dataframe(s3)

    train_matching_model(s1_prep, s2_prep, s3_prep, gt_dict)


if __name__ == "__main__":
    main()
