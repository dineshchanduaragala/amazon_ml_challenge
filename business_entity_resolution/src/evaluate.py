"""
Evaluation module for Business Entity Resolution.
Computes Macro F_0.5, Macro Precision, Macro Recall, Candidate Recall ceiling,
and finds optimal decision threshold on validation set.
"""

from typing import Dict, List, Set, Tuple
import numpy as np


def compute_entity_f05(true_set: Set[str], pred_set: Set[str]) -> float:
    """
    Compute F_0.5 score for a single Source 1 entity.
    
    Singletons:
      - If true_set is empty: score 1.0 if pred_set is empty, else 0.0.
    Non-singletons:
      - If true_set is non-empty: score 0.0 if pred_set is empty.
      - Otherwise: (1.25 * P * R) / (0.25 * P + R).
    """
    if not true_set:
        return 1.0 if not pred_set else 0.0

    if not pred_set:
        return 0.0

    intersection = len(true_set & pred_set)
    if intersection == 0:
        return 0.0

    precision = intersection / len(pred_set)
    recall = intersection / len(true_set)
    denom = (0.25 * precision) + recall

    if denom <= 0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_predictions(
    gt_dict: Dict[str, Set[str]],
    pred_dict: Dict[str, Set[str]],
    all_s1_ids: List[str],
) -> Dict[str, float]:
    """
    Compute Macro F_0.5, Macro Precision, Macro Recall, and Singleton Accuracy
    by evaluating per Source 1 entity and taking the macro-average.
    """
    f05_scores = []
    precision_scores = []
    recall_scores = []
    singleton_scores = []

    for s1_id in all_s1_ids:
        true_set = gt_dict.get(s1_id, set())
        pred_set = pred_dict.get(s1_id, set())

        score = compute_entity_f05(true_set, pred_set)
        f05_scores.append(score)

        if not true_set:
            singleton_scores.append(1.0 if not pred_set else 0.0)
        else:
            if not pred_set:
                precision_scores.append(0.0)
                recall_scores.append(0.0)
            else:
                inter = len(true_set & pred_set)
                precision_scores.append(inter / len(pred_set))
                recall_scores.append(inter / len(true_set))

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_p = float(np.mean(precision_scores)) if precision_scores else 0.0
    macro_r = float(np.mean(recall_scores)) if recall_scores else 0.0
    singleton_acc = float(np.mean(singleton_scores)) if singleton_scores else 1.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "singleton_accuracy": singleton_acc,
        "total_evaluated": len(all_s1_ids),
    }


def compute_candidate_recall(
    gt_dict: Dict[str, Set[str]],
    candidate_dict: Dict[str, List[str]],
    all_s1_ids: List[str],
) -> float:
    """
    Calculate the candidate recall ceiling:
    Total ground truth matches recovered in candidate set / Total ground truth matches.
    """
    total_true_matches = 0
    total_recalled_matches = 0

    for s1_id in all_s1_ids:
        true_set = gt_dict.get(s1_id, set())
        if not true_set:
            continue
        cands_set = set(candidate_dict.get(s1_id, []))
        total_true_matches += len(true_set)
        total_recalled_matches += len(true_set & cands_set)

    if total_true_matches == 0:
        return 1.0

    return float(total_recalled_matches / total_true_matches)


def optimize_threshold(
    gt_dict: Dict[str, Set[str]],
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    all_s1_ids: List[str],
    threshold_range: np.ndarray = np.arange(0.40, 0.96, 0.05),
) -> Tuple[float, float, Dict[float, float]]:
    """
    Scan decision thresholds to find the threshold that maximizes Macro F_0.5.
    Returns: (best_threshold, best_f05, scores_by_threshold)
    """
    best_thresh = 0.65
    best_f05 = -1.0
    scores_by_thresh = {}

    print(f"[ThresholdOpt] Scanning {len(threshold_range)} threshold values for Macro F_0.5...")

    for thresh in threshold_range:
        thresh = round(float(thresh), 3)
        pred_dict: Dict[str, Set[str]] = {}
        for s1_id in all_s1_ids:
            scored_cands = candidate_scores.get(s1_id, [])
            matched = {cid for cid, score in scored_cands if score >= thresh}
            pred_dict[s1_id] = matched

        metrics = evaluate_predictions(gt_dict, pred_dict, all_s1_ids)
        score = metrics["macro_f05"]
        scores_by_thresh[thresh] = score

        if score > best_f05:
            best_f05 = score
            best_thresh = thresh

    print(f"[ThresholdOpt] Best Threshold: {best_thresh:.3f} -> Validation Macro F_0.5: {best_f05:.4f}")
    return best_thresh, best_f05, scores_by_thresh
