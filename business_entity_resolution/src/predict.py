"""
Prediction module for Business Entity Resolution.
Loads the serialized LightGBM model artifact and computes match probabilities
for candidate entity pairs.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.blocking import CandidateBlocker
from src.config import DEFAULT_DECISION_THRESHOLD, MODELS_DIR, MODEL_FILENAME, BATCH_SIZE_FEATURES, TEST_DIR
from src.features import build_feature_matrix
from src.load_data import load_test_data
from src.preprocessing import preprocess_dataframe


class EntityMatcherPredictor:
    """
    Inference engine that loads trained LightGBM model artifact and scores candidate pairs.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        override_threshold: Optional[float] = None,
    ):
        if model_path is None:
            model_path = MODELS_DIR / MODEL_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {model_path}. Please train model first.")

        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.threshold = (
            override_threshold
            if override_threshold is not None
            else artifact.get("optimal_threshold", DEFAULT_DECISION_THRESHOLD)
        )
        self.feature_names = artifact.get("feature_names", [])
        print(f"[Predictor] Loaded model from: {model_path} (Active Threshold: {self.threshold:.3f})")

    def score_candidates(
        self,
        candidate_dict: Dict[str, List[str]],
        df_s1: pd.DataFrame,
        df_targets: pd.DataFrame,
        batch_size: int = BATCH_SIZE_FEATURES,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Score all candidate pairs for Source 1 queries.
        Returns: mapping of source1_entity_id -> list of (target_entity_id, probability).
        """
        candidate_scores: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        active_s1_ids = [s1_id for s1_id, cands in candidate_dict.items() if cands]

        if not active_s1_ids:
            return candidate_scores

        for start_idx in range(0, len(active_s1_ids), 10000):
            end_idx = min(start_idx + 10000, len(active_s1_ids))
            batch_s1_keys = active_s1_ids[start_idx:end_idx]
            batch_cand_dict = {k: candidate_dict[k] for k in batch_s1_keys}

            X_batch, pair_ids = build_feature_matrix(
                batch_cand_dict,
                df_s1,
                df_targets,
                desc=f"Inference Batch {start_idx // 10000 + 1}",
            )

            if len(pair_ids) == 0:
                continue

            probs = self.model.predict_proba(X_batch)[:, 1]

            for (s1_id, target_id), prob in zip(pair_ids, probs):
                candidate_scores[s1_id].append((target_id, float(prob)))

        return candidate_scores


def main():
    parser = argparse.ArgumentParser(description="Run Entity Matcher Prediction on Test Set")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR, help="Path to test data directory")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of records to test")
    args = parser.parse_args()

    print(f"[Predict] Loading test data ({args.sample_size:,} records)...")
    s1, s2, s3 = load_test_data(args.test_dir, sample_size=args.sample_size)

    s1_prep = preprocess_dataframe(s1)
    s2_prep = preprocess_dataframe(s2)
    s3_prep = preprocess_dataframe(s3)
    targets = pd.concat([s2_prep, s3_prep], ignore_index=True)

    blocker = CandidateBlocker(enable_tfidf=True)
    blocker.fit(targets)
    candidates = blocker.block_candidates(s1_prep)

    predictor = EntityMatcherPredictor()
    scores = predictor.score_candidates(candidates, s1_prep, targets)
    print(f"[Predict] Successfully scored {len(scores):,} Source 1 entities.")


if __name__ == "__main__":
    main()
