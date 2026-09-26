"""
Unified Pipeline Orchestrator for Business Entity Resolution.
Supports:
  1. Training on train_source1/2/3 and train_ground_truth
  2. Macro F0.5 Threshold Optimization
  3. Streaming Chunked Inference on test_source1/2/3
  4. Postprocessing and Singleton Handling
  5. Format-Compliant TSV Export (matching_results.tsv & candidate_pairs.tsv)
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

# Add project directory to sys.path to enable direct script execution
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.blocking import CandidateBlocker
from src.config import (
    CANDIDATE_PAIRS_FILENAME,
    DEFAULT_DECISION_THRESHOLD,
    INFERENCE_CHUNK_SIZE,
    MATCHING_RESULTS_FILENAME,
    MODELS_DIR,
    MODEL_FILENAME,
    OUTPUT_DIR,
    TEST_DIR,
    TRAIN_DIR,
)
from src.load_data import (
    load_test_data,
    load_train_data,
    load_tsv,
    parse_ground_truth,
)
from src.postprocess import enforce_candidate_subset, postprocess_matches
from src.predict import EntityMatcherPredictor
from src.preprocessing import preprocess_dataframe
from src.submission import (
    append_candidate_batch,
    append_matching_batch,
    init_submission_files,
)
from src.train import train_matching_model


def run_pipeline(
    mode: str = "full",
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
    output_dir: Path = OUTPUT_DIR,
    sample_size: int = None,
    train_sample_size: int = 50000,
    threshold: float = None,
    chunk_size: int = INFERENCE_CHUNK_SIZE,
):
    """
    Execute end-to-end Entity Resolution pipeline.
    """
    print("=" * 75)
    print("       ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE       ")
    print("=" * 75)
    print(f"Mode               : {mode}")
    print(f"Train Directory    : {train_dir}")
    print(f"Test Directory     : {test_dir}")
    print(f"Output Directory   : {output_dir}")
    print(f"Train Sample Size  : {sample_size or train_sample_size}")
    print(f"Test Sample Size   : {sample_size or 'FULL (All Records)'}")
    print(f"Chunk Size         : {chunk_size:,}")
    print("=" * 75)

    model_path = MODELS_DIR / MODEL_FILENAME
    optimal_thresh = threshold or DEFAULT_DECISION_THRESHOLD

    # -------------------------------------------------------------
    # Step 1: Model Training & Threshold Optimization
    # -------------------------------------------------------------
    if mode in ("full", "train"):
        effective_train_sample = sample_size if sample_size is not None else train_sample_size
        print(f"\n[Step 1/4] Loading and Preprocessing Training Data ({effective_train_sample:,} records)...")
        df_s1, df_s2, df_s3, df_gt = load_train_data(train_dir, sample_size=effective_train_sample)
        gt_dict = parse_ground_truth(df_gt)

        print("[Preprocessing] Normalizing Train Source 1...")
        df_s1_prep = preprocess_dataframe(df_s1)
        print("[Preprocessing] Normalizing Train Source 2...")
        df_s2_prep = preprocess_dataframe(df_s2)
        print("[Preprocessing] Normalizing Train Source 3...")
        df_s3_prep = preprocess_dataframe(df_s3)

        print("\n[Step 2/4] Training Entity Matching Model...")
        model, optimal_thresh, val_metrics = train_matching_model(
            df_s1=df_s1_prep,
            df_s2=df_s2_prep,
            df_s3=df_s3_prep,
            gt_dict=gt_dict,
            save_path=model_path,
        )

        if threshold:
            optimal_thresh = threshold
            print(f"[Override] Using user-specified threshold: {optimal_thresh}")

        # Clean training data from memory
        del df_s1, df_s2, df_s3, df_gt, df_s1_prep, df_s2_prep, df_s3_prep

    if mode == "train":
        print("\n[Complete] Training mode finished successfully.")
        return

    # -------------------------------------------------------------
    # Step 2: Streaming Chunked Inference on Test Dataset
    # -------------------------------------------------------------
    if mode in ("full", "predict"):
        print("\n[Step 3/4] Loading and Normalizing Test Target Pool (S2 + S3)...")
        test_s2_path = test_dir / "test_source2.tsv"
        test_s3_path = test_dir / "test_source3.tsv"
        test_s1_path = test_dir / "test_source1.tsv"

        target_sample = (sample_size * 3) if sample_size else None
        df_test_s2 = load_tsv(test_s2_path, sample_size=target_sample)
        df_test_s3 = load_tsv(test_s3_path, sample_size=target_sample)

        print(f"  Loaded Test S2: {len(df_test_s2):,} records")
        print(f"  Loaded Test S3: {len(df_test_s3):,} records")

        print("[Preprocessing] Normalizing Test Source 2...")
        df_test_s2_prep = preprocess_dataframe(df_test_s2)
        del df_test_s2

        print("[Preprocessing] Normalizing Test Source 3...")
        df_test_s3_prep = preprocess_dataframe(df_test_s3)
        del df_test_s3

        # Combine into target pool
        df_test_targets = pd.concat([df_test_s2_prep, df_test_s3_prep], ignore_index=True)
        del df_test_s2_prep, df_test_s3_prep

        print(f"[CandidateBlocker] Indexing {len(df_test_targets):,} target records...")
        blocker = CandidateBlocker(enable_tfidf=True)
        blocker.fit(df_test_targets)

        # Initialize Predictor
        predictor = EntityMatcherPredictor(
            model_path=model_path,
            override_threshold=optimal_thresh,
        )

        # Initialize output TSVs
        match_output_path = output_dir / MATCHING_RESULTS_FILENAME
        cand_output_path = output_dir / CANDIDATE_PAIRS_FILENAME
        init_submission_files(match_output_path, cand_output_path)

        print(f"\n[Step 4/4] Streaming Inference on Test Source 1...")
        df_test_s1_full = load_tsv(test_s1_path, sample_size=sample_size)
        total_s1 = len(df_test_s1_full)
        print(f"  Total Test Source 1 records: {total_s1:,}")

        num_chunks = (total_s1 + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total_s1)
            chunk_df = df_test_s1_full.iloc[start:end].copy()
            chunk_s1_ids = chunk_df["entity_id"].tolist()

            print(f"\n--- Processing Test Chunk {chunk_idx + 1}/{num_chunks} (Rows {start:,} to {end:,}) ---")
            chunk_prep = preprocess_dataframe(chunk_df)

            # 1. Blocking Candidates
            chunk_candidates = blocker.block_candidates(chunk_prep)

            # 2. Append to candidate_pairs.tsv
            append_candidate_batch(chunk_candidates, cand_output_path, chunk_s1_ids)

            # 3. Predict candidate match scores
            chunk_scores = predictor.score_candidates(
                candidate_dict=chunk_candidates,
                df_s1=chunk_prep,
                df_targets=df_test_targets,
            )

            # 4. Postprocessing (thresholding, singleton handling, sorting)
            chunk_matches = postprocess_matches(
                candidate_scores=chunk_scores,
                all_s1_ids=chunk_s1_ids,
                threshold=optimal_thresh,
            )

            # 5. Enforce candidate subset rule: matches ⊆ candidates
            chunk_matches = enforce_candidate_subset(chunk_matches, chunk_candidates)

            # 6. Append to matching_results.tsv
            append_matching_batch(chunk_matches, match_output_path, chunk_s1_ids)

        print("\n" + "=" * 75)
        print("                 PIPELINE EXECUTION COMPLETED                 ")
        print("=" * 75)
        print("Generated Submission Files:")
        print(f"  1. Matching Results : {match_output_path}")
        print(f"  2. Candidate Pairs  : {cand_output_path}")
        print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="ML Challenge 2026: Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "train", "predict"],
        default="full",
        help="Pipeline execution mode (default: full)",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=TRAIN_DIR,
        help="Directory containing training TSV files",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=TEST_DIR,
        help="Directory containing test TSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where output TSVs will be saved",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Subsample N records for testing (default: None for full dataset)",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=50000,
        help="Training dataset sample size (default: 50,000 for fast optimal training)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold override for matching probability",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=INFERENCE_CHUNK_SIZE,
        help="Chunk size for streaming test inference (default: 50,000)",
    )

    args = parser.parse_args()
    run_pipeline(
        mode=args.mode,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        train_sample_size=args.train_sample_size,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
