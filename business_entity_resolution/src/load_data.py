"""
Data Loading module for Business Entity Resolution.
Responsible exclusively for reading TSV data files and parsing ground truth tables.
No ML or feature logic is placed here.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union
import pandas as pd


def load_tsv(
    file_path: Union[str, Path],
    sample_size: Optional[int] = None,
    usecols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Read a single TSV file with robust encoding and null handling.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"TSV file not found at: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        nrows=sample_size,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    # Clean object/string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def load_train_data(
    train_dir: Union[str, Path],
    sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load training datasets: Source 1, Source 2, Source 3, and Ground Truth.
    
    When sample_size is provided, strategically includes all true match target records
    for the sampled Source 1 IDs to maintain accurate positive/negative training labels.
    """
    train_dir = Path(train_dir)
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    print(f"[DataLoader] Loading training data from: {train_dir}")
    df_gt = load_tsv(gt_path, sample_size=sample_size)
    print(f"  Loaded Ground Truth: {len(df_gt):,} records")

    if sample_size is not None:
        target_s1_ids = set(df_gt["source1_entity_id"])
        
        # Load and filter Source 1
        df_s1_full = load_tsv(s1_path)
        df_s1 = df_s1_full[df_s1_full["entity_id"].isin(target_s1_ids)].copy()
        del df_s1_full
        print(f"  Filtered Source 1: {len(df_s1):,} records")

        # Collect target match IDs from ground truth
        needed_targets = set()
        for m_str in df_gt["matched_entity_ids"]:
            if m_str:
                for mid in m_str.split(","):
                    mid = mid.strip()
                    if mid:
                        needed_targets.add(mid)

        needed_s2 = {tid for tid in needed_targets if tid.startswith("S2-")}
        needed_s3 = {tid for tid in needed_targets if tid.startswith("S3-")}

        # Load Source 2 (true targets + sample pool)
        df_s2_full = load_tsv(s2_path)
        s2_mask = df_s2_full["entity_id"].isin(needed_s2)
        df_s2_pos = df_s2_full[s2_mask]
        df_s2_extra = df_s2_full[~s2_mask].head(sample_size * 2)
        df_s2 = pd.concat([df_s2_pos, df_s2_extra], ignore_index=True)
        del df_s2_full, df_s2_pos, df_s2_extra
        print(f"  Loaded Source 2: {len(df_s2):,} records ({len(needed_s2):,} positive targets)")

        # Load Source 3 (true targets + sample pool)
        df_s3_full = load_tsv(s3_path)
        s3_mask = df_s3_full["entity_id"].isin(needed_s3)
        df_s3_pos = df_s3_full[s3_mask]
        df_s3_extra = df_s3_full[~s3_mask].head(sample_size * 2)
        df_s3 = pd.concat([df_s3_pos, df_s3_extra], ignore_index=True)
        del df_s3_full, df_s3_pos, df_s3_extra
        print(f"  Loaded Source 3: {len(df_s3):,} records ({len(needed_s3):,} positive targets)")
    else:
        df_s1 = load_tsv(s1_path)
        print(f"  Loaded Source 1: {len(df_s1):,} records")

        df_s2 = load_tsv(s2_path)
        print(f"  Loaded Source 2: {len(df_s2):,} records")

        df_s3 = load_tsv(s3_path)
        print(f"  Loaded Source 3: {len(df_s3):,} records")

    return df_s1, df_s2, df_s3, df_gt


def load_test_data(
    test_dir: Union[str, Path],
    sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load test datasets: Source 1, Source 2, and Source 3.
    """
    test_dir = Path(test_dir)
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    print(f"[DataLoader] Loading test data from: {test_dir}")
    df_s1 = load_tsv(s1_path, sample_size=sample_size)
    print(f"  Loaded Test Source 1: {len(df_s1):,} records")

    target_sample = (sample_size * 3) if sample_size else None
    df_s2 = load_tsv(s2_path, sample_size=target_sample)
    print(f"  Loaded Test Source 2: {len(df_s2):,} records")

    df_s3 = load_tsv(s3_path, sample_size=target_sample)
    print(f"  Loaded Test Source 3: {len(df_s3):,} records")

    return df_s1, df_s2, df_s3


def parse_ground_truth(df_gt: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Convert ground truth DataFrame into a mapping:
    source1_entity_id -> set of matched_entity_ids (S2 and S3 IDs).
    """
    gt_dict: Dict[str, Set[str]] = {}
    for row in df_gt.itertuples(index=False):
        s1_id = row.source1_entity_id
        matched_str = getattr(row, "matched_entity_ids", "")
        if matched_str and isinstance(matched_str, str):
            matches = {m.strip() for m in matched_str.split(",") if m.strip()}
            gt_dict[s1_id] = matches
        else:
            gt_dict[s1_id] = set()
    return gt_dict
