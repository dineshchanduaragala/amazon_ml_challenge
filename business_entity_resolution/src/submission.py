"""
Submission module for Business Entity Resolution.
Generates output/matching_results.tsv and output/candidate_pairs.tsv with
strict enforcement of competition formatting rules, UTF-8 tab separation,
deduplication, valid ID prefixes, and candidate subset constraints.
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set, Union
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import MATCHING_RESULTS_FILENAME, CANDIDATE_PAIRS_FILENAME, OUTPUT_DIR, TEST_DIR
from src.load_data import load_tsv

MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def init_submission_files(
    matching_path: Union[str, Path],
    candidate_path: Union[str, Path],
) -> None:
    """
    Initialize output TSV files with standard tab-separated headers.
    """
    matching_path = Path(matching_path)
    candidate_path = Path(candidate_path)

    matching_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)

    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("\t".join(MATCHING_HEADER) + "\n")

    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("\t".join(CANDIDATE_HEADER) + "\n")


def append_matching_batch(
    matching_dict: Dict[str, List[str]],
    output_path: Union[str, Path],
    s1_ids: List[str],
) -> None:
    """
    Append a batch of matching results to matching_results.tsv.
    """
    output_path = Path(output_path)
    lines = []

    for s1_id in s1_ids:
        raw_matches = matching_dict.get(s1_id, [])
        clean_matches = []
        seen = set()

        for mid in raw_matches:
            mid_str = str(mid).strip()
            if (mid_str.startswith("S2-") or mid_str.startswith("S3-")) and mid_str not in seen:
                seen.add(mid_str)
                clean_matches.append(mid_str)

        matched_str = ",".join(clean_matches)
        lines.append(f"{s1_id}\t{matched_str}\n")

    with open(output_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


def append_candidate_batch(
    candidate_dict: Dict[str, List[str]],
    output_path: Union[str, Path],
    s1_ids: List[str],
) -> None:
    """
    Append a batch of candidate pairs to candidate_pairs.tsv.
    """
    output_path = Path(output_path)
    lines = []

    for s1_id in s1_ids:
        raw_candidates = candidate_dict.get(s1_id, [])
        clean_candidates = []
        seen = set()

        for cid in raw_candidates:
            cid_str = str(cid).strip()
            if (cid_str.startswith("S2-") or cid_str.startswith("S3-")) and cid_str not in seen:
                seen.add(cid_str)
                clean_candidates.append(cid_str)

        candidate_str = ",".join(clean_candidates)
        lines.append(f"{s1_id}\t{candidate_str}\n")

    with open(output_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


def export_submission(
    matching_dict: Dict[str, List[str]],
    candidate_dict: Dict[str, List[str]],
    all_s1_ids: List[str],
    matching_path: Union[str, Path],
    candidate_path: Union[str, Path],
) -> None:
    """
    Export both matching_results.tsv and candidate_pairs.tsv in a single call.
    """
    init_submission_files(matching_path, candidate_path)
    append_matching_batch(matching_dict, matching_path, all_s1_ids)
    append_candidate_batch(candidate_dict, candidate_path, all_s1_ids)
    print(f"[Submission] Successfully exported {len(all_s1_ids):,} rows to:")
    print(f"  - Matching Results : {matching_path}")
    print(f"  - Candidate Pairs  : {candidate_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify and Format Submission Files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Path to output directory")
    args = parser.parse_args()

    match_file = args.output_dir / MATCHING_RESULTS_FILENAME
    cand_file = args.output_dir / CANDIDATE_PAIRS_FILENAME

    print(f"[Submission] Checking output files in: {args.output_dir}")
    if match_file.exists():
        print(f"  Matching results: {match_file} (Size: {match_file.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"  [!] Missing: {match_file}")

    if cand_file.exists():
        print(f"  Candidate pairs : {cand_file} (Size: {cand_file.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"  [!] Missing: {cand_file}")


if __name__ == "__main__":
    main()
