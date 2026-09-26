"""
Feature Engineering module for Business Entity Resolution.
Converts each candidate pair into a dense, discriminative feature vector
using string similarity, token overlap, address component matching, and metadata.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.similarity import (
    name_similarity,
    address_similarity,
    component_similarity,
    jaccard_similarity,
)

FEATURE_NAMES = [
    "name_ratio",
    "name_core_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaro_winkler",
    "name_jaccard",
    "name_first_token_match",
    "name_first_2_tokens_match",
    "name_len_diff",
    "name_len_ratio",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaccard",
    "street_number_match",
    "postal_code_match",
    "addr_len_diff",
    "addr_len_ratio",
    "combined_token_jaccard",
    "source_is_s2",
    "source_is_s3",
    "country_is_us",
    "country_is_india",
    "country_is_france",
]


def extract_pair_features(
    s1_row: dict,
    target_row: dict,
) -> List[float]:
    """
    Extract feature vector for a single (Source 1, Target) pair.
    """
    s1_name = s1_row.get("name_norm", "")
    s1_core = s1_row.get("name_core", "")
    s1_tokens = set(s1_row.get("name_tokens", []))
    s1_first = s1_row.get("first_token", "")
    s1_first_2 = s1_row.get("first_2_tokens", "")
    s1_addr = s1_row.get("address_norm", "")
    s1_addr_tokens = set(s1_row.get("address_tokens", []))
    s1_postal = s1_row.get("postal_code", "")
    s1_street = s1_row.get("street_number", "")
    s1_country = str(s1_row.get("country_norm", "")).lower()

    t_name = target_row.get("name_norm", "")
    t_core = target_row.get("name_core", "")
    t_tokens = set(target_row.get("name_tokens", []))
    t_first = target_row.get("first_token", "")
    t_first_2 = target_row.get("first_2_tokens", "")
    t_addr = target_row.get("address_norm", "")
    t_addr_tokens = set(target_row.get("address_tokens", []))
    t_postal = target_row.get("postal_code", "")
    t_street = target_row.get("street_number", "")
    t_id = target_row.get("entity_id", "")

    # 1. Name Similarities
    name_sims = name_similarity(
        s1_name, t_name, core1=s1_core, core2=t_core, tokens1=s1_tokens, tokens2=t_tokens
    )
    name_first_match = 1.0 if (s1_first and t_first and s1_first == t_first) else 0.0
    name_first_2_match = 1.0 if (s1_first_2 and t_first_2 and s1_first_2 == t_first_2) else 0.0

    # 2. Address Similarities
    addr_sims = address_similarity(
        s1_addr, t_addr, tokens1=s1_addr_tokens, tokens2=t_addr_tokens
    )
    comp_sims = component_similarity(s1_postal, t_postal, s1_street, t_street)

    # 3. Cross & Metadata
    all_s1_tokens = s1_tokens | s1_addr_tokens
    all_t_tokens = t_tokens | t_addr_tokens
    combined_jaccard = jaccard_similarity(all_s1_tokens, all_t_tokens)

    source_is_s2 = 1.0 if t_id.startswith("S2-") else 0.0
    source_is_s3 = 1.0 if t_id.startswith("S3-") else 0.0

    country_is_us = 1.0 if s1_country == "us" else 0.0
    country_is_india = 1.0 if s1_country == "india" else 0.0
    country_is_france = 1.0 if s1_country == "france" else 0.0

    return [
        name_sims["name_ratio"],
        name_sims["name_core_ratio"],
        name_sims["name_token_sort_ratio"],
        name_sims["name_token_set_ratio"],
        name_sims["name_partial_ratio"],
        name_sims["name_jaro_winkler"],
        name_sims["name_jaccard"],
        name_first_match,
        name_first_2_match,
        name_sims["name_len_diff"],
        name_sims["name_len_ratio"],
        addr_sims["addr_ratio"],
        addr_sims["addr_token_sort_ratio"],
        addr_sims["addr_token_set_ratio"],
        addr_sims["addr_jaccard"],
        comp_sims["street_number_match"],
        comp_sims["postal_code_match"],
        addr_sims["addr_len_diff"],
        addr_sims["addr_len_ratio"],
        combined_jaccard,
        source_is_s2,
        source_is_s3,
        country_is_us,
        country_is_india,
        country_is_france,
    ]


def build_feature_matrix(
    candidate_dict: Dict[str, List[str]],
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    desc: str = "Extracting Features",
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Construct 2D numpy feature matrix for all candidate pairs in candidate_dict.
    Returns:
      - X: np.ndarray of shape (num_pairs, num_features)
      - pair_ids: List of (source1_id, target_id)
    """
    s1_dict = df_s1.set_index("entity_id").to_dict(orient="index")
    target_dict = df_targets.set_index("entity_id").to_dict(orient="index")

    for eid, row in target_dict.items():
        row["entity_id"] = eid

    feature_rows: List[List[float]] = []
    pair_ids: List[Tuple[str, str]] = []

    for s1_id, cand_ids in tqdm(candidate_dict.items(), desc=desc, leave=False):
        if s1_id not in s1_dict or not cand_ids:
            continue
        s1_row = s1_dict[s1_id]

        for cid in cand_ids:
            if cid not in target_dict:
                continue
            target_row = target_dict[cid]
            feat_vec = extract_pair_features(s1_row, target_row)
            feature_rows.append(feat_vec)
            pair_ids.append((s1_id, cid))

    if not feature_rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), []

    X = np.array(feature_rows, dtype=np.float32)
    return X, pair_ids
