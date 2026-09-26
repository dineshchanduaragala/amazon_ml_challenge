"""
Similarity module for Business Entity Resolution.
Provides string similarity, token overlap, edit distance, and component matching functions.
Keeps similarity metric computation decoupled from model training.
"""

from typing import Dict, Set
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler


def jaccard_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets of tokens.
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return float(intersection / union) if union > 0 else 0.0


def name_similarity(
    name1: str,
    name2: str,
    core1: str = "",
    core2: str = "",
    tokens1: Set[str] = None,
    tokens2: Set[str] = None,
) -> Dict[str, float]:
    """
    Compute dense suite of name similarity features using RapidFuzz and token metrics.
    """
    if tokens1 is None:
        tokens1 = set(name1.split())
    if tokens2 is None:
        tokens2 = set(name2.split())

    ratio_val = fuzz.ratio(name1, name2) / 100.0
    
    if core1 and core2:
        core_ratio_val = fuzz.ratio(core1, core2) / 100.0
    else:
        core_ratio_val = ratio_val

    token_sort_val = fuzz.token_sort_ratio(name1, name2) / 100.0
    token_set_val = fuzz.token_set_ratio(name1, name2) / 100.0
    partial_val = fuzz.partial_ratio(name1, name2) / 100.0
    jaro_winkler_val = JaroWinkler.similarity(name1, name2)
    jaccard_val = jaccard_similarity(tokens1, tokens2)

    len1, len2 = len(name1), len(name2)
    len_diff = float(abs(len1 - len2))
    len_ratio = float(min(len1, len2) / (max(len1, len2) + 1e-5))

    return {
        "name_ratio": ratio_val,
        "name_core_ratio": core_ratio_val,
        "name_token_sort_ratio": token_sort_val,
        "name_token_set_ratio": token_set_val,
        "name_partial_ratio": partial_val,
        "name_jaro_winkler": jaro_winkler_val,
        "name_jaccard": jaccard_val,
        "name_len_diff": len_diff,
        "name_len_ratio": len_ratio,
    }


def address_similarity(
    addr1: str,
    addr2: str,
    tokens1: Set[str] = None,
    tokens2: Set[str] = None,
) -> Dict[str, float]:
    """
    Compute dense suite of address similarity features.
    """
    if tokens1 is None:
        tokens1 = set(addr1.split())
    if tokens2 is None:
        tokens2 = set(addr2.split())

    ratio_val = fuzz.ratio(addr1, addr2) / 100.0
    token_sort_val = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    token_set_val = fuzz.token_set_ratio(addr1, addr2) / 100.0
    jaccard_val = jaccard_similarity(tokens1, tokens2)

    len1, len2 = len(addr1), len(addr2)
    len_diff = float(abs(len1 - len2))
    len_ratio = float(min(len1, len2) / (max(len1, len2) + 1e-5))

    return {
        "addr_ratio": ratio_val,
        "addr_token_sort_ratio": token_sort_val,
        "addr_token_set_ratio": token_set_val,
        "addr_jaccard": jaccard_val,
        "addr_len_diff": len_diff,
        "addr_len_ratio": len_ratio,
    }


def component_similarity(
    postal1: str,
    postal2: str,
    street1: str,
    street2: str,
) -> Dict[str, float]:
    """
    Compute ternary matching indicators for address components:
    1.0 = Exact Match, 0.5 = Missing/Unknown in either record, 0.0 = Conflict.
    """
    # Postal / PIN Code
    if postal1 and postal2:
        postal_match = 1.0 if postal1 == postal2 else 0.0
    else:
        postal_match = 0.5

    # Street Number
    if street1 and street2:
        street_match = 1.0 if street1 == street2 else 0.0
    else:
        street_match = 0.5

    return {
        "postal_code_match": postal_match,
        "street_number_match": street_match,
    }
