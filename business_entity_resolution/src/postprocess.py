"""
Postprocessing module for Business Entity Resolution.
Converts raw candidate match probabilities into final decision sets.
Enforces precision-weighted F0.5 decision thresholds, singleton identification,
confidence sorting, deduplication, and candidate subset constraints.
"""

from typing import Dict, List, Set, Tuple


def postprocess_matches(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    all_s1_ids: List[str],
    threshold: float,
    min_confidence: float = 0.0,
) -> Dict[str, List[str]]:
    """
    Apply decision thresholding and singleton filtering to candidate match scores.
    
    Parameters:
      - candidate_scores: Dict mapping s1_id -> list of (target_id, score)
      - all_s1_ids: Complete list of all S1 entity IDs to include (even if empty)
      - threshold: Cutoff probability for positive match prediction
      - min_confidence: Optional floor for minimum single match confidence
      
    Returns:
      - final_matches: Dict mapping s1_id -> list of matched target entity IDs
    """
    final_matches: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

    for s1_id in all_s1_ids:
        scored_pairs = candidate_scores.get(s1_id, [])
        if not scored_pairs:
            continue

        # Filter by threshold and valid ID prefix
        valid_matches = []
        seen_targets: Set[str] = set()

        for tid, score in scored_pairs:
            tid_clean = str(tid).strip()
            if score >= threshold and (tid_clean.startswith("S2-") or tid_clean.startswith("S3-")):
                if tid_clean not in seen_targets:
                    seen_targets.add(tid_clean)
                    valid_matches.append((tid_clean, score))

        # Sort matches by score descending
        valid_matches.sort(key=lambda x: x[1], reverse=True)

        # Singleton handling: If no candidate crosses threshold, keep empty
        final_matches[s1_id] = [tid for tid, _ in valid_matches]

    return final_matches


def enforce_candidate_subset(
    matching_dict: Dict[str, List[str]],
    candidate_dict: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Strictly enforce the rule: matching_results ⊆ candidate_pairs.
    Any matched ID that was not in the candidate set is pruned.
    """
    filtered_matches: Dict[str, List[str]] = {}

    for s1_id, matches in matching_dict.items():
        candidates = set(candidate_dict.get(s1_id, []))
        filtered = [mid for mid in matches if mid in candidates]
        filtered_matches[s1_id] = filtered

    return filtered_matches
