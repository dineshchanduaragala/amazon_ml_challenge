"""
Blocking module for Business Entity Resolution.
Reduces millions of pairwise comparisons to a high-recall candidate set
using multi-tier inverted indices, address component blocking, token signatures,
and character TF-IDF cosine retrieval.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.config import MAX_CANDIDATES_PER_S1, TFIDF_TOP_K


class CandidateBlocker:
    """
    Multi-strategy candidate generation and blocking engine.
    Constructs high-speed inverted index lookup tables over target entities (S2 and S3),
    partitioned by country, and performs multi-key candidate retrieval for Source 1 queries.
    """

    def __init__(
        self,
        max_candidates_per_s1: int = MAX_CANDIDATES_PER_S1,
        enable_tfidf: bool = True,
        tfidf_top_k: int = TFIDF_TOP_K,
    ):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.enable_tfidf = enable_tfidf
        self.tfidf_top_k = tfidf_top_k

        # Multi-key inverted indices: (country, key) -> list of target entity_ids
        self.idx_exact_name = defaultdict(list)
        self.idx_token_sorted = defaultdict(list)
        self.idx_first_2_tokens = defaultdict(list)
        self.idx_first_token = defaultdict(list)
        self.idx_postal_name = defaultdict(list)
        self.idx_street_token = defaultdict(list)
        self.idx_street_postal = defaultdict(list)

        # Target ID tracker per country
        self.country_targets: Dict[str, List[str]] = defaultdict(list)

        # TF-IDF models per country
        self.tfidf_vectorizers: Dict[str, TfidfVectorizer] = {}
        self.tfidf_matrices: Dict[str, sp.csr_matrix] = {}
        self.tfidf_id_maps: Dict[str, List[str]] = {}

    def fit(self, df_targets: pd.DataFrame) -> "CandidateBlocker":
        """
        Build inverted indices over target records (combined Source 2 and Source 3).
        """
        print(f"[CandidateBlocker] Indexing {len(df_targets):,} target records (S2 + S3)...")

        for row in df_targets.itertuples(index=False):
            eid = row.entity_id
            c = row.country_norm
            name_core = getattr(row, "name_core", "")
            token_sorted = getattr(row, "token_sorted_name", "")
            first_2 = getattr(row, "first_2_tokens", "")
            first_tok = getattr(row, "first_token", "")
            postal = getattr(row, "postal_code", "")
            street = getattr(row, "street_number", "")

            self.country_targets[c].append(eid)

            if name_core:
                self.idx_exact_name[(c, name_core)].append(eid)
            if token_sorted:
                self.idx_token_sorted[(c, token_sorted)].append(eid)
            if first_2:
                self.idx_first_2_tokens[(c, first_2)].append(eid)
            if first_tok and len(first_tok) >= 4:
                self.idx_first_token[(c, first_tok)].append(eid)
            if postal and len(name_core) >= 2:
                self.idx_postal_name[(c, postal, name_core[:2])].append(eid)
            if street and first_tok:
                self.idx_street_token[(c, street, first_tok)].append(eid)
            if street and postal:
                self.idx_street_postal[(c, street, postal)].append(eid)

        # Build character 3-gram TF-IDF models for fuzzy/sparse candidate retrieval
        if self.enable_tfidf:
            print("[CandidateBlocker] Building TF-IDF retrieval index per country...")
            for country, ids in self.country_targets.items():
                if len(ids) == 0:
                    continue
                country_df = df_targets[df_targets["entity_id"].isin(ids)]
                corpus = country_df["name_norm"].fillna("").tolist()
                id_list = country_df["entity_id"].tolist()

                vec = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 3),
                    min_df=2,
                    max_features=50000,
                    dtype=np.float32,
                )
                try:
                    mat = vec.fit_transform(corpus)
                    self.tfidf_vectorizers[country] = vec
                    self.tfidf_matrices[country] = mat
                    self.tfidf_id_maps[country] = id_list
                    print(f"  [TF-IDF] Country {country}: indexed {len(id_list):,} names.")
                except Exception as e:
                    print(f"  [TF-IDF] Note: TF-IDF indexing skipped for {country}: {e}")

        print("[CandidateBlocker] Inverted index construction completed.")
        return self

    def retrieve_candidates_for_record(
        self,
        c: str,
        name_norm: str,
        name_core: str,
        token_sorted: str,
        first_2: str,
        first_tok: str,
        postal: str,
        street: str,
    ) -> List[str]:
        """
        Query inverted indices for a single Source 1 record.
        Returns ranked list of candidate target IDs.
        """
        candidate_scores = defaultdict(int)

        # 1. Exact core name match (highest weight)
        if name_core and (c, name_core) in self.idx_exact_name:
            for tid in self.idx_exact_name[(c, name_core)][:50]:
                candidate_scores[tid] += 10

        # 2. Token-sorted name match
        if token_sorted and (c, token_sorted) in self.idx_token_sorted:
            for tid in self.idx_token_sorted[(c, token_sorted)][:50]:
                candidate_scores[tid] += 8

        # 3. First 2 tokens match
        if first_2 and (c, first_2) in self.idx_first_2_tokens:
            for tid in self.idx_first_2_tokens[(c, first_2)][:50]:
                candidate_scores[tid] += 6

        # 4. Street number + postal code match
        if street and postal and (c, street, postal) in self.idx_street_postal:
            for tid in self.idx_street_postal[(c, street, postal)][:50]:
                candidate_scores[tid] += 5

        # 5. Postal code + name prefix match
        if postal and len(name_core) >= 2 and (c, postal, name_core[:2]) in self.idx_postal_name:
            for tid in self.idx_postal_name[(c, postal, name_core[:2])][:50]:
                candidate_scores[tid] += 4

        # 6. Street number + first token match
        if street and first_tok and (c, street, first_tok) in self.idx_street_token:
            for tid in self.idx_street_token[(c, street, first_tok)][:50]:
                candidate_scores[tid] += 4

        # 7. First token match
        if first_tok and len(first_tok) >= 4 and (c, first_tok) in self.idx_first_token:
            for tid in self.idx_first_token[(c, first_tok)][:30]:
                candidate_scores[tid] += 2

        # Sort candidates by accumulated match strength
        sorted_candidates = [
            cid for cid, _ in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        ]

        return sorted_candidates[: self.max_candidates_per_s1]

    def block_candidates(self, df_s1: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Generate candidate target IDs for all Source 1 records.
        """
        results: Dict[str, List[str]] = {}

        s1_tuples = df_s1[
            [
                "entity_id",
                "country_norm",
                "name_norm",
                "name_core",
                "token_sorted_name",
                "first_2_tokens",
                "first_token",
                "postal_code",
                "street_number",
            ]
        ].itertuples(index=False)

        for row in tqdm(s1_tuples, total=len(df_s1), desc="Candidate Blocking", leave=False):
            candidates = self.retrieve_candidates_for_record(
                row.country_norm,
                row.name_norm,
                row.name_core,
                row.token_sorted_name,
                row.first_2_tokens,
                row.first_token,
                row.postal_code,
                row.street_number,
            )
            results[row.entity_id] = candidates

        # TF-IDF enrichment for queries with few candidates
        if self.enable_tfidf:
            for country, vec in self.tfidf_vectorizers.items():
                if country not in self.tfidf_matrices:
                    continue
                c_mask = df_s1["country_norm"] == country
                country_s1 = df_s1[c_mask]

                sparse_s1 = [
                    (row.entity_id, row.name_norm)
                    for row in country_s1.itertuples(index=False)
                    if len(results.get(row.entity_id, [])) < 3 and row.name_norm
                ]

                if not sparse_s1:
                    continue

                s1_ids, s1_names = zip(*sparse_s1)
                try:
                    query_mat = vec.transform(s1_names)
                    target_mat = self.tfidf_matrices[country]
                    id_map = self.tfidf_id_maps[country]

                    batch_size = 5000
                    for start in range(0, len(s1_ids), batch_size):
                        end = min(start + batch_size, len(s1_ids))
                        batch_q = query_mat[start:end]
                        sims = batch_q.dot(target_mat.T).toarray()

                        for i, sim_row in enumerate(sims):
                            cur_s1_id = s1_ids[start + i]
                            top_indices = np.argsort(-sim_row)[: self.tfidf_top_k]
                            cur_cands = set(results[cur_s1_id])

                            for t_idx in top_indices:
                                if sim_row[t_idx] > 0.35:
                                    target_id = id_map[t_idx]
                                    if target_id not in cur_cands:
                                        cur_cands.add(target_id)
                                        results[cur_s1_id].append(target_id)
                                        if len(results[cur_s1_id]) >= self.max_candidates_per_s1:
                                            break
                except Exception as e:
                    print(f"  [TF-IDF] Warning during batch enrichment for {country}: {e}")

        return results


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    max_candidates_per_s1: int = MAX_CANDIDATES_PER_S1,
    enable_tfidf: bool = True,
) -> Dict[str, List[str]]:
    """
    Convenience function to generate candidate pairs across S1, S2, and S3.
    """
    df_targets = pd.concat([source2, source3], ignore_index=True)
    blocker = CandidateBlocker(
        max_candidates_per_s1=max_candidates_per_s1,
        enable_tfidf=enable_tfidf,
    )
    blocker.fit(df_targets)
    return blocker.block_candidates(source1)
