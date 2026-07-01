"""
Retrieval layer — hybrid TF-IDF + keyword boost search over the SHL catalog.

Design:
- TF-IDF cosine similarity gives broad semantic coverage over names/descriptions.
- Keyword boost rewards exact technology/role name matches in the assessment name.
- Level filter (optional) removes clearly mismatched job levels.
- Results are returned sorted by score, ready for LLM re-ranking.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from catalog import Assessment, CatalogManager

logger = logging.getLogger(__name__)

# ── Scored result ─────────────────────────────────────────────────────────────

@dataclass
class ScoredAssessment:
    assessment: Assessment
    score: float

    def __lt__(self, other: "ScoredAssessment") -> bool:
        return self.score < other.score


# ── TF-IDF index (built once at import time) ───────────────────────────────────

class CatalogIndex:
    """Builds a TF-IDF matrix over all assessment search texts."""

    _instance: Optional["CatalogIndex"] = None

    def __init__(self) -> None:
        self._catalog = CatalogManager.get()
        self._assessments = self._catalog.all()

        logger.info("Building TF-IDF index over %d assessments…", len(self._assessments))
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            max_features=20_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        corpus = self._catalog.get_search_texts()
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        logger.info("TF-IDF index ready  shape=%s", self._tfidf_matrix.shape)

    @classmethod
    def get(cls) -> "CatalogIndex":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def search(
        self,
        query: str,
        top_k: int = 30,
        job_level_filter: Optional[List[str]] = None,
        type_code_filter: Optional[List[str]] = None,
    ) -> List[ScoredAssessment]:
        """
        Return up to top_k ScoredAssessments ranked by TF-IDF + keyword boost.

        Parameters
        ----------
        query : str
            Free-text query derived from conversation context.
        top_k : int
            Number of candidates to return.
        job_level_filter : list[str], optional
            If given, prefer assessments that include at least one of these levels.
        type_code_filter : list[str], optional
            If given, prefer assessments with at least one matching type code.
        """
        if not query.strip():
            return []

        query_vec = self._vectorizer.transform([query.lower()])
        sims = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Keyword boost: reward direct name matches
        query_tokens = set(re.split(r"\W+", query.lower()))
        boosts = np.zeros(len(self._assessments), dtype=float)
        for i, a in enumerate(self._assessments):
            name_tokens = set(re.split(r"\W+", a.name.lower()))
            overlap = query_tokens & name_tokens
            if overlap:
                boosts[i] += 0.15 * len(overlap)

        # Type code filter boost
        if type_code_filter:
            tc_set = {c.upper() for c in type_code_filter}
            for i, a in enumerate(self._assessments):
                if tc_set & set(a.type_codes):
                    boosts[i] += 0.10

        # Job level boost (not hard filter — avoids missing good matches)
        if job_level_filter:
            lvl_set = {lvl.lower() for lvl in job_level_filter}
            for i, a in enumerate(self._assessments):
                a_levels = {lvl.lower() for lvl in a.job_levels}
                if lvl_set & a_levels:
                    boosts[i] += 0.08

        combined = sims + boosts

        # Get top_k indices
        top_indices = np.argsort(combined)[::-1][:top_k]

        results: List[ScoredAssessment] = []
        for idx in top_indices:
            score = float(combined[idx])
            if score > 0.0:  # Skip zero-score items
                results.append(ScoredAssessment(
                    assessment=self._assessments[idx],
                    score=round(score, 4),
                ))

        return results


# ── Public helper ─────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = 30,
    job_level_filter: Optional[List[str]] = None,
    type_code_filter: Optional[List[str]] = None,
) -> List[Assessment]:
    """Convenience wrapper — returns plain Assessment list."""
    index = CatalogIndex.get()
    scored = index.search(
        query=query,
        top_k=top_k,
        job_level_filter=job_level_filter,
        type_code_filter=type_code_filter,
    )
    return [s.assessment for s in scored]


def get_assessments_for_comparison(names: List[str]) -> List[Assessment]:
    """
    For comparison queries: look up assessments by name (exact + fuzzy).
    Returns whatever was found (may be empty if names are not in catalog).
    """
    catalog = CatalogManager.get()
    results: List[Assessment] = []
    seen: set[str] = set()
    for name in names:
        a = catalog.fuzzy_name_lookup(name)
        if a and a.entity_id not in seen:
            results.append(a)
            seen.add(a.entity_id)
    return results
