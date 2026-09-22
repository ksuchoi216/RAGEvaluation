"""준비된 검색 기능 연결과 순위 기반 결과 결합."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING, Protocol

from langchain_core.documents import Document

from ailib.agent.shared.types import ScoredDocument

from .types import RetrievalResult, SearchInput, SourceScore

if TYPE_CHECKING:
    from ailib.agent.vector_db.vector_db import VectorDB


class SearchFunction(Protocol):
    def __call__(self, query: str, *, top_k: int) -> list[ScoredDocument]: ...


class Retriever(Protocol):
    def invoke(
        self, search_input: SearchInput, *, top_k: int
    ) -> list[RetrievalResult]: ...


DocumentKey = Callable[[Document], Hashable]


def chunk_key(document: Document) -> Hashable:
    """문서 전체가 아니라 개별 청크를 식별한다."""
    if document.id is not None:
        return document.id
    metadata = document.metadata
    if metadata.get("doc_id") is not None and metadata.get("chunk_index") is not None:
        return metadata["doc_id"], metadata["chunk_index"]
    raise ValueError("Fusion requires a chunk identifier or a document_key callback")


def fuse_rankings(
    rankings: Sequence[list[RetrievalResult]],
    weights: Sequence[float],
    *,
    top_k: int,
    document_key: DocumentKey,
    rrf_k: int,
) -> list[RetrievalResult]:
    results: dict[Hashable, RetrievalResult] = {}
    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue
        seen = set()
        for rank, result in enumerate(ranking, start=1):
            key = document_key(result.document)
            if key in seen:
                continue
            seen.add(key)
            contribution = weight / (rrf_k + rank)
            previous = results.get(key)
            if previous is None:
                results[key] = replace(
                    result, score=contribution, score_kind="rrf", higher_is_better=True
                )
            else:
                results[key] = replace(
                    previous,
                    score=previous.score + contribution,
                    source_scores=previous.source_scores + result.source_scores,
                )
    return sorted(results.values(), key=lambda result: result.score, reverse=True)[
        :top_k
    ]


class VectorDBRetriever:
    """DB의 검색 시그니처가 다르면 search에 인자를 바인딩한 함수를 전달한다."""

    def __init__(
        self,
        vector_db: VectorDB,
        *,
        search: SearchFunction | None = None,
        score_kind: str | None = None,
        higher_is_better: bool | None = None,
        document_key: DocumentKey = chunk_key,
        rrf_k: int = 60,
    ):
        self.search = (
            search if search is not None else getattr(vector_db, "search", None)
        )
        if not callable(self.search):
            raise ValueError("VectorDB must provide search or a search adapter")
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        self.score_kind = score_kind or "unknown"
        self.higher_is_better = higher_is_better
        if search is None:
            if score_kind is None:
                self.score_kind = getattr(vector_db, "search_score_kind", "unknown")
            if higher_is_better is None:
                self.higher_is_better = getattr(
                    vector_db, "search_higher_is_better", None
                )
        self.document_key = document_key
        self.rrf_k = rrf_k

    def invoke(self, search_input: SearchInput, *, top_k: int) -> list[RetrievalResult]:
        return self.search_queries(
            search_input.dense_queries, top_k=top_k, source="dense"
        )

    def search_queries(
        self, queries: tuple[str, ...], *, top_k: int, source: str
    ) -> list[RetrievalResult]:
        if not queries:
            raise ValueError(f"{source} queries must not be empty")
        rankings = []
        for query in dict.fromkeys(queries):
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Search queries must be non-empty strings")
            ranking = []
            for rank, result in enumerate(self.search(query, top_k=top_k), start=1):
                if not isfinite(result.score):
                    raise ValueError("Search returned a non-finite score")
                score = SourceScore(
                    source,
                    query,
                    result.score,
                    self.score_kind,
                    self.higher_is_better,
                    rank,
                )
                ranking.append(
                    RetrievalResult(
                        result.document,
                        result.score,
                        self.score_kind,
                        self.higher_is_better,
                        source_scores=(score,),
                    )
                )
            rankings.append(ranking)
        if len(rankings) == 1:
            return rankings[0][:top_k]
        return fuse_rankings(
            rankings,
            [1 / len(rankings)] * len(rankings),
            top_k=top_k,
            document_key=self.document_key,
            rrf_k=self.rrf_k,
        )


class HybridRetriever:
    def __init__(
        self,
        dense: VectorDBRetriever,
        sparse: VectorDBRetriever,
        *,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ):
        weights = (dense_weight, sparse_weight)
        if (
            any(not isfinite(weight) or weight < 0 for weight in weights)
            or not isfinite(sum(weights))
            or sum(weights) <= 0
        ):
            raise ValueError("Search weights must be finite, non-negative, and sum > 0")
        self.weights = tuple(weight / sum(weights) for weight in weights)
        self.dense = dense
        self.sparse = sparse

    def invoke(self, search_input: SearchInput, *, top_k: int) -> list[RetrievalResult]:
        rankings = [
            retriever.search_queries(queries, top_k=top_k, source=source)
            if weight
            else []
            for retriever, queries, source, weight in (
                (self.dense, search_input.dense_queries, "dense", self.weights[0]),
                (self.sparse, search_input.sparse_queries, "sparse", self.weights[1]),
            )
        ]
        return fuse_rankings(
            rankings,
            self.weights,
            top_k=top_k,
            document_key=self.dense.document_key,
            rrf_k=self.dense.rrf_k,
        )
