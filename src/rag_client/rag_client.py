"""준비된 VectorDB를 이용하는 확장 가능한 검색 파이프라인."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING, Any

from ailib.agent.llm_client.config import RAGConfig

from .retrievers import (
    DocumentKey,
    HybridRetriever,
    Retriever,
    SearchFunction,
    VectorDBRetriever,
    chunk_key,
)
from .types import RetrievalResult, SearchInput

if TYPE_CHECKING:
    from ailib.agent.llm_client.llm_client import LLMClient
    from ailib.agent.vector_db.vector_db import VectorDB


class RAGClient(ABC):
    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db
        self.retriever = self.create_retriever(vector_db)

    @abstractmethod
    def create_retriever(self, vector_db: VectorDB) -> Retriever:
        """이미 준비된 검색 기능으로 retriever를 구성한다."""

    def invoke(
        self, query: str, top_k: int = 5, *, candidate_k: int | None = None
    ) -> Any:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        candidate_k = top_k if candidate_k is None else candidate_k
        for name, value in (("top_k", top_k), ("candidate_k", candidate_k)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k")

        search_input = self.preprocess(query)
        results = self.retrieve(search_input, candidate_k=candidate_k)
        results = self.postprocess(query, results, top_k=top_k)
        return self.format_output(results)

    def preprocess(self, query: str) -> SearchInput:
        return SearchInput(query, (query,), (query,))

    def retrieve(
        self, search_input: SearchInput, *, candidate_k: int
    ) -> list[RetrievalResult]:
        return self.retriever.invoke(search_input, top_k=candidate_k)

    def postprocess(
        self, query: str, results: list[RetrievalResult], *, top_k: int
    ) -> list[RetrievalResult]:
        return results[:top_k]

    def format_output(self, results: list[RetrievalResult]) -> Any:
        return results


class VectorRAGClient(RAGClient):
    def __init__(self, vector_db: VectorDB, *, retriever: Retriever | None = None):
        self._provided_retriever = retriever
        super().__init__(vector_db)

    def create_retriever(self, vector_db: VectorDB) -> Retriever:
        if self._provided_retriever is not None:
            return self._provided_retriever
        return VectorDBRetriever(vector_db)


class HybridRAGClient(RAGClient):
    """준비된 dense 및 sparse 검색 함수를 가중 RRF로 결합한다."""

    def __init__(
        self,
        vector_db: VectorDB,
        sparse_search: SearchFunction,
        *,
        dense_search: SearchFunction | None = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        document_key: DocumentKey = chunk_key,
        rrf_k: int = 60,
    ):
        if not callable(sparse_search):
            raise ValueError("Hybrid search requires a prepared sparse_search function")
        self._dense = VectorDBRetriever(
            vector_db, search=dense_search, document_key=document_key, rrf_k=rrf_k
        )
        self._sparse = VectorDBRetriever(
            vector_db, search=sparse_search, document_key=document_key, rrf_k=rrf_k
        )
        self._weights = dense_weight, sparse_weight
        super().__init__(vector_db)

    def create_retriever(self, vector_db: VectorDB) -> Retriever:
        return HybridRetriever(
            self._dense,
            self._sparse,
            dense_weight=self._weights[0],
            sparse_weight=self._weights[1],
        )


class DefaultRAGClient(VectorRAGClient):
    """기존 RAGConfig 기반 query expansion 및 rerank 구현."""

    def __init__(
        self,
        vector_db: VectorDB,
        llm_client: LLMClient | None = None,
        config: RAGConfig | None = None,
        *,
        retriever: Retriever | None = None,
        rerank_score_threshold: float | None = None,
    ):
        self.llm_client = llm_client
        self.config = config if config is not None else RAGConfig()
        self.rerank_score_threshold = rerank_score_threshold
        if self.config.preprocess not in (None, "query_expansion"):
            raise ValueError("Unsupported preprocess setting")
        if self.config.postprocess not in (None, "rerank"):
            raise ValueError("Unsupported postprocess setting")
        if self.config.preprocess and getattr(llm_client, "model", None) is None:
            raise ValueError("query_expansion requires an LLM model")
        if self.config.postprocess and getattr(llm_client, "reranker", None) is None:
            raise ValueError("rerank requires a reranker")
        if not isfinite(self.config.similarity_score_threshold):
            raise ValueError("Search score threshold must be finite")
        if rerank_score_threshold is not None:
            if (
                not isfinite(rerank_score_threshold)
                or self.config.postprocess != "rerank"
            ):
                raise ValueError(
                    "A finite rerank threshold requires rerank postprocessing"
                )
        if retriever is None and self.config.rag_search_type != "normal":
            raise ValueError(
                "Non-normal search requires an explicit prepared retriever"
            )
        super().__init__(vector_db, retriever=retriever)

    def preprocess(self, query: str) -> SearchInput:
        if self.config.preprocess is None:
            return super().preprocess(query)
        prompt = (
            "Expand the following query with related keywords to improve search "
            "retrieval. Only return the expanded query, without any prefixes or "
            f"explanations:\n{query}"
        )
        expanded_query = self.llm_client.model.invoke(prompt)
        if not isinstance(expanded_query, str) or not expanded_query.strip():
            raise ValueError("query_expansion must return a non-empty string")
        return SearchInput(query, (expanded_query.strip(),), (expanded_query.strip(),))

    def postprocess(
        self, query: str, results: list[RetrievalResult], *, top_k: int
    ) -> list[RetrievalResult]:
        threshold = self.config.similarity_score_threshold
        if threshold > 0:
            if any(
                result.score_kind == "unknown" or result.higher_is_better is None
                for result in results
            ):
                raise ValueError("Filtering requires an explicit search score meaning")
            results = [
                result
                for result in results
                if (
                    result.score >= threshold
                    if result.higher_is_better
                    else result.score <= threshold
                )
            ]
        if results and self.config.postprocess == "rerank":
            documents = [result.document for result in results]
            reranked = self.llm_client.reranker.rerank(query, documents)
            remaining = list(results)
            results = []
            for item in reranked:
                if not isfinite(item.score):
                    raise ValueError("Reranker returned a non-finite score")
                original = next(
                    (
                        result
                        for result in remaining
                        if result.document == item.document
                    ),
                    None,
                )
                if original is None:
                    raise ValueError(
                        "Reranker returned an unknown or duplicate document"
                    )
                remaining.remove(original)
                if (
                    self.rerank_score_threshold is None
                    or item.score >= self.rerank_score_threshold
                ):
                    results.append(replace(original, rerank_score=item.score))
        return results[:top_k]
