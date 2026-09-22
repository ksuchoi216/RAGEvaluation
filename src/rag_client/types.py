"""검색 단계 사이에 전달하는 입력과 결과."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class SearchInput:
    original_query: str
    dense_queries: tuple[str, ...]
    sparse_queries: tuple[str, ...]


@dataclass(frozen=True)
class SourceScore:
    source: str
    query: str
    score: float
    score_kind: str
    higher_is_better: bool | None
    rank: int


@dataclass(frozen=True)
class RetrievalResult:
    """score는 검색 점수이며, rerank 후에도 덮어쓰지 않는다."""

    document: Document
    score: float
    score_kind: str = "unknown"
    higher_is_better: bool | None = None
    rerank_score: float | None = None
    source_scores: tuple[SourceScore, ...] = ()
