from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from ailib.agent.llm_client.config import RAGConfig
from rag_client import (
    DefaultRAGClient,
    HybridRAGClient,
    RAGClient,
    SearchInput,
    VectorDBRetriever,
    VectorRAGClient,
)
from ailib.agent.shared.types import ScoredDocument


def scored(key, score):
    return ScoredDocument(
        Document(id=key, page_content=key, metadata={"page": 1}), score
    )


class PreparedDB:
    search_score_kind = "cosine"
    search_higher_is_better = True

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results[query][:top_k]


def test_pipeline_preserves_original_question_and_runs_overrides_in_order():
    db = PreparedDB({"expanded": [scored("a", 0.8), scored("b", 0.7)]})
    events = []

    class CustomRAGClient(VectorRAGClient):
        def create_retriever(self, vector_db):
            events.append("init")
            return super().create_retriever(vector_db)

        def preprocess(self, query):
            events.append("preprocess")
            return SearchInput(query, ("expanded",), (query,))

        def retrieve(self, search_input, *, candidate_k):
            events.append("retrieve")
            return super().retrieve(search_input, candidate_k=candidate_k)

        def postprocess(self, query, results, *, top_k):
            events.append(query)
            return super().postprocess(query, results, top_k=top_k)

        def format_output(self, results):
            events.append("format")
            return [result.document.id for result in results]

    client = CustomRAGClient(db)
    assert client.invoke("original", top_k=1, candidate_k=2) == ["a"]
    assert client.invoke("original", top_k=1, candidate_k=2) == ["a"]
    assert events == ["init"] + ["preprocess", "retrieve", "original", "format"] * 2
    assert db.calls == [("expanded", 2), ("expanded", 2)]


def test_base_class_requires_retriever_implementation():
    with pytest.raises(TypeError, match="abstract"):
        RAGClient(PreparedDB({}))


def test_empty_results_reach_formatter():
    class CustomRAGClient(VectorRAGClient):
        def format_output(self, results):
            return {"count": len(results)}

    assert CustomRAGClient(PreparedDB({"q": []})).invoke("q") == {"count": 0}


@pytest.mark.parametrize(
    "top_k,candidate_k", [(0, 2), (2, 1), (True, 2), (1.5, 2), (1, -1), (1, False)]
)
def test_invalid_limits_fail_before_search(top_k, candidate_k):
    db = PreparedDB({})
    with pytest.raises(ValueError):
        VectorRAGClient(db).invoke("q", top_k=top_k, candidate_k=candidate_k)
    assert not db.calls


def test_retrieval_errors_propagate():
    def failed_search(query, top_k):
        raise ConnectionError("unavailable")

    db = PreparedDB({})
    client = VectorRAGClient(db, retriever=VectorDBRetriever(db, search=failed_search))
    with pytest.raises(ConnectionError, match="unavailable"):
        client.invoke("q")


def test_hyde_uses_separate_dense_and_sparse_queries_and_preserves_scores():
    db = PreparedDB({"hypothesis": [scored("a", 0.9), scored("b", 0.8)]})
    sparse = PreparedDB({"original": [scored("b", 12), scored("c", 10)]})

    class HyDEClient(HybridRAGClient):
        def preprocess(self, query):
            return SearchInput(query, ("hypothesis",), (query,))

    results = HyDEClient(db, sparse.search).invoke("original", top_k=2)
    assert [result.document.id for result in results] == ["b", "a"]
    assert results[0].score == pytest.approx(0.5 / 62 + 0.5 / 61)
    assert results[0].score_kind == "rrf"
    assert [
        (score.source, score.query, score.score) for score in results[0].source_scores
    ] == [("dense", "hypothesis", 0.8), ("sparse", "original", 12)]
    assert results[0].document.metadata == {"page": 1}


def test_multi_query_fuses_rankings_and_does_not_double_count_duplicate_queries():
    db = PreparedDB(
        {
            "one": [scored("a", 0.9), scored("b", 0.8)],
            "two": [scored("b", 0.5), scored("c", 0.4)],
        }
    )

    class MultiQueryClient(VectorRAGClient):
        def preprocess(self, query):
            return SearchInput(query, ("one", "two", "one"), (query,))

    results = MultiQueryClient(db).invoke("q", top_k=2)
    assert [result.document.id for result in results] == ["b", "a"]
    assert results[0].score == pytest.approx(0.5 / 62 + 0.5 / 61)
    assert db.calls == [("one", 2), ("two", 2)]


def test_fusion_preserves_distinct_chunks_and_documents_without_document_ids():
    documents = [
        Document(
            page_content="same text", metadata={"doc_id": doc, "chunk_index": chunk}
        )
        for doc, chunk in [("a", 0), ("a", 1), ("b", 0)]
    ]
    db = PreparedDB({"q": [ScoredDocument(doc, 0.8) for doc in documents]})
    results = HybridRAGClient(db, db.search).invoke("q", top_k=3)
    assert [result.document.metadata for result in results] == [
        doc.metadata for doc in documents
    ]


def test_fusion_requires_stable_identity_but_allows_custom_key():
    document = Document(page_content="text", metadata={"custom_key": "a"})
    db = PreparedDB({"q": [ScoredDocument(document, 0.8)]})
    with pytest.raises(ValueError, match="identity|identifier|chunk"):
        HybridRAGClient(db, db.search).invoke("q")
    client = HybridRAGClient(
        db, db.search, document_key=lambda doc: doc.metadata["custom_key"]
    )
    assert len(client.invoke("q")) == 1


@pytest.mark.parametrize(
    "weights", [(-1, 1), (0, 0), (float("nan"), 1), (1, float("inf"))]
)
def test_invalid_hybrid_weights_fail_at_initialization(weights):
    db = PreparedDB({})
    with pytest.raises(ValueError):
        HybridRAGClient(
            db, db.search, dense_weight=weights[0], sparse_weight=weights[1]
        )


def test_zero_weight_search_is_not_called():
    db = PreparedDB({"q": [scored("a", 0.8)]})

    def unavailable(query, top_k):
        raise AssertionError("disabled search called")

    client = HybridRAGClient(db, unavailable, dense_weight=1, sparse_weight=0)
    assert client.invoke("q")[0].document.id == "a"


def test_default_client_expands_then_reranks_original_question_without_losing_scores():
    db = PreparedDB({"expanded": [scored("a", 0.9), scored("b", 0.8)]})
    seen = []

    def rerank(query, documents):
        seen.append((query, [doc.id for doc in documents]))
        return [ScoredDocument(documents[1], 0.99), ScoredDocument(documents[0], 0.5)]

    llm = SimpleNamespace(
        model=SimpleNamespace(invoke=lambda prompt: " expanded "),
        reranker=SimpleNamespace(rerank=rerank),
    )
    config = RAGConfig()
    config.preprocess = "query_expansion"
    config.postprocess = "rerank"
    results = DefaultRAGClient(db, llm, config).invoke(
        "original", top_k=1, candidate_k=2
    )
    assert seen == [("original", ["a", "b"])]
    assert results[0].document.id == "b"
    assert results[0].score == 0.8
    assert results[0].rerank_score == 0.99


@pytest.mark.parametrize("higher_is_better,expected", [(True, "far"), (False, "near")])
def test_threshold_uses_search_score_direction(higher_is_better, expected):
    db = PreparedDB({"q": [scored("near", 0.1), scored("far", 0.9)]})
    db.search_higher_is_better = higher_is_better
    config = RAGConfig()
    config.similarity_score_threshold = 0.5
    assert DefaultRAGClient(db, None, config).invoke("q")[0].document.id == expected


def test_unknown_score_threshold_is_rejected():
    db = PreparedDB({"q": [scored("a", 0.8)]})
    db.search_score_kind = "unknown"
    db.search_higher_is_better = None
    config = RAGConfig()
    config.similarity_score_threshold = 0.5
    with pytest.raises(ValueError, match="score"):
        DefaultRAGClient(db, None, config).invoke("q")


def test_empty_results_do_not_call_reranker():
    config = RAGConfig()
    config.postprocess = "rerank"

    def rerank(query, documents):
        raise AssertionError("reranker called for empty results")

    llm = SimpleNamespace(reranker=SimpleNamespace(rerank=rerank))
    assert DefaultRAGClient(PreparedDB({"q": []}), llm, config).invoke("q") == []


@pytest.mark.parametrize("answer", [None, "", "  ", {"text": "query"}])
def test_invalid_query_expansion_is_not_silently_ignored(answer):
    config = RAGConfig()
    config.preprocess = "query_expansion"
    llm = SimpleNamespace(model=SimpleNamespace(invoke=lambda prompt: answer))
    with pytest.raises(ValueError, match="query|expansion"):
        DefaultRAGClient(PreparedDB({}), llm, config).invoke("q")


@pytest.mark.parametrize(
    "metric,kind,expected",
    [
        ("EUCLIDEAN_DISTANCE", "squared_l2", "near"),
        ("MAX_INNER_PRODUCT", "inner_product", "far"),
    ],
)
def test_ready_faiss_index_supplies_score_semantics_without_reindexing(
    tmp_path, metric, kind, expected
):
    pytest.importorskip("faiss")
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_core.embeddings import Embeddings

    from ailib.agent.vector_db.faiss import FaissVectorDB

    class QueryOnlyEmbedding(Embeddings):
        def embed_query(self, text):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            raise AssertionError("Prepared documents must not be embedded again")

    embedding = QueryOnlyEmbedding()
    store = FAISS.from_embeddings(
        [("near", [1.0, 0.0]), ("far", [2.0, 0.0])],
        embedding,
        ids=["near", "far"],
        distance_strategy=getattr(DistanceStrategy, metric),
    )
    store.save_local(str(tmp_path / "vectordb"))
    db = FaissVectorDB(tmp_path, embedding=embedding)
    config = RAGConfig()
    config.similarity_score_threshold = 0.5 if kind == "squared_l2" else 1.5
    results = DefaultRAGClient(db, config=config).invoke("q", top_k=1, candidate_k=2)
    assert results[0].document.id == expected
    assert results[0].score_kind == kind
    assert db.vector_store.index.ntotal == 2


def test_adapter_does_not_inherit_unrelated_backend_score_semantics():
    db = PreparedDB({"q": [scored("a", 10)]})
    adapter = VectorDBRetriever(db, search=db.search)
    result = VectorRAGClient(db, retriever=adapter).invoke("q")[0]
    assert result.score_kind == "unknown"
    assert result.higher_is_better is None


def test_filter_runs_before_rerank_and_rerank_threshold_is_separate():
    db = PreparedDB({"q": [scored("a", 0.9), scored("b", 0.7), scored("c", 0.1)]})
    config = RAGConfig()
    config.similarity_score_threshold = 0.5
    config.postprocess = "rerank"
    seen = []

    def rerank(query, documents):
        seen.extend(doc.id for doc in documents)
        return [ScoredDocument(documents[1], 0.8), ScoredDocument(documents[0], 0.2)]

    llm = SimpleNamespace(reranker=SimpleNamespace(rerank=rerank))
    client = DefaultRAGClient(db, llm, config, rerank_score_threshold=0.5)
    assert [result.document.id for result in client.invoke("q")] == ["b"]
    assert seen == ["a", "b"]


def test_reranker_errors_propagate():
    config = RAGConfig()
    config.postprocess = "rerank"

    def rerank(query, documents):
        raise TimeoutError("reranker timeout")

    llm = SimpleNamespace(reranker=SimpleNamespace(rerank=rerank))
    with pytest.raises(TimeoutError, match="reranker timeout"):
        DefaultRAGClient(PreparedDB({"q": [scored("a", 1)]}), llm, config).invoke("q")


@pytest.mark.parametrize("query", [None, "", "  ", 10])
def test_invalid_question_is_rejected(query):
    with pytest.raises(ValueError, match="query"):
        VectorRAGClient(PreparedDB({})).invoke(query)


def test_missing_search_requires_explicit_adapter():
    with pytest.raises(ValueError, match="search"):
        VectorRAGClient(object())


def test_tied_rrf_scores_keep_first_seen_order_and_ignore_repeated_hits():
    db = PreparedDB({"q": [scored("a", 0.9), scored("a", 0.9)]})
    sparse = PreparedDB({"q": [scored("b", 10)]})
    results = HybridRAGClient(db, sparse.search).invoke("q", top_k=2)
    assert [result.document.id for result in results] == ["a", "b"]
    assert [result.score for result in results] == pytest.approx([0.5 / 61, 0.5 / 61])


def test_opensearch_filters_native_scores_without_calling_them_cosine():
    from ailib.agent.vector_db.opensearch import OpenSearchDB

    class SearchStore:
        def similarity_search_with_score(self, query, k):
            assert query == "q"
            return [
                (Document(id="a", page_content="a"), 8.0),
                (Document(id="b", page_content="b"), 2.0),
            ][:k]

    db = OpenSearchDB.__new__(OpenSearchDB)
    db.vector_store = SearchStore()
    config = RAGConfig()
    config.similarity_score_threshold = 5.0
    results = DefaultRAGClient(db, config=config).invoke("q", top_k=2)
    assert [result.document.id for result in results] == ["a"]
    assert results[0].score_kind == "opensearch_score"


@pytest.mark.parametrize("sparse_search", [None, False, "bm25"])
def test_hybrid_requires_prepared_sparse_search(sparse_search):
    with pytest.raises(ValueError, match="sparse"):
        HybridRAGClient(PreparedDB({}), sparse_search)


def test_multi_query_hybrid_fuses_within_each_path_before_combining_paths():
    db = PreparedDB(
        {
            "one": [scored("a", 0.9), scored("b", 0.8)],
            "two": [scored("b", 0.7), scored("c", 0.6)],
        }
    )
    sparse = PreparedDB({"original": [scored("c", 12), scored("a", 10)]})

    class MultiQueryHybrid(HybridRAGClient):
        def preprocess(self, query):
            return SearchInput(query, ("one", "two"), (query,))

    results = MultiQueryHybrid(db, sparse.search).invoke("original", top_k=2)
    assert [result.document.id for result in results] == ["a", "b"]
    assert [result.score for result in results] == pytest.approx([1 / 62, 0.5 / 61])
    assert [(item.query, item.rank) for item in results[1].source_scores] == [
        ("one", 2),
        ("two", 1),
    ]
