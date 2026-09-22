# RAGClient

완성된 VectorDB를 받아 검색 파이프라인을 실행합니다. 인덱스 구축과 문서 임베딩은
VectorDB에서 미리 수행합니다.

```text
RAGClient
├─ VectorRAGClient       dense 검색
├─ HybridRAGClient       dense + sparse 검색, weighted RRF 결합
└─ 사용자 정의 클래스   필요한 단계 override
```

기존 설정 기반 구현은 `DefaultRAGClient(VectorRAGClient)`에 있습니다.

## 기본 사용

```python
from rag_client import VectorRAGClient

# vector_db는 이미 인덱스를 준비한 VectorDB 인스턴스입니다.
client = VectorRAGClient(vector_db)
results = client.invoke("검색할 질문", top_k=5, candidate_k=20)

for result in results:
    print(result.document.page_content, result.score, result.score_kind)
```

`candidate_k`는 검색 후보 수, `top_k`는 최종 최대 반환 수입니다.
candidate_k를 생략하면 top_k와 같습니다.

## 단계 확장

`__init__`에서 `create_retriever(vector_db)`를 한 번 실행합니다.
`invoke()`는 다음 순서를 유지합니다.

```text
preprocess(query)
  → retrieve(search_input, candidate_k=...)
  → postprocess(original_query, results, top_k=...)
  → format_output(results)
```

`create_retriever()`만 추상 메서드입니다. 나머지 단계는 기본 구현이 있어
필요한 것만 override하면 됩니다. 후처리는 top_k 이하의 결과를 반환해야 합니다.
format_output은 표현만 바꾸며 빈 결과에서도 실행됩니다.

```python
from rag_client import SearchInput, VectorRAGClient


class CustomRAGClient(VectorRAGClient):
    def __init__(self, vector_db, query_model):
        self.query_model = query_model
        super().__init__(vector_db)

    def preprocess(self, query: str) -> SearchInput:
        expanded = self.query_model.invoke(f"검색어를 확장하세요: {query}")
        if not isinstance(expanded, str) or not expanded.strip():
            raise ValueError("검색어 확장 결과가 비어 있습니다")
        return SearchInput(query, (query, expanded.strip()), (query,))

    def format_output(self, results):
        return [result.document.page_content for result in results]
```

원래 질문은 후처리에 그대로 전달합니다. SearchInput의 dense_queries와
sparse_queries는 검색 전용입니다. 같은 쿼리는 한 번만 검색하고, 여러 쿼리의
결과는 동일 가중치 RRF로 결합합니다. 요청별 상태는 인스턴스에 저장하지 않습니다.

## Hybrid 및 HyDE

```python
from rag_client import HybridRAGClient, SearchInput


class HyDERAGClient(HybridRAGClient):
    def __init__(self, vector_db, sparse_search, query_model):
        self.query_model = query_model
        super().__init__(vector_db, sparse_search, dense_weight=0.3, sparse_weight=0.7)

    def preprocess(self, query: str) -> SearchInput:
        hypothesis = self.query_model.invoke(f"검색용 가상 답안을 작성하세요: {query}")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("가상 답안이 비어 있습니다")
        return SearchInput(query, (hypothesis.strip(),), (query,))
```

`sparse_search(query, *, top_k)`는 준비된 sparse 인덱스에서 관련도 순으로 정렬된
`list[ScoredDocument]`를 반환하는 함수입니다. RAGClient가 sparse 인덱스를
만들지는 않습니다. dense 검색 인자를 바인딩해야 하면 `dense_search`를 전달합니다.
가중치 0인 경로는 호출하지 않습니다.

Hybrid는 각 경로의 복수 쿼리를 먼저 결합하고, 두 경로를 다시 weighted RRF로
결합합니다. 가중치 합은 1로 정규화하며 `rrf_k=60`을 기본으로 사용합니다.
동점은 먼저 발견된 순서를 유지합니다.

결합하려면 각 청크에 안정적인 `Document.id` 또는 `(doc_id, chunk_index)`
메타데이터가 있어야 합니다. backend 간 ID가 다르면 `document_key` 콜백으로
동일 청크를 식별하는 공통 키를 제공하세요. 문서 내용만으로 중복을 제거하지 않습니다.

## 점수와 backend adapter

`RetrievalResult.score`는 검색 점수이고 `rerank_score`는 별도 필드입니다.
`source_scores`에는 검색 경로·쿼리별 원점수와 순위가 남습니다. Document와
출처 메타데이터는 유지됩니다.

기본 adapter는 DB의 `search_score_kind`, `search_higher_is_better`를 읽습니다.
명시적으로 search 함수를 전달하면 backend의 점수 의미를 상속하지 않습니다.
함수가 다른 검색 방식을 사용할 수 있기 때문입니다.

```python
from rag_client import HybridRetriever, VectorDBRetriever, VectorRAGClient

dense = VectorDBRetriever(vector_db)
sparse = VectorDBRetriever(
    vector_db,
    search=sparse_search,
    score_kind="bm25",
    higher_is_better=True,
)
retriever = HybridRetriever(dense, sparse, dense_weight=0.3, sparse_weight=0.7)
client = VectorRAGClient(vector_db, retriever=retriever)
```

이처럼 backend별 검색 인자를 함수로 바인딩하고 점수 의미를 명시할 수 있습니다.
의미를 지정하지 않은 원점수는 unknown으로 보존합니다. RRF는 원점수의 크기를
비교하지 않고 각 검색 결과의 순위를 사용합니다.

## 기존 RAGClient에서 이전

`RAGClient`는 이제 추상 클래스이므로 직접 생성하지 않습니다.

```python
from ailib.agent.llm_client.config import RAGConfig
from rag_client import DefaultRAGClient

config = RAGConfig()
config.preprocess = "query_expansion"
config.postprocess = "rerank"
client = DefaultRAGClient(vector_db, llm_client, config)
results = client.invoke("검색할 질문", candidate_k=20, top_k=5)
```

- 기존 LLMClient와 RAGConfig를 재사용할 수 있습니다.
- RAGConfig의 normal 외 검색 설정은 준비된 `retriever=`를 명시해야 합니다.
- query expansion 결과는 검색에 사용하고 rerank에는 원래 질문을 사용합니다.
- 반환값은 RetrievalResult입니다. document와 score 접근은 유지되지만,
  rerank 점수는 `rerank_score`에서 읽습니다.
- similarity_score_threshold는 양수일 때 검색 점수에 적용하고, 그 후 rerank합니다.
  거리 점수는 임계값 이하, 높은 값이 좋은 점수는 임계값 이상을 유지합니다.
  의미를 알 수 없는 점수에 필터를 적용하면 ValueError가 발생합니다.
- rerank 필터는 별도 생성자 인자 `rerank_score_threshold`로 설정합니다.
- 검색·모델 호출 실패는 전파됩니다. 잘못된 expansion 결과를 원래 질문으로
  조용히 대체하지 않습니다.
- 기존 rag.py의 SimpleRAG와 MetadataHeaderRAG는 변경하지 않습니다.

## 검증

프로젝트 루트에서 `python -m pytest -q tests/test_rag_client.py`를 실행합니다.
FAISS 통합 테스트는 로컬 인덱스와 고정 벡터만 사용하며 외부 모델을 호출하지 않습니다.
