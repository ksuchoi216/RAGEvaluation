# RAGClient Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 완성된 VectorDB를 이용하는 상속 가능한 RAG 검색 파이프라인을 제공한다.

**Architecture:** RAGClient가 단계 순서를 소유한다. VectorRAGClient와 HybridRAGClient는 준비된 검색 기능을 연결하고, DefaultRAGClient는 기존 LLM/config 기능을 제공한다. 검색 입력과 결과는 작은 데이터 클래스로 명시한다.

**Tech Stack:** Python 3.10+, dataclasses, ABC, LangChain Document, pytest.

**Spec:** ../specs/2026-09-20-rag-client-design.md

## Global Constraints

- 인덱스 구축은 VectorDB 책임이며 클라이언트에서 수행하지 않는다.
- 요청 상태를 인스턴스에 저장하지 않는다.
- 기존 공유 ScoredDocument와 legacy rag.py는 유지한다.
- 외부 서비스 없이 회귀 테스트한다. 관계없는 작업은 변경하지 않는다.

## Review Focus

- 음수, bool, 실수인 후보 수는 검색 전에 거부한다.
- 원래 질문과 HyDE 검색어를 혼동하지 않는다.
- 서로 다른 문서의 동일 chunk_index를 중복으로 취급하지 않는다.
- RRF 가중치의 NaN/무한대와 0 가중치 검색의 영향을 차단한다.
- rerank 실패 및 알 수 없는 점수에 대한 threshold 설정을 빈 결과로 숨기지 않는다.

### Task 1: 검색 계약과 공통 파이프라인

Files: ailib/agent/rag_client/{types.py,retrievers.py,rag_client.py,__init__.py}, ailib/tests/test_rag_client.py.

Interfaces: SearchInput(original_query, dense_queries, sparse_queries), RetrievalResult(document, score, score_kind, higher_is_better, rerank_score, source_scores). RAGClient.invoke(query, top_k=5, *, candidate_k=None).

- [x] 가짜 검색 저장소를 이용해 단계 override, 빈 결과 formatting, 후보 수 검증, 예외 전파 테스트를 작성하고 실행한다.
- [x] 공통 파이프라인과 단일 dense 검색 연결을 구현한다.
- [x] `python -m pytest -q ailib/tests/test_rag_client.py`로 검증한다.

```python
results = VectorRAGClient(vector_db).invoke("question", top_k=5, candidate_k=20)
assert len(results) == 5
assert vector_db.calls == [("question", 20)]
```

### Task 2: 복수 질문 및 hybrid 결합

Files: ailib/agent/rag_client/retrievers.py, ailib/tests/test_rag_client.py.

Interfaces: VectorDBRetriever accepts a prepared search callable returning ScoredDocument. HybridRAGClient takes vector_db and sparse_search; optional dense_search binds backend-specific arguments. Both return RetrievalResult lists.

- [x] HyDE의 dense/sparse 입력 분리, RRF 순위, stable chunk ID 중복 제거, 결합 점수 보존, 잘못된 가중치 테스트를 먼저 실행한다.
- [x] 경로 내부 복수 쿼리 결합 후 경로 간 weighted RRF를 구현한다. 동일 순위는 입력 순서를 유지한다.
- [x] 독립적으로 계산한 기대값으로 검증한다.

```python
assert fused[0].document.id == "shared"
assert fused[0].score == pytest.approx(0.5 / 62 + 0.5 / 61)
```

### Task 3: 기존 기능 이전과 backend 점수 계약

Files: ailib/agent/rag_client/rag_client.py, ailib/agent/vector_db/{vector_db.py,faiss.py,opensearch.py}, ailib/tests/test_rag_client.py, ailib/agent/rag_client/README.md.

Interfaces: DefaultRAGClient(vector_db, llm_client, config=None, *, retriever=None); prepared backend exposes search_score_kind and search_higher_is_better. Unknown semantics remain explicit.

- [x] expansion, 원래 질문 기반 rerank, 검색 전후 점수 보존, threshold 방향, 빈 입력, malformed 모델 출력, backend 점수 계약 테스트를 먼저 실행한다.
- [x] 기존 기능을 concrete class로 옮기고 backend score metadata를 제공한다. 준비된 실제 FAISS로 integration 테스트할 수 있으면 수행한다.
- [x] 사용 예시와 migration 안내를 작성한다.
- [x] 타깃 pytest, 상위 프로젝트 pytest, ailib/tests 및 변경 파일 lint를 실행한다.
- [x] diff를 검토하고 결과와 한계를 기록한다.

## Execution record

- User instruction: 실행. 현재 세션에서 직접 구현하며 추가 승인 단계 없이 완료한다.
- Ruling: 기존 작업 공간에서 제한된 파일만 수정한다. ailib/develop의 작업 시작 상태는 clean이다. 상위 저장소의 기존 수정·staging은 유지한다.
- Pre-flight: Task 2 uses Task 1 SearchInput/Result; Task 3 uses Task 1 pipeline and Task 2 retrievers. No conflicting interfaces.

- Task 1: complete — 공통 실행 흐름, override 및 빈 결과/입력 검증.
- Task 2: complete — 복수 쿼리 및 hybrid RRF, 청크 식별, 원점수 보존.
- Task 3: complete — DefaultRAGClient 이전, FAISS/OpenSearch 점수 의미, 사용 문서.
- Review: 독립 리뷰가 발견한 sparse_search=None의 dense fallback을 실패 테스트로 재현하고 수정했다. 복수 쿼리+hybrid 2단계 결합 테스트도 추가했다.
- Verification: 임시 환경 /tmp/rag-client-verify에서 ailib/tests 52 passed; 기존 Python 환경의 상위 프로젝트 pytest 39 passed. 변경 RAG 모듈 및 테스트 ruff 통과, compileall 및 git diff --check 통과.
- Environment: 기존 Python에서 numpy/faiss import가 AttributeError로 실패해 별도 임시 venv를 사용했다. 사용자 환경/프로젝트 의존성은 수정하지 않았다. 임시 환경에는 langchain-community deprecation warning 1건이 남는다.
- Integration: 작업 도중 외부에서 ailib의 중간 구현이 ff84485로 커밋되었다. 해당 커밋을 유지하고 후속 변경은 현재 작업 공간에 남겼다.
