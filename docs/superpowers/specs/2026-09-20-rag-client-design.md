# RAGClient 단계별 추상화

## 목적과 합의된 경계

완성된 VectorDB를 주입받아 검색하는 RAGClient를 공통 파이프라인과
구체 구현으로 분리한다. 가독성을 우선하고, 불필요하게 메서드나 클래스 계층을
늘리지 않는다. 사용자는 필요한 단계만 상속하여 override할 수 있어야 한다.

문서 적재, 청킹, 문서 임베딩, 인덱스 구축·관리는 VectorDB의 책임이다.
RAGClient는 준비된 저장소를 사용하며 인덱스를 만들거나 다시 구축하지 않는다.
검색 시 질문 임베딩은 VectorDB 또는 준비된 retriever의 검색 기능에 맡긴다.
답변 생성과 평가 실행기는 이번 변경 범위에 포함하지 않는다.

## 클래스 구조

```text
RAGClient                     VectorDB 보관과 공통 실행 흐름
├─ VectorRAGClient            dense retriever 구성
├─ HybridRAGClient            준비된 dense/sparse 검색 기능의 결합 구성
└─ 사용자 정의 클래스         필요한 단계 override
```

사용자 정의 클래스는 RAGClient뿐 아니라 VectorRAGClient 또는
HybridRAGClient를 상속할 수 있다. CustomRAGClient는 사용 예시 이름이며,
라이브러리에 빈 클래스로 추가하지 않는다.

기반 클래스는 LLMClient를 필수 의존성으로 갖지 않는다. 전처리 모델이나
reranker는 필요한 구체 구현에만 주입한다. 동일 알고리즘에서 모델만 바뀌는
경우에는 새 하위 클래스를 만들지 않고 주입할 객체를 바꾼다.

## 실행 계약

초기화 시 `create_retriever()`를 한 번 호출한다. 하위 클래스는 retriever
구성에 필요한 의존성을 준비한 뒤 기반 클래스의 초기화를 호출한다.
기반 VectorDB에는 공통 search 계약이 없으므로 구체 retriever가 지원하는
검색 기능에 맞추어 연결한다. 필요하면 ailib/agent/vector_db의 공통 계약과
구체 backend를 수정할 수 있다. 다만 dense 전용 backend에 sparse 검색을
강제하지 않고, 지원하는 검색 기능과 필요한 인자를 명확히 드러낸다.

| 메서드 | 입력 | 출력과 기본 동작 |
|---|---|---|
| create_retriever | 준비된 VectorDB | retriever 반환, 추상 메서드 |
| preprocess | 원래 질문 | SearchInput 반환, 기본은 원래 질문 사용 |
| retrieve | SearchInput, candidate_k | 검색 후보 반환, retriever에 위임 |
| postprocess | 원래 질문, 검색 후보, top_k | 후처리 결과, 기본은 상위 top_k개 선택 |
| format_output | 후처리 결과 | 기본은 구조화된 결과 그대로 반환 |
| invoke | 질문, top_k, candidate_k | 위 네 단계를 순서대로 실행하고 최종 출력 반환 |

`invoke(query, top_k=5, *, candidate_k=None)` 형태를 사용한다.
candidate_k를 생략하면 top_k와 같게 한다. rerank를 사용할 때는 명시적으로
`candidate_k=20, top_k=5`처럼 지정한다. 두 값은 양의 정수이며
candidate_k는 top_k 이상이어야 한다. 잘못된 값은 검색 전에 ValueError로 알린다.

빈 후보도 postprocess와 format_output으로 전달한다. 기본 구현은 빈 목록을
반환하며, reranker 구현은 빈 입력에 외부 호출을 하지 않는다.
후처리 구현은 top_k 이하를 반환하는 계약을 지킨다. format_output은 검색,
재평가, 개수 제한을 수행하지 않고 표현만 변환한다.

## 검색 입력과 결과

SearchInput은 원래 질문, dense 검색어 목록, sparse 검색어 목록을 명시하는
작은 데이터 클래스다. 기본 전처리는 두 목록에 원래 질문을 넣는다.
HyDE는 dense 목록에 가상 답안을, sparse 목록에 원래 질문을 넣을 수 있다.
Multi-query는 여러 검색어를 반환한다. rerank에는 원래 질문을 전달한다.
요청별 검색어나 중간 결과를 클라이언트 인스턴스에 저장하지 않는다.

검색 결과는 Document와 순위 판단에 사용한 점수의 의미를 함께 보존한다.
기존 공용 ScoredDocument를 광범위하게 바꾸는 대신 RAG 계층의 결과 타입으로
검색 점수, 점수 종류, 정렬 방향, 선택적인 rerank 점수를 구분한다.
Hybrid 결과는 결합 점수와 구성 검색 결과의 점수·순위를 구분한다.
거리, cosine, BM25, RRF, rerank 점수를 동일한 범위로 간주하지 않는다.

공통 파이프라인은 similarity_score_threshold 하나를 모든 점수에 적용하지
않는다. 필터링이 필요한 구현은 어떤 단계의 어떤 점수를 비교하는지 명시한다.
점수 의미를 알 수 없는 backend의 반환값을 임의로 정규화하지 않는다.

문서 ID, 청크 ID, 페이지, 출처 메타데이터는 결과 변환 과정에서 보존한다.
검색 결과 결합은 안정적인 청크 ID를 사용해 중복을 제거한다. 같은 문서의
서로 다른 청크를 문서 ID만으로 합치지 않는다. 결합이 필요한 retriever는
청크 식별 규칙을 명시적으로 제공받는다. 필요한 경우 VectorDB의 검색 결과나
문서 적재 시 메타데이터 부여를 개선할 수 있다. 기존 인덱스는 backend가
보관한 식별자를 사용하거나 명시적인 식별 함수를 제공한다. RAGClient가
식별 정보를 만들기 위해 인덱스를 재구축하지 않는다.

## Dense 및 hybrid 연결

VectorRAGClient는 준비된 VectorDB의 dense 검색을 연결한다. 단일 쿼리는
backend 순서를 유지하고, 여러 쿼리는 retriever 내부에서 RRF로 결합한다.
HybridRAGClient는 같은 VectorDB를 보관하면서 준비된 dense/sparse 검색
adapter를 받아 두 경로를 구성한다. 해당 기능이 없는 VectorDB에서 sparse
인덱스를 자동으로 만들거나 dense 검색으로 몰래 대체하지 않는다.

adapter는 SearchInput의 해당 검색어 목록을 검색하고, 경로 내부의 복수 쿼리
결과를 먼저 결합한다. Hybrid retriever는 그 두 순위 목록을 가중 RRF로 결합하고
candidate_k개를 반환한다. 경로별 후보 수 역시 candidate_k를 사용한다.
RRF 상수 기본값은 60, dense/sparse 가중치는 각각 0.5로 두되 생성자에서
변경할 수 있다. 가중치는 음수가 아니고 합이 양수여야 하며 합으로 정규화한다.
동점은 입력 경로와 기존 순위의 안정적인 순서로 처리한다.

backend 고유 hybrid API가 별도 검색어를 지원하지 않으면 그 제한을 숨기지
않는다. 이번 공통 hybrid 구현은 두 검색 경로를 각각 호출할 수 있는 adapter를
사용하며, backend 고유 API 지원은 사용자 정의 retriever로 확장할 수 있다.

## 기존 동작과 변경 범위

현재 RAGClient의 query expansion, rerank, 필터링은 공통 기반 클래스에서
분리한다. 기존 설정 기반 사용은 DefaultRAGClient라는 구체 구현으로 옮긴다.
이 클래스는 선택한 준비된 retriever와 기존 LLMClient/RAGConfig를 받아
기존 기능을 제공하며, query expansion과 rerank는 각 단계에서 수행한다.
검색 입력 변경 후에도 rerank는 원래 질문을 사용하는 것으로 동작을 명시한다.

기존 RAGClient 직접 생성은 추상 클래스 도입으로 호환성이 바뀐다.
사용 예시에 DefaultRAGClient 또는 VectorRAGClient로의 이전 방법을 적는다.
기존 임계값 설정은 검색 점수 필터로만 해석하고, 점수 종류·정렬 방향이
명시된 경우 검색 후 rerank 전에 적용한다. 알 수 없는 점수에 임계값을
설정하면 명확한 오류를 반환한다. rerank 점수 필터는 별도로 명시한다.

rag.py의 SimpleRAG 및 MetadataHeaderRAG는 이번 변경에서 유지한다.
사용자가 ailib/agent/vector_db 수정도 허용했다. 검색 기능 연결, 점수 의미,
청크 식별자 제공에 필요한 변경은 범위에 포함하며, 저장소 전체를 다시
설계하거나 관계없는 CRUD 기능을 변경하지 않는다. VectorDB를 수정하면
해당 backend의 기존 호출 방식과 반환값에 대한 회귀 검증을 포함한다.
이미 작업 중인 상위 저장소의 평가 코드, README, 의존성 변경은 섞지 않는다.
ailib는 별도 Git 저장소이므로 구현과 테스트의 저장소 경계를 확인한다.

## 오류 및 검증

지원하지 않는 검색 기능이나 잘못된 설정은 명확한 예외로 알린다.
외부 검색·모델 호출 실패를 정상적인 빈 결과로 바꾸지 않는다.
잘못된 전처리 모델 출력은 오류로 알리며, 원래 질문으로의 자동 복귀는
사용자가 별도로 구현한 정책에서만 수행한다.

외부 DB나 모델 없이 가짜 VectorDB, retriever, 모델로 다음을 검증한다.

- retriever가 초기화 시 한 번 구성되고 인덱스 구축이 발생하지 않는지
- 단계 호출 순서와 각 단계 override, 원래 질문과 검색 입력의 구분
- 20개 후보 검색 후 5개 반환 및 rerank 생략 시 개수 제한
- 빈 결과에서도 출력 변환이 실행되는지
- HyDE의 경로별 입력, 복수 쿼리 및 hybrid의 RRF 결합·중복 제거
- 서로 다른 점수 종류의 보존과 명시적인 필터링
- 잘못된 개수, 가중치, 누락된 검색 기능, 외부 호출 실패 처리
- 기존 query expansion 및 rerank 기능의 구체 클래스 이전

## 문서 검토 근거와 비목표

ref/RAG-Evaluation/docs의 1~6 및 4-1, 4-2, 5_generation_e2e_axis를 검토했다.
1~3은 구축 책임 및 출처·임베딩 설정 보존, 4 계열은 복수 입력·hybrid·후처리
후보 수, 5~6은 구성 교체와 검색/생성 평가 분리의 근거다.
문서의 실험 우승 모델이나 조합을 공통 기본값으로 고정하지 않는다.

이번 변경은 모든 전처리 알고리즘 구현, 반복 검색, 평가 실행기, 캐시,
멀티모달 파이프라인, 비동기 API, 전면적인 인덱스 구조 개편을 포함하지 않는다.
이러한 기능은 이번 단계 계약을 이용해 후속 작업에서 확장한다.
