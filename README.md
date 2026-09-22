# RAGEvaluation

`notebooks/Auto_Evaluate_v2.ipynb`의 답변 평가를 재사용 가능한
`src/rag_evaluation` 패키지로 정리했습니다. RAG가 생성한 답변을 기준 답변과
비교하며, 검색기나 답변 생성기는 각 프로젝트에서 별도로 구현합니다.

## 설치

Python 3.10 이상에서 저장소 루트 기준으로 실행합니다.

```bash
pip install -e .
```

예제 파일을 복사하고 사용할 제공자의 API 키를 입력합니다.

```bash
cp .env.example .env
```

- `ANTHROPIC_API_KEY`: 기본 Claude 평가 모델을 사용할 때 필요합니다.
- `OPENAI_API_KEY`: OpenAI 평가 모델을 사용할 때 필요합니다.

`.env`에는 API 키만 설정하면 됩니다. 모델 이름이나 `mlflow_url`은 필요하지
않습니다. `.env`는 Git에서 제외되며 `.env.example`만 공유합니다.

## 사용 예시

저장소 루트에서 아래 코드를 실행합니다. `RAGEvaluator()`는 현재 작업 디렉터리의
`.env`를 읽습니다. 이미 설정된 환경 변수는 덮어쓰지 않습니다.

```python
from rag_evaluation import RAGEvaluator

evaluator = RAGEvaluator(model_provider="claude")

result = evaluator.evaluate(
    questions=["대한민국의 수도는 어디인가요?"],
    generated_answers=["서울입니다."],
    reference_answers=["대한민국의 수도는 서울입니다."],
)

print(result.verdicts)              # 입력별 최종 판정: O 또는 X
print(result.tonic_similarity)      # 0~5점
print(result.mlflow_similarity)     # 1~5점
print(result.mlflow_correctness)    # 1~5점
print(result.allganize_correctness) # 0 또는 1
```

다른 디렉터리나 노트북에서는 `RAGEvaluator(env_file="../.env")`처럼
파일 경로를 지정할 수 있습니다. 파일 없이 환경 변수만 설정해도 동작합니다.

`model_provider`는 `"claude"`(기본값), `"openai"`, `"local"`을 받습니다.
Claude는 `claude-haiku-4-5`, OpenAI는 `gpt-4.1-nano`를 사용합니다.
선택한 제공자는 네 가지 평가 지표 모두에 적용됩니다. `"local"`은 아직
지원하지 않아 `NotImplementedError`를 발생시킵니다. 임베딩은 사용하지 않습니다.

```python
evaluator = RAGEvaluator()  # Claude 기본값
evaluator = RAGEvaluator(model_provider="openai")
```

이 프로젝트는 `from rag_evaluation import RAGEvaluator`로 가져오는 패키지입니다.
검색과 답변 생성은 사용하는 프로젝트에서 처리합니다.

기존 노트북과 동일하게 유사성/MLflow 점수는 4점 이상, Allganize 점수는
1점이면 정답으로 투표합니다. 네 표 중 세 표 이상이 정답이어야 `O`이며,
2:2 동점은 `X`입니다. 기존 `llm_evaluate(...)`의 목록 반환값에 해당하는 값은
`result.verdicts`입니다.

입력은 길이가 같은 문자열 목록이어야 합니다. 빈 목록 세 개를 전달하면
외부 호출 없이 빈 결과를 반환합니다. 입력 순서와 결과 순서는 같습니다.

## 다른 평가 모델 연결

프롬프트 문자열을 받아 점수 문자열을 반환하는 함수를 전달할 수 있습니다.
LangChain 모델을 이미 사용하고 있다면 다음과 같이 연결합니다.

```python
from langchain_core.output_parsers import StrOutputParser
from rag_evaluation import RAGEvaluator

# similarity_llm, correctness_llm은 프로젝트에서 설정한 채팅 모델입니다.
similarity_chain = similarity_llm | StrOutputParser()
correctness_chain = correctness_llm | StrOutputParser()

evaluator = RAGEvaluator(
    similarity_judge=similarity_chain.invoke,
    correctness_judge=correctness_chain.invoke,
    mlflow_model="openai:/YOUR_MODEL_NAME",
    on_error="record",
)
```

MLflow 호출도 교체하려면 생성자에 `mlflow_evaluator=평가함수`를 전달합니다.
이 함수는 `(questions, generated_answers, reference_answers, *, model, on_error)`를
받아 `(유사성_점수_목록, 정확성_점수_목록)`을 반환해야 합니다. 이를 통해
실제 외부 서비스를 호출하지 않고 전체 평가 흐름을 테스트할 수 있습니다.

`on_error="raise"`가 기본값입니다. 직접 호출하는 평가 모델의 오류는
`EvaluationError`로 감싸며, 메트릭 이름과 0부터 시작하는 행 번호를 제공합니다.
원래 예외는 `__cause__`에 보존합니다. MLflow의 비정상 점수는 `ValueError`를
발생시킵니다.

`on_error="record"`는 개별 평가 실패를 로그와 `-1`로 기록하고 계속합니다.
`-1`은 노트북처럼 오답 표로 취급하므로, 평가 실패와 실제 오답을 구분하려면
개별 점수도 확인해야 합니다. MLflow 실행 전체 실패나 결과 테이블 구조 오류는
이 옵션에서도 숨기지 않습니다. 기존 MLflow run 안에서는 nested run을 만듭니다.

## 구성과 검증

- `evaluator.py`: 환경 설정, 평가 모델 구성, 평가 실행
- `results.py`: 결과 점수 검증과 다수결 판정
- `validation.py`: 평가 진입점에서 공유하는 입력 및 오류 정책 검증
- `mlflow_evaluation.py`: MLflow 평가 및 점수 검증
- `prompts.py`: 원본 노트북의 두 평가 프롬프트

결과 객체는 점수 목록 길이와 각 메트릭의 범위를 검증합니다. `NaN`, 무한대,
문자열, bool 점수는 거부하며, `-1`은 실패를 기록하는 값으로 허용합니다.

MLflow는 원본 지표를 유지하기 위해 기존 `metrics.genai` 메트릭과
`models.evaluate`를 사용합니다. 최신 `mlflow.genai.evaluate`의 scorer API로
전환한 구현은 아닙니다.
참고: [MLflow 평가 API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html).

```bash
pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src tests
```

테스트는 외부 LLM과 MLflow 호출을 대체하므로 API 키나 네트워크 없이 실행할 수
있습니다. 실제 모델 호출과 MLflow 저장소 연동은 별도로 실행해 확인해야 합니다.

## RAG 검색 클라이언트

검색 파이프라인은 `src/rag_client`, 답변 평가는 `src/rag_evaluation`에서
개발합니다. 두 패키지는 `pip install -e .`로 함께 설치됩니다.

```python
from rag_client import VectorRAGClient

# vector_db는 인덱스 구축을 마친 VectorDB 인스턴스입니다.
client = VectorRAGClient(vector_db)
results = client.invoke("검색할 질문", top_k=5, candidate_k=20)
```

`rag_client`는 설정, 공통 타입, LLM 및 VectorDB 구현을 `ailib`에서 사용합니다.
저장소의 `ailib` 하위 모듈과 해당 의존성이 필요하며, 별도 프로젝트에서는
`ailib`도 Python에서 가져올 수 있도록 설정해야 합니다.

기존 `ailib.agent.rag_client` import는 `rag_client`로 변경하세요.
자세한 내용은 [RAG 클라이언트 사용법](src/rag_client/README.md)을 참고하세요.
