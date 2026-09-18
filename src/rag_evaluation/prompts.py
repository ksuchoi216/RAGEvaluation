"""노트북에서 사용한 답변 유사성 및 정확성 평가 프롬프트."""

TONIC_SIMILARITY_PROMPT = (
    "Considering the reference answer and the new answer to the following question, "
    "on a scale of 0 to 5, where 5 means the same and 0 means not at all similar, "
    "how similar in meaning is the new answer to the reference answer? Respond with just "
    "a number and no additional text.\nQUESTION: {question}\n"
    "REFERENCE ANSWER: {reference_answer}\nNEW ANSWER: {llm_answer}\n"
)

ALLGANIZE_CORRECTNESS_PROMPT = '''
question = """
{question}
"""

target_answer = """
{reference_answer}
"""

generated_answer = """
{llm_answer}
"""

Check if target_answer and generated_answer match by referring to question.
If target_answer and generated_answer match 1, answer 0 if they do not match.
Only 1 or 0 must be created.
'''
