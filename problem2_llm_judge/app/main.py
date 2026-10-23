from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(
    title="LLM-as-Judge Evaluation Pipeline",
    description=(
        "A transparent baseline for evaluating generated answers. "
        "The current implementation uses rules, not an LLM."
    ),
    version="1.0.0",
)


class EvaluationRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    reference_answer: str = ""
    context: str = ""
    mode: Literal["pointwise"] = "pointwise"


class EvaluationResponse(BaseModel):
    question: str
    scores: dict[str, int]
    weighted_score: float
    feedback: list[str]
    judge_type: str


def word_set(text: str) -> set[str]:
    """Normalize text into a set of words."""
    import re

    return set(re.findall(r"\b\w+\b", text.lower()))


def score_answer(
    question: str,
    answer: str,
    reference_answer: str,
    context: str,
):
    """
    Calculate transparent lexical baseline scores.

    These are heuristic scores, not scores produced by an LLM.
    """
    answer_words = word_set(answer)
    question_words = word_set(question)

    if reference_answer.strip():
        reference_words = word_set(reference_answer)
        correctness = round(
            100
            * len(answer_words & reference_words)
            / max(len(reference_words), 1)
        )
    else:
        correctness = 0

    if context.strip():
        context_words = word_set(context)
        faithfulness = round(
            100
            * len(answer_words & context_words)
            / max(len(answer_words), 1)
        )
    else:
        faithfulness = 0

    relevance = round(
        100
        * len(answer_words & question_words)
        / max(len(question_words), 1)
    )

    completeness = min(100, len(answer.split()) * 5)
    instruction_following = 100 if answer.strip() else 0

    scores = {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "completeness": completeness,
        "relevance": relevance,
        "instruction_following": instruction_following,
    }

    weights = {
        "correctness": 0.30,
        "faithfulness": 0.25,
        "completeness": 0.20,
        "relevance": 0.15,
        "instruction_following": 0.10,
    }

    weighted_score = round(
        sum(scores[name] * weights[name] for name in scores),
        2,
    )

    feedback = []

    if not reference_answer.strip():
        feedback.append(
            "Correctness could not be assessed against a reference answer."
        )

    if not context.strip():
        feedback.append(
            "Faithfulness could not be assessed against supporting context."
        )

    if relevance < 40:
        feedback.append(
            "Review whether the answer directly addresses the question."
        )

    if completeness < 40:
        feedback.append(
            "Consider adding relevant detail if the question requires it."
        )

    if not feedback:
        feedback.append(
            "This is a heuristic baseline. Human review is recommended."
        )

    return scores, weighted_score, feedback


@app.get("/")
def root():
    return {
        "message": "LLM-as-Judge Evaluation Pipeline",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "judge_type": "rule_based_baseline",
        "is_actual_llm_judge": False,
    }


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest):
    scores, weighted_score, feedback = score_answer(
        question=request.question,
        answer=request.answer,
        reference_answer=request.reference_answer,
        context=request.context,
    )

    return EvaluationResponse(
        question=request.question,
        scores=scores,
        weighted_score=weighted_score,
        feedback=feedback,
        judge_type="rule_based_baseline_not_llm",
    )