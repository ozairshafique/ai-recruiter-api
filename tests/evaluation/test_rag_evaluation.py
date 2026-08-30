
from deepeval import evaluate
import sys
import pytest
import time
import os
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
from deepeval.test_case import LLMTestCase
from langchain_groq import ChatGroq
from deepeval.models import DeepEvalBaseLLM
from datasets import Dataset

# ── Custom Groq LLM for DeepEval ──────

class GroqEvalLLM(DeepEvalBaseLLM):
    """ Custom LLM wrapper for Groq models in DeepEval """
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(override=True)
        self.model = ChatGroq(
            api_key=os.environ["GROQ_API_KEY"],
            model="openai/gpt-oss-120b",
            temperature=0.0,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return "openai/gpt-oss-120b"


# ── Fixtures for Testing ───────────────
@pytest.fixture(scope="module")
def groq_model():
    """ Fixture to initialize the Groq LLM for testing """
    return GroqEvalLLM()

@pytest.fixture
def test_cases():
    """Sample test cases for evaluation."""
    return [
        LLMTestCase(
            input="What are the candidate's skills?",
            actual_output="The candidate has experience with Python, FastAPI, LangChain, Docker, AWS and MongoDB.",
            expected_output="Python, FastAPI, LangChain, Docker, AWS, MongoDB",
            retrieval_context=[
                "Technical Skills: Python, FastAPI, LangChain, Docker, AWS, MongoDB"
    ],
        ),
        LLMTestCase(
            input="What is the candidate's experience?",
            actual_output="The candidate has 3 years of software engineering experience and worked as an AI Engineer Intern at Asclepyus SRL.",
            expected_output="3 years software engineering experience and AI Engineer Intern at Asclepyus SRL",
            retrieval_context=[
                "AI Engineer Intern at Asclepyus SRL",
                "3 years of software engineering experience"
    ]
),
        LLMTestCase(
            input="What is the candidate's education?",
            actual_output="The candidate holds an MSc in Computer Science specializing in Artificial Intelligence from the University of Bari.",
            expected_output="MSc Computer Science specializing in Artificial Intelligence from University of Bari",retrieval_context=[
                "MSc Computer Science specializing in Artificial Intelligence",
                "University of Bari Aldo Moro"
    ]
),
    ]

# ── individual Metrics Tests ─────────
def test_answer_relevancy(groq_model, test_cases):
    """ Testa the answer relevancy metric with the sample test cases """
    results = AnswerRelevancyMetric(
        threshold=0.7,
        model=groq_model,
    )
    for tests in test_cases:
        results.measure(tests)
        time.sleep(2)
        print(f"Answer Relevancy Score for '{tests.input}': {results.score}")
        assert results.score >= 0.7, f"Too low: {results.score}"


def test_faithfulness(groq_model, test_cases):
    """ Test the faithfulness metric with the sample test cases """
    results = FaithfulnessMetric(
        threshold=0.7,
        model=groq_model,
    )
    for tests in test_cases:
        results.measure(tests)
        time.sleep(2)  # To avoid hitting rate limits
        print(f"Faithfulness Score for '{tests.input}': {results.score}")
        assert results.score >= 0.7, f"Too low: {results.score}"

def test_contextual_recall(groq_model, test_cases):
    """ Test the contextual recall metric with the sample test cases """
    results = ContextualRecallMetric(
        threshold=0.7,
        model=groq_model,
    )
    for tests in test_cases:
        results.measure(tests)
        time.sleep(2)
        print(f"Contextual Recall Score for '{tests.input}': {results.score}")
        assert results.score >= 0.7, f"Too low: {results.score}"

def test_contextual_precision(groq_model, test_cases):
    """ Test the contextual precision metric with the sample test cases """
    results = ContextualPrecisionMetric(
        threshold=0.7,
        model=groq_model,
    )
    for tests in test_cases:
        results.measure(tests)
        time.sleep(2)
        print(f"Contextual Precision Score for '{tests.input}': {results.score}")
        assert results.score >= 0.7, f"Too low: {results.score}"

# ── Full Evaluation Test ─────────────
def test_full_evaluation(groq_model, test_cases):
    """Full RAG evaluation with all metrics."""
    metrics = [
        AnswerRelevancyMetric(    threshold=0.7, model=groq_model),
        FaithfulnessMetric(       threshold=0.7, model=groq_model),
        ContextualRecallMetric(   threshold=0.7, model=groq_model),
        ContextualPrecisionMetric(threshold=0.7, model=groq_model),
    ]

    print(f"\n{'='*50}")
    print(f"  DeepEval RAG Evaluation Results")
    print(f"{'='*50}")

    for test in test_cases:
        for metric in metrics:
            metric.measure(test)
            time.sleep(2)
            status = "✅" if metric.score >= 0.7 else "❌"
            print(
                f"  {status} "
                f"{metric.__class__.__name__:35}: "
                f"{metric.score:.4f}"
            )
            assert metric.score >= 0.7

    print(f"\n{'='*50}")
    print(f"  All tests passed! ✅")
    print(f"{'='*50}")
