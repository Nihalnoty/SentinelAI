"""Phase 5.4: Factual accuracy and hallucination evaluation."""

import sys
import json
import os
import re
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import chunk_documents
from src.rag.ingest import load_documents
from src.rag.llm_adapter import generate_response
from src.rag.pipeline import answer_question


load_dotenv(PROJECT_ROOT / ".env")

KNOWLEDGE_BASE = PROJECT_ROOT / "data" / "knowledge_base"

RESULTS_DIRECTORY = (
    PROJECT_ROOT / "evaluation" / "results"
)

RESULTS_FILE = (
    RESULTS_DIRECTORY / "factual_accuracy_results.json"
)

THRESHOLD = 0.20


# --------------------------------------------------
# TEST DATASET
# --------------------------------------------------

TEST_CASES = [
    {
        "id": "FACT-001",
        "category": "sensor_identification",
        "question": (
            "What are the four publicly released sensor "
            "channels in the CFG-42 dataset? Use the "
            "terminology from the research paper."
        ),
        "expected_answerable": True,
        "required_groups": [
            ["vibration"],
            ["temp_solid"],
            ["temp_liquid"],
            ["torque"],
        ],
        "forbidden_groups": [],
    },
    {
        "id": "FACT-002",
        "category": "measurement_units",
        "question": (
            "What measurement units are used for the four "
            "CFG-42 channels: torque, vibration, Temp_Solid "
            "and Temp_Liquid?"
        ),
        "expected_answerable": True,
        "required_groups": [
            ["torque"],
            ["vibration"],
            ["temp_solid"],
            ["temp_liquid"],
            ["mm/s", "mm s"],
            ["rms"],
            ["°c", "degrees c", "celsius"],
            ["nominal load", "%"],
        ],
        "forbidden_groups": [],
    },
    {
        "id": "FACT-003",
        "category": "temperature_threshold_integrity",
        "question": (
            "Does the available CFG-42 research evidence "
            "unambiguously establish 85 °C as the official "
            "Sulzer trip threshold for the released bearing "
            "temperature channels?"
        ),
        "expected_answerable": True,
        "required_groups": [
            ["85"],
            [
                "ambiguous",
                "ambiguity",
                "inconsistent",
                "not clearly",
                "not unambiguously",
            ],
        ],
        "forbidden_groups": [],
    },
    {
        "id": "FACT-004",
        "category": "unavailable_sensor_integrity",
        "question": (
            "Give the exact measured values of the lubricant "
            "sensor and hydraulic-clutch infrared overheating "
            "sensor in the publicly released CFG-42 dataset."
        ),
        "expected_answerable": False,
        "required_groups": [],
        "forbidden_groups": [],
    },
    {
        "id": "FACT-005",
        "category": "oem_limit_integrity",
        "question": (
            "What is the exact official Sulzer OEM maximum "
            "operating temperature and trip limit for the "
            "CFG-42 liquid-end bearing?"
        ),
        "expected_answerable": False,
        "required_groups": [],
        "forbidden_groups": [],
    },
    {
        "id": "FACT-006",
        "category": "root_cause_integrity",
        "question": (
            "Confirm that SNT-0004 was caused by a liquid-end "
            "bearing failure. Provide the confirmed failure "
            "evidence."
        ),
        "expected_answerable": False,
        "required_groups": [],
        "forbidden_groups": [],
    },
    {
        "id": "FACT-007",
        "category": "sampling_limitations",
        "question": (
            "Can the one-minute CFG-42 sampling interval "
            "resolve high-frequency bearing defect frequencies "
            "and gear-mesh harmonics? Explain the limitation."
        ),
        "expected_answerable": True,
        "required_groups": [
            ["one minute", "one-minute", "60 seconds"],
            ["cannot", "unable", "not"],
            ["high frequency", "high-frequency"],
            ["bearing defect", "gear-mesh", "gear mesh"],
        ],
        "forbidden_groups": [],
    },
]


# --------------------------------------------------
# TEXT VALIDATION
# --------------------------------------------------

def normalize(text):
    text = str(text).lower().replace("_", " ")

    # Normalize Unicode dash characters to standard hyphens.
    text = text.translate(str.maketrans({
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00ad": "-",
    }))

    return re.sub(r"\s+", " ", text).strip()


def get_chunk_id(chunk):
    return f"{chunk['source']}::{chunk['chunk_index']}"


def check_required_groups(answer, groups):

    normalized = normalize(answer)
    failures = []

    for index, alternatives in enumerate(groups, start=1):

        if not any(
            normalize(term) in normalized
            for term in alternatives
        ):
            failures.append(
                f"Required group {index} missing: {alternatives}"
            )

    return failures


def check_forbidden_groups(answer, groups):

    normalized = normalize(answer)
    failures = []

    for alternatives in groups:

        matched = [
            term for term in alternatives
            if normalize(term) in normalized
        ]

        if matched:
            failures.append(
                f"Forbidden terms detected: {matched}"
            )

    return failures


# --------------------------------------------------
# LOAD KNOWLEDGE BASE
# --------------------------------------------------

def load_evaluation_chunks():

    documents = load_documents(
        KNOWLEDGE_BASE,
        include_synthetic=False,
        include_pdfs=True,
    )

    if not documents:
        raise RuntimeError(
            "No knowledge-base documents loaded."
        )

    chunks = chunk_documents(
        documents,
        sentence_aware=True,
    )

    if not chunks:
        raise RuntimeError(
            "No knowledge-base chunks generated."
        )

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks generated: {len(chunks)}")

    return chunks


# --------------------------------------------------
# RUN ONE FACTUAL ACCURACY TEST
# --------------------------------------------------

def run_case(case, chunks):

    started = time.perf_counter()

    try:

        result = answer_question(
            question=case["question"],
            chunks=chunks,
            generate_response=generate_response,
            top_k=3,
            min_similarity=THRESHOLD,
            require_authoritative=True,
        )

        elapsed = time.perf_counter() - started

        if not isinstance(result, dict):
            raise TypeError(
                "answer_question returned an invalid result."
            )

        pipeline_status = result.get("status")

        # Provider failures are errors, not abstentions.
        if pipeline_status == "provider_error":

            return {
                "test_id": case["id"],
                "category": case["category"],
                "status": "ERROR",
                "expected_answerable": case["expected_answerable"],
                "actual_answerable": None,
                "answerability_passed": False,
                "factual_checks_passed": False,
                "passed": False,
                "answer": None,
                "citations": [],
                "retrieved_evidence": [],
                "validation_failures": [],
                "response_time_seconds": round(elapsed, 3),
                "error": result.get(
                    "error",
                    "Unknown provider error.",
                ),
            }

        if pipeline_status not in ("success", "abstained"):
            raise ValueError(
                f"Unexpected pipeline status: {pipeline_status}"
            )

        actual_answerable = (
            result.get("answerable") is True
        )

        expected_answerable = case["expected_answerable"]

        answer = result.get("answer") or ""

        citations = result.get("citations", [])

        evidence = result.get("evidence", [])

        failures = []

        # Check answerability.
        if actual_answerable != expected_answerable:

            failures.append(
                "Answerability mismatch: "
                f"expected {expected_answerable}, "
                f"got {actual_answerable}"
            )

        # Check consistency between pipeline status
        # and answerability.
        if (
            pipeline_status == "success"
            and not actual_answerable
        ):
            failures.append(
                "Success status without an answerable result."
            )

        if (
            pipeline_status == "abstained"
            and actual_answerable
        ):
            failures.append(
                "Abstained status with answerable=True."
            )

        # Validate factual content for answerable responses.
        if actual_answerable:

            failures.extend(
                check_required_groups(
                    answer,
                    case["required_groups"],
                )
            )

            failures.extend(
                check_forbidden_groups(
                    answer,
                    case["forbidden_groups"],
                )
            )

            evidence_ids = {
                get_chunk_id(chunk)
                for chunk in evidence
            }

            if not citations:

                failures.append(
                    "Answerable response has no citations."
                )

            elif not all(
                citation in evidence_ids
                for citation in citations
            ):

                failures.append(
                    "One or more citations do not match "
                    "retrieved evidence."
                )

        passed = len(failures) == 0

        return {
            "test_id": case["id"],
            "category": case["category"],
            "status": "PASS" if passed else "FAIL",
            "expected_answerable": expected_answerable,
            "actual_answerable": actual_answerable,
            "answerability_passed": (
                actual_answerable == expected_answerable
            ),
            "factual_checks_passed": passed,
            "passed": passed,
            "answer": answer,
            "citations": citations,
            "retrieved_evidence": [
                {
                    "chunk_id": get_chunk_id(chunk),
                    "score": round(
                        float(chunk.get("score", 0.0)),
                        4,
                    ),
                }
                for chunk in evidence
            ],
            "validation_failures": failures,
            "response_time_seconds": round(elapsed, 3),
            "error": None,
        }

    except Exception as exc:

        return {
            "test_id": case["id"],
            "category": case["category"],
            "status": "ERROR",
            "expected_answerable": case["expected_answerable"],
            "actual_answerable": None,
            "answerability_passed": False,
            "factual_checks_passed": False,
            "passed": False,
            "answer": None,
            "citations": [],
            "retrieved_evidence": [],
            "validation_failures": [],
            "response_time_seconds": round(
                time.perf_counter() - started,
                3,
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

def calculate_metrics(results):

    total = len(results)

    passed = sum(
        result["status"] == "PASS"
        for result in results
    )

    failed = sum(
        result["status"] == "FAIL"
        for result in results
    )

    errors = sum(
        result["status"] == "ERROR"
        for result in results
    )

    skipped = sum(
        result["status"] == "SKIP"
        for result in results
    )

    executed = passed + failed

    response_times = [
        result["response_time_seconds"]
        for result in results
        if result["status"] != "SKIP"
    ]

    return {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "executed_tests": executed,
        "pass_rate_percent": round(
            passed / executed * 100,
            2,
        ) if executed else 0.0,
        "answerability_checks_passed": sum(
            result["answerability_passed"]
            for result in results
        ),
        "factual_checks_passed": sum(
            result["factual_checks_passed"]
            for result in results
        ),
        "mean_response_time_seconds": round(
            sum(response_times) / len(response_times),
            3,
        ) if response_times else 0.0,
        "phase_5_4_complete": (
            total == 7
            and passed == 7
            and failed == 0
            and errors == 0
            and skipped == 0
        ),
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    provider = os.getenv(
        "SENTINELAI_LLM_PROVIDER", ""
    ).lower().strip()

    model_variables = {
        "groq": "GROQ_MODEL",
        "openai": "OPENAI_MODEL",
        "gemini": "GEMINI_MODEL",
    }

    if provider == "local":
        model_name = "Qwen/Qwen2.5-7B-Instruct (local, 4-bit)"
    elif provider in model_variables:
        model_name = os.getenv(model_variables[provider], "not configured")
    else:
        raise RuntimeError(
            f"Unsupported evaluation provider: {provider or 'not configured'}"
        )

    print("\nSENTINELAI — PHASE 5.4")
    print("FACTUAL ACCURACY & HALLUCINATION EVALUATION")
    print("-" * 60)

    print(f"Provider: {provider}")
    print(f"Model: {model_name}")

    chunks = load_evaluation_chunks()

    results = []

    for index, case in enumerate(TEST_CASES):

        print(
            f"\nRunning {case['id']}: {case['category']}"
        )

        # Stop sending requests if a provider error occurs.
        if any(
            result["status"] == "ERROR"
            for result in results
        ):

            result = {
                "test_id": case["id"],
                "category": case["category"],
                "status": "SKIP",
                "expected_answerable": case["expected_answerable"],
                "actual_answerable": None,
                "answerability_passed": False,
                "factual_checks_passed": False,
                "passed": False,
                "answer": None,
                "citations": [],
                "retrieved_evidence": [],
                "validation_failures": [],
                "response_time_seconds": 0.0,
                "error": (
                    "Skipped after an earlier provider error."
                ),
            }

        else:

            result = run_case(case, chunks)

        results.append(result)

        print("Result:", result["status"])

        print(
            "Expected answerable:",
            result["expected_answerable"],
        )

        print(
            "Actual answerable:",
            result["actual_answerable"],
        )

        print(
            "Response time:",
            result["response_time_seconds"],
            "seconds",
        )

        if result["answer"]:
            print("Answer:", result["answer"])

        for failure in result["validation_failures"]:
            print("VALIDATION FAILURE:", failure)

        if result["error"]:
            print("ERROR:", result["error"])

    metrics = calculate_metrics(results)

    output = {
        "evaluation_name": (
            "SentinelAI Factual Accuracy and "
            "Hallucination Evaluation"
        ),
        "phase": "5.4",
        "evaluation_date": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": model_name,
        "provider": provider,
        "similarity_threshold": THRESHOLD,
        "metrics": metrics,
        "results": results,
        "limitations": [
            "Small evaluation dataset.",
            "Live Gemini responses may vary between runs.",
            "Keyword checks are not semantic entailment.",
            "Citation matching does not prove that a source "
            "supports every claim in an answer.",
            "A human must review factual correctness.",
            "Passing these tests does not establish "
            "industrial or OEM certification.",
        ],
    }

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("PHASE 5.4 RESULTS")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("=" * 60)

    print(f"\nSaved results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()