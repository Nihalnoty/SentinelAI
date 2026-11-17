import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import chunk_documents
from src.rag.ingest import load_documents
from src.rag.pipeline import answer_question


KNOWLEDGE_BASE = PROJECT_ROOT / "data" / "knowledge_base"

RESULTS_DIRECTORY = (
    PROJECT_ROOT / "evaluation" / "results"
)

RESULTS_FILE = (
    RESULTS_DIRECTORY / "factual_accuracy_offline_results.json"
)

THRESHOLD = 0.20


# --------------------------------------------------
# EXISTING FACTUAL ACCURACY TEST CASES
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
            ["ambiguous", "ambiguity", "inconsistent",
             "not clearly", "not unambiguously"],
        ],
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
            ["one minute", "60 seconds"],
            ["cannot", "unable", "not"],
            ["high frequency", "high-frequency"],
            ["bearing defect", "gear-mesh", "gear mesh"],
        ],
    },
]


# --------------------------------------------------
# CONTROLLED MOCK RESPONSES
# --------------------------------------------------

MOCK_ANSWERS = {
    "FACT-001": (
        "The four recorded sensor channels are "
        "Vibration, Temp_Solid, Temp_Liquid and Torque."
    ),

    "FACT-002": (
        "Torque is expressed as a percentage of nominal load. "
        "Vibration is measured in mm/s RMS. "
        "Temp_Solid and Temp_Liquid are measured in degrees Celsius."
    ),

    "FACT-003": (
        "No. The available evidence is ambiguous and does not "
        "unambiguously establish 85 °C as the official Sulzer "
        "trip threshold for the released bearing temperature channels."
    ),

    "FACT-004": None,
    "FACT-005": None,
    "FACT-006": None,

    "FACT-007": (
        "The dataset uses a one minute sampling interval. "
        "It cannot resolve high-frequency bearing defect "
        "frequencies or gear-mesh harmonics."
    ),
}


def make_mock_generator(test_id):
    """
    Return a controlled response generator.

    It extracts citation IDs from the actual evidence prompt.
    No external API is called.
    """

    def mock_generate(prompt):

        if MOCK_ANSWERS[test_id] is None:
            return json.dumps({
                "answerable": False,
                "answer": "Insufficient evidence.",
                "citations": [],
            })

        # Match only actual evidence headers, e.g. [source.pdf::4].
        # Do not mistake passage text such as [PDF page 1] for a chunk ID.
        citation_ids = re.findall(
            r"^\[([^\]\r\n]+::\d+)\]\s*$",
            prompt,
            flags=re.MULTILINE,
        )

        if not citation_ids:
            return json.dumps({
                "answerable": False,
                "answer": "Insufficient evidence.",
                "citations": [],
            })

        return json.dumps({
            "answerable": True,
            "answer": MOCK_ANSWERS[test_id],
            "citations": citation_ids,
        })

    return mock_generate


# --------------------------------------------------
# TEXT VALIDATION
# --------------------------------------------------

def normalize(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).lower().replace("_", " "),
    ).strip()


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


def get_chunk_id(chunk):
    return f"{chunk['source']}::{chunk['chunk_index']}"


# --------------------------------------------------
# LOAD REAL KNOWLEDGE BASE
# --------------------------------------------------

def load_evaluation_chunks():

    documents = load_documents(
        KNOWLEDGE_BASE,
        include_synthetic=False,
        include_pdfs=True,
    )

    if not documents:
        raise RuntimeError("No knowledge-base documents loaded.")

    chunks = chunk_documents(
        documents,
        sentence_aware=True,
    )

    if not chunks:
        raise RuntimeError("No knowledge-base chunks generated.")

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks generated: {len(chunks)}")

    return chunks


# --------------------------------------------------
# RUN ONE OFFLINE TEST
# --------------------------------------------------

def run_case(case, chunks):

    started = time.perf_counter()

    result = answer_question(
        question=case["question"],
        chunks=chunks,
        generate_response=make_mock_generator(case["id"]),
        top_k=3,
        min_similarity=THRESHOLD,
        require_authoritative=True,
    )

    elapsed = time.perf_counter() - started

    actual_answerable = result.get("answerable") is True
    expected_answerable = case["expected_answerable"]

    answer = result.get("answer", "")
    citations = result.get("citations", [])
    evidence = result.get("evidence", [])

    failures = []

    if result.get("status") == "provider_error":
        failures.append("Unexpected provider error in offline mode.")

    if actual_answerable != expected_answerable:
        failures.append("Answerability mismatch.")

    if actual_answerable:

        failures.extend(
            check_required_groups(
                answer,
                case["required_groups"],
            )
        )

        evidence_ids = {
            get_chunk_id(chunk)
            for chunk in evidence
        }

        if not citations:
            failures.append("No citations returned.")

        elif not all(
            citation in evidence_ids
            for citation in citations
        ):
            failures.append(
                "Citation does not match returned evidence."
            )

    passed = len(failures) == 0

    return {
        "test_id": case["id"],
        "category": case["category"],
        "status": "PASS" if passed else "FAIL",
        "expected_answerable": expected_answerable,
        "actual_answerable": actual_answerable,
        "answer": answer,
        "citations": citations,
        "validation_failures": failures,
        "response_time_seconds": round(elapsed, 4),
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\nSENTINELAI - OFFLINE FACTUAL ACCURACY")
    print("=" * 60)
    print("Evaluation mode: MOCK / OFFLINE")
    print("Gemini API calls: ZERO")
    print("=" * 60)

    chunks = load_evaluation_chunks()

    results = []

    for case in TEST_CASES:

        print(
            f"\nRunning {case['id']}: "
            f"{case['category']}"
        )

        result = run_case(case, chunks)
        results.append(result)

        print("Result:", result["status"])

        for failure in result["validation_failures"]:
            print("FAILURE:", failure)

    passed = sum(
        result["status"] == "PASS"
        for result in results
    )

    failed = len(results) - passed

    metrics = {
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate_percent": round(
            passed / len(results) * 100,
            2,
        ),
        "evaluation_mode": "offline_mock",
        "live_gemini_tested": False,
    }

    output = {
        "evaluation_name": (
            "SentinelAI Offline Factual Accuracy Evaluation"
        ),
        "evaluation_date": datetime.now(
            timezone.utc
        ).isoformat(),
        "metrics": metrics,
        "results": results,
        "limitations": [
            "Uses controlled mock responses, not live Gemini.",
            "Tests actual RAG retrieval and pipeline validation.",
            "Keyword checks do not prove semantic correctness.",
            "Citation matching does not prove claim entailment.",
            "Does not establish real Gemini factual accuracy.",
        ],
    }

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_FILE.write_text(
        json.dumps(
            output,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("OFFLINE EVALUATION SUMMARY")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved:", RESULTS_FILE)


if __name__ == "__main__":
    main()