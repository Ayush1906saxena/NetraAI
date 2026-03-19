"""
Prometheus metrics for Netra AI model monitoring.

Exposes key operational metrics for the DR screening service:
- Inference latency
- Screening throughput
- DR grade distribution
- Image quality rejection rate
- Model confidence
- Human-AI agreement
- False negative tracking

These metrics are scraped by Prometheus and visualized in Grafana
dashboards for real-time operational monitoring.
"""

from prometheus_client import Counter, Gauge, Histogram, Info, Summary


# ---------------------------------------------------------------------------
# Inference latency
# ---------------------------------------------------------------------------
INFERENCE_LATENCY = Histogram(
    "netra_inference_latency_seconds",
    "Time taken for a single model inference (including preprocessing)",
    labelnames=["model_name", "device"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ---------------------------------------------------------------------------
# Screening counts
# ---------------------------------------------------------------------------
SCREENING_COUNT = Counter(
    "netra_screening_total",
    "Total number of screenings processed",
    labelnames=["store_id", "status"],
)

# ---------------------------------------------------------------------------
# DR grade distribution
# ---------------------------------------------------------------------------
DR_GRADE_DISTRIBUTION = Counter(
    "netra_dr_grade_total",
    "Count of predictions per DR grade",
    labelnames=["grade", "grade_name"],
)

# Convenience constants for labeling
_GRADE_NAMES = {
    0: "no_dr",
    1: "mild_npdr",
    2: "moderate_npdr",
    3: "severe_npdr",
    4: "pdr",
}


def record_dr_grade(grade: int) -> None:
    """Record a DR grade prediction in Prometheus."""
    grade_name = _GRADE_NAMES.get(grade, f"unknown_{grade}")
    DR_GRADE_DISTRIBUTION.labels(grade=str(grade), grade_name=grade_name).inc()


# ---------------------------------------------------------------------------
# Image Quality Assessment rejection rate
# ---------------------------------------------------------------------------
IQA_REJECTION_RATE = Counter(
    "netra_iqa_decisions_total",
    "IQA accept/reject decisions",
    labelnames=["decision"],
)

IQA_QUALITY_SCORE = Histogram(
    "netra_iqa_quality_score",
    "Distribution of IQA quality scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


def record_iqa_decision(accepted: bool, quality_score: float) -> None:
    """Record an IQA accept/reject decision."""
    IQA_REJECTION_RATE.labels(decision="accepted" if accepted else "rejected").inc()
    IQA_QUALITY_SCORE.observe(quality_score)


# ---------------------------------------------------------------------------
# Model confidence
# ---------------------------------------------------------------------------
MODEL_CONFIDENCE = Histogram(
    "netra_model_confidence",
    "Distribution of model confidence (max softmax probability)",
    labelnames=["model_name"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

LOW_CONFIDENCE_COUNT = Counter(
    "netra_low_confidence_total",
    "Number of predictions below confidence threshold (flagged for review)",
    labelnames=["model_name"],
)

CONFIDENCE_THRESHOLD = 0.7


def record_model_confidence(model_name: str, confidence: float) -> None:
    """Record model confidence and flag low-confidence predictions."""
    MODEL_CONFIDENCE.labels(model_name=model_name).observe(confidence)
    if confidence < CONFIDENCE_THRESHOLD:
        LOW_CONFIDENCE_COUNT.labels(model_name=model_name).inc()


# ---------------------------------------------------------------------------
# Human-AI agreement
# ---------------------------------------------------------------------------
HUMAN_AI_AGREEMENT = Counter(
    "netra_human_ai_agreement_total",
    "Tracking agreement between model predictions and ophthalmologist reviews",
    labelnames=["outcome"],
)

HUMAN_AI_GRADE_DIFF = Histogram(
    "netra_human_ai_grade_diff",
    "Distribution of grade differences between AI and human reviewer",
    buckets=(0, 1, 2, 3, 4),
)


def record_human_review(
    ai_grade: int,
    human_grade: int,
) -> None:
    """
    Record a human-AI comparison when an ophthalmologist reviews a screening.

    Args:
        ai_grade: Model-predicted DR grade (0-4).
        human_grade: Ophthalmologist-assigned DR grade (0-4).
    """
    diff = abs(ai_grade - human_grade)
    agreed = ai_grade == human_grade

    HUMAN_AI_AGREEMENT.labels(outcome="agree" if agreed else "disagree").inc()
    HUMAN_AI_GRADE_DIFF.observe(diff)

    # Track if AI missed a referable case
    if human_grade >= 2 and ai_grade < 2:
        FALSE_NEGATIVE_COUNT.labels(severity="referable_missed").inc()
    elif human_grade >= 3 and ai_grade < 3:
        FALSE_NEGATIVE_COUNT.labels(severity="severe_missed").inc()


# ---------------------------------------------------------------------------
# False negatives (critical safety metric)
# ---------------------------------------------------------------------------
FALSE_NEGATIVE_COUNT = Counter(
    "netra_false_negative_total",
    "Count of confirmed false negatives (AI predicted non-referable, "
    "human reviewer found referable DR)",
    labelnames=["severity"],
)

# ---------------------------------------------------------------------------
# System-level metrics
# ---------------------------------------------------------------------------
PREPROCESSING_LATENCY = Histogram(
    "netra_preprocessing_latency_seconds",
    "Time for fundus image preprocessing",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
)

MODEL_LOAD_TIME = Gauge(
    "netra_model_load_time_seconds",
    "Time taken to load the model into memory",
    labelnames=["model_name"],
)

ACTIVE_MODELS = Gauge(
    "netra_active_models",
    "Number of models currently loaded in memory",
)

GPU_MEMORY_USED_MB = Gauge(
    "netra_gpu_memory_used_mb",
    "GPU memory used by inference models",
    labelnames=["device"],
)

# ---------------------------------------------------------------------------
# Batch processing metrics
# ---------------------------------------------------------------------------
BATCH_SIZE_HISTOGRAM = Histogram(
    "netra_batch_size",
    "Distribution of inference batch sizes",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

QUEUE_DEPTH = Gauge(
    "netra_inference_queue_depth",
    "Number of images waiting in the inference queue",
)

# ---------------------------------------------------------------------------
# Model version tracking
# ---------------------------------------------------------------------------
MODEL_INFO = Info(
    "netra_model",
    "Currently deployed model information",
)


def set_model_info(
    model_name: str,
    version: str,
    checkpoint: str,
    num_params: int,
) -> None:
    """Record currently deployed model metadata."""
    MODEL_INFO.info({
        "model_name": model_name,
        "version": version,
        "checkpoint": checkpoint,
        "num_params": str(num_params),
    })


# ---------------------------------------------------------------------------
# Convenience: record a full screening event
# ---------------------------------------------------------------------------
def record_screening(
    store_id: str,
    model_name: str,
    dr_grade: int,
    confidence: float,
    latency_seconds: float,
    iqa_accepted: bool,
    iqa_score: float,
    status: str = "completed",
) -> None:
    """
    One-call convenience to record all metrics for a single screening.

    Args:
        store_id: Identifier for the screening location.
        model_name: Name/version of the DR model used.
        dr_grade: Predicted DR grade (0-4).
        confidence: Max softmax probability.
        latency_seconds: Total inference time.
        iqa_accepted: Whether the image passed IQA.
        iqa_score: IQA quality score.
        status: Screening status (completed, failed, rejected).
    """
    SCREENING_COUNT.labels(store_id=store_id, status=status).inc()
    INFERENCE_LATENCY.labels(model_name=model_name, device="auto").observe(latency_seconds)
    record_dr_grade(dr_grade)
    record_model_confidence(model_name, confidence)
    record_iqa_decision(iqa_accepted, iqa_score)
