import mlflow

# Lazy-loaded — avoids blocking app startup (or module import, e.g. during
# tests) on a network call to download model weights. Loaded once, on
# first actual use.
_nli_pipeline = None


def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        _nli_pipeline = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small"
        )
    return _nli_pipeline


def detect_hallucination(answer: str, context: str) -> dict:
    context_short = context[:512]
    answer_short  = answer[:256]

    input_text = f"{context_short} [SEP] {answer_short}"
    result     = _get_nli_pipeline()(input_text)[0]

    is_hallucination = result["label"].lower() == "contradiction"

    # NOTE: bug-fixed — removed nested=True (was crashing with no active parent run)
    with mlflow.start_run(run_name="hallucination_check"):
        mlflow.log_param("answer_length",  len(answer))
        mlflow.log_param("context_length", len(context))
        mlflow.log_metric("hallucination_confidence", round(result["score"], 3))
        mlflow.log_param("hallucination_detected", str(is_hallucination))

    return {
        "label":              result["label"],
        "confidence":         round(result["score"], 3),
        "is_hallucination":   is_hallucination,
        "warning":            "⚠️ Possible hallucination detected" if is_hallucination else "✅ Answer appears grounded"
    }
