import json
import os
from typing import List, Dict
import mlflow

HARD_NEGATIVES_FILE = "hard_negatives.json"


def mine_hard_negative(query: str, answer: str, docs_used: List, rating: bool):
    if rating:
        return

    entry = {
        "query": query,
        "hard_negative_chunks": [d.page_content[:500] for d in docs_used],
        "bad_answer": answer[:300]
    }

    existing = _load_negatives()
    existing.append(entry)
    _save_negatives(existing)

    with mlflow.start_run(run_name="hard_negative_mined"):
        mlflow.log_param("query", query[:100])
        mlflow.log_metric("num_negative_chunks", len(docs_used))


def _load_negatives() -> List[Dict]:
    if not os.path.exists(HARD_NEGATIVES_FILE):
        return []
    try:
        with open(HARD_NEGATIVES_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save_negatives(data: List[Dict]):
    try:
        with open(HARD_NEGATIVES_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def get_mining_stats() -> dict:
    negatives = _load_negatives()
    return {
        "total_hard_negatives_mined": len(negatives),
        "ready_for_finetuning": len(negatives) >= 20
    }


def prepare_training_pairs(positive_source: str = "feedback_log.json") -> List[Dict]:
    negatives = _load_negatives()

    positives_by_query = {}
    if os.path.exists(positive_source):
        try:
            with open(positive_source) as f:
                feedback = json.load(f)
            for entry in feedback:
                if entry.get("rating"):
                    positives_by_query[entry["query"]] = entry["answer"]
        except Exception:
            pass

    triples = []
    for neg_entry in negatives:
        query = neg_entry["query"]
        positive = positives_by_query.get(query)
        if positive:
            for neg_chunk in neg_entry["hard_negative_chunks"]:
                triples.append({"query": query, "positive": positive, "negative": neg_chunk})

    with mlflow.start_run(run_name="training_pairs_prepared"):
        mlflow.log_metric("num_triples", len(triples))

    return triples


def run_finetune_script(triples: List[Dict], output_path: str = "./finetuned_retriever", epochs: int = 3):
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError("sentence-transformers not installed — run: pip install sentence-transformers")

    if len(triples) < 10:
        raise ValueError(f"Only {len(triples)} training triples available — need more feedback before fine-tuning is worthwhile.")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    examples = [InputExample(texts=[t["query"], t["positive"], t["negative"]]) for t in triples]

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=8)
    train_loss = losses.TripletLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        output_path=output_path
    )

    with mlflow.start_run(run_name="retriever_finetune"):
        mlflow.log_param("num_triples", len(triples))
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("output_path", output_path)

    return {"status": "complete", "output_path": output_path, "triples_used": len(triples)}
