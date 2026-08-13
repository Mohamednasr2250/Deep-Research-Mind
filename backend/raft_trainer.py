import json
import random
from typing import List, Dict
import mlflow


def generate_raft_example(question: str, golden_chunk: str, distractor_chunks: List[str], llm) -> Dict:
    answer_prompt = f"""Context: {golden_chunk}

Question: {question}
Answer using only this context:"""
    gold_answer = llm.invoke(answer_prompt).strip()

    all_chunks = [golden_chunk] + distractor_chunks
    random.shuffle(all_chunks)

    combined_context = "\n\n".join(f"[Passage {i+1}]: {c}" for i, c in enumerate(all_chunks))

    return {
        "question": question,
        "context": combined_context,
        "golden_chunk": golden_chunk,
        "answer": gold_answer
    }


def build_raft_dataset(qa_pairs: List[Dict], hybrid_retriever, num_distractors: int = 3) -> List[Dict]:
    dataset = []
    for pair in qa_pairs:
        question = pair["question"]
        golden   = pair["golden_chunk"]

        retrieved = hybrid_retriever.retrieve(question, k=num_distractors + 3)
        distractors = [d.page_content for d in retrieved if d.page_content.strip() != golden.strip()][:num_distractors]

        if len(distractors) < num_distractors:
            continue

        dataset.append({"question": question, "golden_chunk": golden, "distractors": distractors})

    with mlflow.start_run(run_name="raft_dataset_built"):
        mlflow.log_metric("num_examples", len(dataset))

    return dataset


def materialize_raft_examples(dataset: List[Dict], llm) -> List[Dict]:
    examples = []
    for item in dataset:
        example = generate_raft_example(item["question"], item["golden_chunk"], item["distractors"], llm)
        examples.append(example)
    return examples


def save_raft_dataset(examples: List[Dict], path: str = "raft_dataset.jsonl"):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def run_raft_finetune(dataset_path: str = "raft_dataset.jsonl", output_dir: str = "./raft_finetuned", base_model: str = "google/flan-t5-small"):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Missing packages — run: pip install peft transformers datasets")

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q", "v"]
    )
    model = get_peft_model(model, lora_config)

    def preprocess(example):
        input_text  = f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer:"
        target_text = example["answer"]
        model_inputs = tokenizer(input_text, max_length=1024, truncation=True, padding="max_length")
        labels = tokenizer(target_text, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(output_dir)

    with mlflow.start_run(run_name="raft_finetune"):
        mlflow.log_param("base_model", base_model)
        mlflow.log_param("dataset_size", len(dataset))
        mlflow.log_param("output_dir", output_dir)

    return {"status": "complete", "output_dir": output_dir}


def compare_before_after(eval_questions: List[Dict], base_llm, finetuned_llm, evaluator_fn) -> dict:
    base_answers = [base_llm.invoke(q["prompt"]) for q in eval_questions]
    finetuned_answers = [finetuned_llm.invoke(q["prompt"]) for q in eval_questions]

    base_scores = evaluator_fn(
        questions=[q["question"] for q in eval_questions],
        answers=base_answers,
        contexts=[q["contexts"] for q in eval_questions],
        ground_truths=[q["ground_truth"] for q in eval_questions]
    )
    finetuned_scores = evaluator_fn(
        questions=[q["question"] for q in eval_questions],
        answers=finetuned_answers,
        contexts=[q["contexts"] for q in eval_questions],
        ground_truths=[q["ground_truth"] for q in eval_questions]
    )

    with mlflow.start_run(run_name="raft_before_after_comparison"):
        mlflow.log_metric("base_faithfulness", base_scores["faithfulness"])
        mlflow.log_metric("finetuned_faithfulness", finetuned_scores["faithfulness"])
        mlflow.log_metric("improvement", round(finetuned_scores["faithfulness"] - base_scores["faithfulness"], 3))

    return {"before": base_scores, "after": finetuned_scores}
