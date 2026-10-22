"""Fine-tune DistilBERT for BANKING77 classification on a Kaggle GPU."""

import argparse
from pathlib import Path

import numpy as np

from fintech_support.data import (
    download_data,
    ensure_known_labels,
    label_maps,
    read_split,
    training_split,
)
from fintech_support.evaluation import evaluate, score, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/distilbert"))
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, max-length, and learning-rate must be positive")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("warmup-ratio must be in [0, 1)")

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable: launch the training job with krun --gpu P100")
    set_seed(args.seed)

    download_data(args.data_dir)
    source_train = read_split(args.data_dir, "train")
    test = read_split(args.data_dir, "test")
    label2id, id2label = label_maps(source_train)
    ensure_known_labels(test, label2id)
    train, validation = training_split(source_train, seed=args.seed)
    print(
        f"train={len(train)} validation={len(validation)} test={len(test)} "
        f"labels={len(label2id)} GPUs={torch.cuda.device_count()} "
        f"epochs={args.epochs} lr={args.learning_rate} warmup={args.warmup_ratio}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def to_dataset(frame):
        dataset = Dataset.from_dict(
            {
                "text": frame["text"].tolist(),
                "labels": [label2id[label] for label in frame["category"]],
            }
        )
        return dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length),
            batched=True,
            remove_columns=["text"],
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    training_args = TrainingArguments(
        output_dir=str(Path("artifacts") / "distilbert-checkpoints"),
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=True,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )

    def compute_metrics(prediction):
        return score(prediction.label_ids, np.argmax(prediction.predictions, axis=-1))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=to_dataset(train),
        eval_dataset=to_dataset(validation),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    output_dir = args.output_dir
    trainer.save_model(str(output_dir / "model"))
    write_json(
        output_dir / "training_history.json",
        {
            "global_step": trainer.state.global_step,
            "best_validation_macro_f1": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "history": trainer.state.log_history,
        },
    )

    val_pred = trainer.predict(to_dataset(validation))
    test_pred = trainer.predict(to_dataset(test))

    def named_evaluation(prediction):
        actual = [id2label[int(index)] for index in prediction.label_ids]
        predicted = [id2label[int(index)] for index in np.argmax(prediction.predictions, axis=-1)]
        return evaluate(actual, predicted)

    metrics = {
        "model": args.model_name,
        "seed": args.seed,
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
        },
        "validation": named_evaluation(val_pred),
        "test": named_evaluation(test_pred),
    }
    write_json(output_dir / "metrics.json", metrics)
    for split in ("validation", "test"):
        result = metrics[split]
        print(f"{split}: accuracy={result['accuracy']:.4f}, macro_f1={result['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
