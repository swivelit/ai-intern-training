import argparse
import os
import collections
import inspect

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)


def safe_training_args(**kwargs):
    """
    Creates TrainingArguments with compatibility across transformers versions
    that may use `evaluation_strategy` or `eval_strategy`.
    """
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())

    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**filtered)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="distilbert-base-cased-distilled-squad")
    p.add_argument("--output_dir", type=str, default="outputs/qa_model")
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_samples", type=int, default=2000)
    p.add_argument("--eval_samples", type=int, default=500)
    return p.parse_args()


def prepare_train_features(examples, tokenizer, max_length, doc_stride):
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def prepare_eval_features(examples, tokenizer, max_length, doc_stride):
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []

    new_offsets = []
    for i, offsets in enumerate(tokenized["offset_mapping"]):
        seq_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])
        new_offsets.append([o if seq_ids[k] == 1 else None for k, o in enumerate(offsets)])

    tokenized["offset_mapping"] = new_offsets
    return tokenized


def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}

    features_per_example = collections.defaultdict(list)
    for i, f in enumerate(features):
        features_per_example[example_id_to_index[f["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]

        best_text = ""
        best_score = -1e9

        for fi in feature_indices:
            start_logits = all_start_logits[fi]
            end_logits = all_end_logits[fi]
            offsets = features[fi]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-1:-n_best_size-1:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size-1:-1].tolist()

            for s in start_indexes:
                for e in end_indexes:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s:
                        continue
                    if (e - s + 1) > max_answer_length:
                        continue

                    start_char, end_char = offsets[s][0], offsets[e][1]
                    text = context[start_char:end_char]
                    score = start_logits[s] + end_logits[e]

                    if score > best_score:
                        best_score = score
                        best_text = text

        predictions[example["id"]] = best_text

    return predictions


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("rajpurkar/squad")
    train_ds = ds["train"]
    eval_ds = ds["validation"]

    if args.train_samples != -1:
        train_ds = train_ds.select(range(min(args.train_samples, len(train_ds))))
    if args.eval_samples != -1:
        eval_ds = eval_ds.select(range(min(args.eval_samples, len(eval_ds))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    train_features = train_ds.map(
        lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=train_ds.column_names,
    )

    eval_features = eval_ds.map(
        lambda x: prepare_eval_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    metric = evaluate.load("squad")

    training_args = safe_training_args(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=eval_features,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()

    raw_preds = trainer.predict(eval_features)
    preds = postprocess_qa_predictions(eval_ds, eval_features, raw_preds.predictions)

    formatted_preds = [{"id": k, "prediction_text": v} for k, v in preds.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_ds]

    results = metric.compute(predictions=formatted_preds, references=references)
    print("SQuAD metrics:", results)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
