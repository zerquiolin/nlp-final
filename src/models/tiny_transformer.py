import os
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


@dataclass
class HFTextDataset(Dataset):
    encodings: Dict[str, torch.Tensor]
    labels: List[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_tiny_transformer(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    num_labels: int,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "./experiments/tiny_transformer",
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 5e-5,
):
    # Avoid tokenizer parallelism warning when forking (e.g., with Trainer)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    torch.set_default_dtype(torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_enc = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=128,
    )
    val_enc = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=128,
    )

    train_dataset = HFTextDataset(train_enc, train_labels)
    val_dataset = HFTextDataset(val_enc, val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=460,
        save_steps=460,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_pin_memory=False,  # MPS does not support pinned memory
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        preds = logits.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return tokenizer, model, trainer
