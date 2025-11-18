import csv
import json
import os
from typing import Literal

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.utils import set_seed, get_device, ensure_dir
from src.data.preprocess import (
    load_data,
    encode_labels,
    split_data,
    build_vocab,
    texts_to_sequences,
)
from src.data.dataset import create_dataloaders
from src.models.rnn_models import LSTMClassifier, GRUClassifier
from src.models.baselines import (
    train_tfidf_logreg,
    train_tfidf_svm,
    eval_tfidf_model,
)
from src.models.tiny_transformer import train_tiny_transformer
from src.evaluation.evaluation import evaluate_rnn_model
from src.evaluation.metrics import compute_classification_metrics


EXPERIMENTS_CSV = "experiments/results.csv"


def log_experiment_row(row: dict):
    ensure_dir("experiments")
    file_exists = os.path.isfile(EXPERIMENTS_CSV)
    with open(EXPERIMENTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_rnn(
    csv_path: str,
    model_type: Literal["lstm", "gru"] = "lstm",
    max_vocab_size: int = 20000,
    max_len: int = 70,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    batch_size: int = 32,
    epochs: int = 32,
    lr: float = 1e-3,
):
    set_seed()
    device = get_device()

    df = load_data(csv_path)
    df, label_encoder = encode_labels(df)
    train_df, val_df, test_df = split_data(df)

    word2idx = build_vocab(
        train_df["clean_text"].tolist(), max_vocab_size=max_vocab_size
    )
    vocab_size = len(word2idx)

    train_seqs = texts_to_sequences(train_df["clean_text"].tolist(), word2idx, max_len)
    val_seqs = texts_to_sequences(val_df["clean_text"].tolist(), word2idx, max_len)
    test_seqs = texts_to_sequences(test_df["clean_text"].tolist(), word2idx, max_len)

    num_classes = df["label"].nunique()
    train_labels = train_df["label"].tolist()
    val_labels = val_df["label"].tolist()
    test_labels = test_df["label"].tolist()

    train_loader, val_loader, test_loader = create_dataloaders(
        train_seqs,
        train_labels,
        val_seqs,
        val_labels,
        test_seqs,
        test_labels,
        batch_size=batch_size,
    )

    if model_type == "lstm":
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
    else:
        model = GRUClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    learning_curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x = x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # quick val
        _, _, val_metrics = evaluate_rnn_model(
            model,
            val_loader,
            device,
            target_names=label_encoder.classes_.tolist(),
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )
        learning_curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
        )

    # final test eval
    y_true, y_pred, test_metrics = evaluate_rnn_model(
        model,
        test_loader,
        device,
        target_names=label_encoder.classes_.tolist(),
    )
    print("Test metrics:", test_metrics["accuracy"], test_metrics["f1_macro"])

    row = {
        "model": f"rnn_{model_type}",
        "accuracy": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "notes": f"embed={embed_dim},hidden={hidden_dim},epochs={epochs}",
        "learning_curve": json.dumps(learning_curve),
    }
    log_experiment_row(row)

    return {
        "model": model,
        "word2idx": word2idx,
        "label_encoder": label_encoder,
        "test_metrics": test_metrics,
    }


def train_baselines(csv_path: str):
    set_seed()
    df = load_data(csv_path)
    df, label_encoder = encode_labels(df)
    train_df, val_df, test_df = split_data(df)

    # majority baseline
    majority_label = train_df["label"].value_counts().idxmax()
    majority_preds = [majority_label] * len(test_df)
    maj_metrics = compute_classification_metrics(
        test_df["label"].tolist(),
        majority_preds,
        target_names=label_encoder.classes_.tolist(),
    )
    log_experiment_row(
        {
            "model": "majority",
            "accuracy": maj_metrics["accuracy"],
            "f1_macro": maj_metrics["f1_macro"],
            "notes": "always predict most frequent class",
            "learning_curve": json.dumps([]),
        }
    )

    # tf-idf + logreg
    vec_lr, clf_lr, lr_curve = train_tfidf_logreg(
        train_df["clean_text"].tolist(),
        train_df["label"].tolist(),
    )
    preds_lr, report_lr = eval_tfidf_model(
        vec_lr,
        clf_lr,
        test_df["clean_text"].tolist(),
        test_df["label"].tolist(),
        target_names=label_encoder.classes_.tolist(),
    )
    metrics_lr = compute_classification_metrics(
        test_df["label"].tolist(),
        preds_lr,
        target_names=label_encoder.classes_.tolist(),
    )
    log_experiment_row(
        {
            "model": "tfidf_logreg",
            "accuracy": metrics_lr["accuracy"],
            "f1_macro": metrics_lr["f1_macro"],
            "notes": "max_features=20000, ngram=(1,2)",
            "learning_curve": json.dumps(lr_curve),
        }
    )

    # tf-idf + svm
    vec_svm, clf_svm, svm_curve = train_tfidf_svm(
        train_df["clean_text"].tolist(),
        train_df["label"].tolist(),
    )
    preds_svm, report_svm = eval_tfidf_model(
        vec_svm,
        clf_svm,
        test_df["clean_text"].tolist(),
        test_df["label"].tolist(),
        target_names=label_encoder.classes_.tolist(),
    )
    metrics_svm = compute_classification_metrics(
        test_df["label"].tolist(),
        preds_svm,
        target_names=label_encoder.classes_.tolist(),
    )
    log_experiment_row(
        {
            "model": "tfidf_svm",
            "accuracy": metrics_svm["accuracy"],
            "f1_macro": metrics_svm["f1_macro"],
            "notes": "LinearSVC, max_features=20000, ngram=(1,2)",
            "learning_curve": json.dumps(svm_curve),
        }
    )

    return {
        "majority": maj_metrics,
        "tfidf_logreg": metrics_lr,
        "tfidf_svm": metrics_svm,
        "label_encoder": label_encoder,
        "vec_logreg": vec_lr,
        "clf_logreg": clf_lr,
        "logreg_learning_curve": lr_curve,
        "vec_svm": vec_svm,
        "clf_svm": clf_svm,
        "svm_learning_curve": svm_curve,
    }


def train_transformer_experiment(csv_path: str):
    set_seed()
    df = load_data(csv_path)
    df, label_encoder = encode_labels(df)
    train_df, val_df, test_df = split_data(df)

    num_labels = df["label"].nunique()
    tokenizer, model, trainer = train_tiny_transformer(
        train_texts=train_df["clean_text"].tolist(),
        train_labels=train_df["label"].tolist(),
        val_texts=val_df["clean_text"].tolist(),
        val_labels=val_df["label"].tolist(),
        num_labels=num_labels,
        output_dir="./experiments/tiny_transformer",
    )

    # test eval
    test_enc = tokenizer(
        test_df["clean_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
    )
    import numpy as np

    device = model.device
    with torch.no_grad():
        inputs = {
            "input_ids": torch.tensor(test_enc["input_ids"], device=device),
            "attention_mask": torch.tensor(test_enc["attention_mask"], device=device),
        }
        outputs = model(**inputs)
    preds = outputs.logits.argmax(-1).cpu().numpy()
    y_true = test_df["label"].to_numpy()

    metrics = compute_classification_metrics(
        y_true.tolist(),
        preds.tolist(),
        target_names=label_encoder.classes_.tolist(),
    )

    transformer_learning_curve = []
    for entry in trainer.state.log_history:
        if "epoch" not in entry:
            continue
        if "loss" in entry:
            transformer_learning_curve.append(
                {"epoch": entry["epoch"], "train_loss": float(entry["loss"])}
            )
        eval_keys = {"eval_loss", "eval_accuracy", "eval_f1"}
        if eval_keys.intersection(entry.keys()):
            eval_loss = entry.get("eval_loss")
            transformer_learning_curve.append(
                {
                    "epoch": entry["epoch"],
                    "eval_loss": float(eval_loss) if eval_loss is not None else None,
                    "eval_accuracy": entry.get("eval_accuracy"),
                    "eval_f1": entry.get("eval_f1"),
                }
            )

    log_experiment_row(
        {
            "model": "tiny_transformer_distilbert",
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "notes": "distilbert-base-uncased, epochs=3",
            "learning_curve": json.dumps(transformer_learning_curve),
        }
    )

    return {
        "tokenizer": tokenizer,
        "model": model,
        "label_encoder": label_encoder,
        "test_metrics": metrics,
    }
