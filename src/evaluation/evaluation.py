from typing import Tuple, List

import torch
from tqdm import tqdm

from .metrics import compute_classification_metrics


def evaluate_rnn_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    target_names: List[str],
) -> Tuple[List[int], List[int], dict]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, labels in tqdm(dataloader, desc="Evaluating RNN"):
            x = x.to(device)
            labels = labels.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, target_names)
    return y_true, y_pred, metrics
