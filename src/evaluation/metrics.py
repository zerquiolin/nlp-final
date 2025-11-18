from typing import List, Dict

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    target_names: List[str],
) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": report,
        "confusion_matrix": cm,
    }
