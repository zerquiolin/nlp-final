# metrics_report.py
"""
metrics_report.py
Genera métricas extendidas, plots y archivos para informes a partir de:
- y_true (list/np.array)
- y_pred  (list/np.array)
- y_scores (optional) --> array-like shape (n_samples, n_classes) with probabilities/logits

Funcionalidades:
- compute_metrics_dict: métricas numéricas (accuracy, f1 macro/micro, per-class precision/recall/f1, support)
- save_classification_table: guarda classification_report como CSV
- plot_confusion_matrix: imagen PNG (normalizada opcional)
- plot_roc_curves (multiclase One-vs-Rest, guarda PNG)
- plot_precision_recall_curves (multiclase, guarda PNG)
- plot_calibration_curve (si se pasan probabilidades)
- generate_report: orquesta todo y guarda en output_dir
- CLI: pasar un CSV con columnas 'true','pred' y opcionales 'prob_0','prob_1',...
"""
import os
import json
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

# -----------------------
# Helpers / core metrics
# -----------------------
def compute_metrics_dict(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
) -> Dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    if target_names is None:
        target_names = [str(l) for l in labels]
    # base metrics
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_micro = float(f1_score(y_true, y_pred, average="micro"))

    # per-class precision/recall/f1/support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    per_class = []
    for i, lab in enumerate(labels):
        per_class.append(
            {
                "label": int(lab),
                "label_name": target_names[i] if i < len(target_names) else str(lab),
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
        )

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels.tolist(),
        "target_names": target_names,
    }

    # multiclass AUC/PR if scores provided
    if y_scores is not None:
        y_scores = np.asarray(y_scores)
        n_classes = y_scores.shape[1]
        # binarize true labels
        try:
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        except Exception:
            # fallback: infer classes from labels
            classes = sorted(list(set(y_true) | set(y_pred)))
            y_true_bin = label_binarize(y_true, classes=classes)
        per_class_auc = {}
        per_class_ap = {}
        aucs = []
        aps = []
        for c in range(y_scores.shape[1]):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_scores[:, c])
                roc_auc = float(auc(fpr, tpr))
            except Exception:
                roc_auc = None
            try:
                ap = float(average_precision_score(y_true_bin[:, c], y_scores[:, c]))
            except Exception:
                ap = None
            per_class_auc[c] = roc_auc
            per_class_ap[c] = ap
            if roc_auc is not None:
                aucs.append(roc_auc)
            if ap is not None:
                aps.append(ap)
        # macro AUC/AP
        metrics["per_class_auc"] = per_class_auc
        metrics["per_class_ap"] = per_class_ap
        metrics["roc_auc_macro"] = float(np.mean(aucs)) if aucs else None
        metrics["avg_precision_macro"] = float(np.mean(aps)) if aps else None

        # brier score (mean over classes)
        try:
            brier = np.mean([brier_score_loss(y_true_bin[:, c], y_scores[:, c]) for c in range(y_scores.shape[1])])
            metrics["brier_score_mean"] = float(brier)
        except Exception:
            metrics["brier_score_mean"] = None

    return metrics

# -----------------------
# Save and plot helpers
# -----------------------
def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_classification_table(y_true, y_pred, target_names, path_csv: str):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.to_csv(path_csv, index=True)
    return df

def plot_confusion_matrix(cm, labels, out_path: str, normalize: bool = True, figsize=(6,6), cmap=None):
    cm = np.array(cm)
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, where=row_sums!=0)
    else:
        cm_norm = cm
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           ylabel="True label", xlabel="Predicted label", title="Confusion matrix (normalized)" if normalize else "Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # annotate with counts (use original cm)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{cm[i,j]}\n" + (f"{cm_norm[i,j]:.2f}" if normalize else "")
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_roc_curves(y_true, y_scores, out_path: str):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_classes = y_scores.shape[1]
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    except Exception:
        classes = sorted(list(set(y_true)))
        y_true_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(8,6))
    aucs = []
    for c in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:,c], y_scores[:,c])
            roc_auc = auc(fpr,tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, label=f"class {c} (AUC={roc_auc:.3f})")
        except Exception:
            continue
    ax.plot([0,1], [0,1], "k--", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_precision_recall_curves(y_true, y_scores, out_path: str):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_classes = y_scores.shape[1]
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    except Exception:
        classes = sorted(list(set(y_true)))
        y_true_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(8,6))
    aps = []
    for c in range(n_classes):
        try:
            prec, rec, _ = precision_recall_curve(y_true_bin[:,c], y_scores[:,c])
            ap = average_precision_score(y_true_bin[:,c], y_scores[:,c])
            aps.append(ap)
            ax.plot(rec, prec, label=f"class {c} (AP={ap:.3f})")
        except Exception:
            continue
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize="small")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_calibration_curve(y_true, y_scores, out_path: str, n_bins=10):
    # show reliability diagram averaged over classes
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_classes = y_scores.shape[1]
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    except Exception:
        classes = sorted(list(set(y_true)))
        y_true_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(6,6))
    # average prob and fraction for each class and overlay
    for c in range(n_classes):
        try:
            prob_true, prob_pred = calibration_curve(y_true_bin[:,c], y_scores[:,c], n_bins=n_bins)
            ax.plot(prob_pred, prob_true, marker='o', label=f"class {c}")
        except Exception:
            continue
    ax.plot([0,1],[0,1], "k--", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curves")
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

# -----------------------
# Orquestador
# -----------------------
def generate_report(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    output_dir: str = "reports",
    prefix: str = "report",
) -> Dict[str, str]:
    """
    Genera métricas, tablas y figuras. Devuelve diccionario con rutas de archivos generados.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, prefix)

    # compute metrics
    metrics = compute_metrics_dict(y_true, y_pred, y_scores=y_scores, target_names=target_names)
    save_json(metrics, f"{base}_metrics.json")

    # save classification table CSV
    labels = metrics.get("labels", None)
    tnames = target_names if target_names is not None else [str(l) for l in labels] if labels is not None else None
    class_csv = f"{base}_classification_report.csv"
    df_report = save_classification_table(y_true, y_pred, target_names=tnames, path_csv=class_csv)

    # confusion matrix plot (normalized)
    cm_png = f"{base}_confusion_matrix.png"
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), labels=[str(l) for l in metrics["labels"]], out_path=cm_png, normalize=True)

    results = {
        "metrics_json": f"{base}_metrics.json",
        "classification_csv": class_csv,
        "confusion_png": cm_png,
    }

    if y_scores is not None:
        y_scores = np.asarray(y_scores)
        # ROC
        roc_png = f"{base}_roc.png"
        plot_roc_curves(y_true, y_scores, roc_png)
        results["roc_png"] = roc_png
        # PR
        pr_png = f"{base}_pr.png"
        plot_precision_recall_curves(y_true, y_scores, pr_png)
        results["pr_png"] = pr_png
        # calibration
        calib_png = f"{base}_calibration.png"
        plot_calibration_curve(y_true, y_scores, calib_png)
        results["calibration_png"] = calib_png

    return results

# -----------------------
# Small utility to load CSV for CLI usage
# -----------------------
def load_csv_preds(path: str) -> Tuple[List[int], List[int], Optional[np.ndarray]]:
    """
    Espera CSV con columnas:
      - true: etiqueta verdadera (int or str)
      - pred: etiqueta predicha (int or str)
      - prob_0, prob_1, ... (opcional) -> probabilidades por clase en entero índice
    """
    df = pd.read_csv(path)
    if "true" not in df.columns or "pred" not in df.columns:
        raise ValueError("CSV must contain 'true' and 'pred' columns")
    y_true = df["true"].tolist()
    y_pred = df["pred"].tolist()
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if prob_cols:
        # order by prob_0, prob_1, ...
        prob_cols = sorted(prob_cols, key=lambda x: int(x.split("_",1)[1]))
        y_scores = df[prob_cols].values.astype(float)
    else:
        y_scores = None
    return y_true, y_pred, y_scores

# -----------------------
# CLI
# -----------------------
def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Genera reporte de métricas (CSV + plots) desde predicciones.")
    parser.add_argument("--preds_csv", type=str, help="CSV con columnas 'true','pred' y opcionales 'prob_0','prob_1',...", required=True)
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--prefix", type=str, default="report")
    args = parser.parse_args()
    y_true, y_pred, y_scores = load_csv_preds(args.preds_csv)
    results = generate_report(y_true, y_pred, y_scores=y_scores, output_dir=args.output_dir, prefix=args.prefix)
    print("Generated files:")
    for k,v in results.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    _cli()
