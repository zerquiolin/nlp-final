from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


def train_tfidf_logreg(
    train_texts, train_labels, max_features=20000, cv: int = 3
) -> Tuple[TfidfVectorizer, LogisticRegression, List[dict]]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(X_train, train_labels)

    # learning curve computed with pipeline to re-fit vectorizer per split
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)),
        LogisticRegression(max_iter=200, n_jobs=-1),
    )
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        train_texts,
        train_labels,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    )
    learning_curve_points = []
    for size, tr_scores, v_scores in zip(train_sizes, train_scores, val_scores):
        learning_curve_points.append(
            {
                "train_size": int(size),
                "train_accuracy_mean": tr_scores.mean().item(),
                "val_accuracy_mean": v_scores.mean().item(),
            }
        )

    return vectorizer, clf, learning_curve_points


def eval_tfidf_model(vectorizer, model, texts, labels, target_names=None):
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    report = classification_report(
        labels, preds, target_names=target_names, output_dict=True
    )
    return preds, report


def train_tfidf_svm(train_texts, train_labels, max_features=20000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(train_texts)
    clf = LinearSVC()
    clf.fit(X_train, train_labels)

    # learning curve computed with pipeline to re-fit vectorizer per split
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)),
        LinearSVC(),
    )
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        train_texts,
        train_labels,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    )
    learning_curve_points = []
    for size, tr_scores, v_scores in zip(train_sizes, train_scores, val_scores):
        learning_curve_points.append(
            {
                "train_size": int(size),
                "train_accuracy_mean": float(tr_scores.mean()),
                "val_accuracy_mean": float(v_scores.mean()),
            }
        )

    return vectorizer, clf, learning_curve_points
