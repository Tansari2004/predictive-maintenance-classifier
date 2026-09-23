import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import FEATURES, TARGET, fetch_data


RANDOM_STATE = 42


def build_model() -> Pipeline:
    numeric = FEATURES[1:]
    categorical = [FEATURES[0]]
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler())]),
                numeric,
            ),
            (
                "category",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )


def select_threshold(y_true, probabilities, max_false_positive_rate: float = 0.10) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    best = None
    for threshold in candidates:
        predictions = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if fp + tn else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        if fpr <= max_false_positive_rate:
            score = (recall, precision, threshold)
            if best is None or score > best[0]:
                best = (score, threshold)
    return float(best[1] if best else 0.5)


def metrics_at_threshold(y_true, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    frame = fetch_data()
    X_train, X_test, y_train, y_test = train_test_split(
        frame[FEATURES],
        frame[TARGET],
        test_size=0.20,
        stratify=frame[TARGET],
        random_state=RANDOM_STATE,
    )

    model = build_model()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probabilities = cross_val_predict(
        model, X_train, y_train, cv=folds, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = select_threshold(y_train, oof_probabilities)

    model.fit(X_train, y_train)
    test_probabilities = model.predict_proba(X_test)[:, 1]
    metrics = metrics_at_threshold(y_test, test_probabilities, threshold)
    metrics.update(
        {
            "dataset_rows": int(len(frame)),
            "test_rows": int(len(y_test)),
            "test_failures": int(y_test.sum()),
            "random_state": RANDOM_STATE,
            "model": "class-weighted logistic regression",
        }
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    predictions = test_probabilities >= threshold
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=["No failure", "Failure"])
    plt.title(f"Holdout confusion matrix (threshold={threshold:.2f})")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
