from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "train_cleaned.csv"
TEST_PATH = DATA_DIR / "test_cleaned.csv"
RANDOM_STATE = 42
THRESHOLDS = np.arange(0.30, 0.701, 0.01)


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    y = train["Survived"].copy()
    X = train.drop(columns=["Survived"]).copy()
    test_ids = test["PassengerId"].copy()
    X_test = test.copy()

    if "PassengerId" in X.columns:
        X = X.drop(columns=["PassengerId"])
    if "PassengerId" in X_test.columns:
        X_test = X_test.drop(columns=["PassengerId"])

    # Normalize dtypes so every estimator sees the same numeric matrix.
    return X.astype(float), y, X_test.astype(float), test_ids


def build_models():
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=1500,
            max_depth=6,
            min_samples_leaf=2,
            min_samples_split=6,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=1500,
            max_depth=8,
            min_samples_split=4,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=400,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),
        "SVC": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=1.5,
                        gamma="scale",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def best_threshold(y_true, probabilities):
    best_accuracy = -1.0
    best_cutoff = 0.50
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_true, predictions)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_cutoff = float(threshold)
    return best_accuracy, best_cutoff


def evaluate_candidates(models, X, y, cv):
    probabilities_by_model = {}
    leaderboard = []

    for name, model in models.items():
        probabilities = cross_val_predict(
            clone(model),
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        probabilities_by_model[name] = probabilities
        accuracy, threshold = best_threshold(y, probabilities)
        leaderboard.append(
            {
                "name": name,
                "members": [name],
                "type": "single",
                "accuracy": accuracy,
                "threshold": threshold,
            }
        )

    model_names = list(models.keys())
    for group_size in range(2, min(4, len(model_names)) + 1):
        for member_names in combinations(model_names, group_size):
            blended_probabilities = np.mean(
                [probabilities_by_model[name] for name in member_names],
                axis=0,
            )
            accuracy, threshold = best_threshold(y, blended_probabilities)
            leaderboard.append(
                {
                    "name": " + ".join(member_names),
                    "members": list(member_names),
                    "type": "blend",
                    "accuracy": accuracy,
                    "threshold": threshold,
                }
            )

    leaderboard.sort(
        key=lambda row: (row["accuracy"], -len(row["members"]), row["name"]),
        reverse=True,
    )
    return leaderboard


def fit_and_predict(best_candidate, models, X, y, X_test):
    test_probabilities = []
    for model_name in best_candidate["members"]:
        fitted_model = clone(models[model_name])
        fitted_model.fit(X, y)
        test_probabilities.append(fitted_model.predict_proba(X_test)[:, 1])

    averaged_probabilities = np.mean(test_probabilities, axis=0)
    return (averaged_probabilities >= best_candidate["threshold"]).astype(int)


def main():
    X, y, X_test, test_ids = load_data()
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    leaderboard = evaluate_candidates(models, X, y, cv)
    best_candidate = leaderboard[0]

    print("Top cross-validated candidates (OOF accuracy with tuned threshold):")
    for row in leaderboard[:10]:
        print(
            f"{row['name']}: accuracy={row['accuracy']:.4f}, "
            f"threshold={row['threshold']:.2f}"
        )

    print(
        "\nSelected strategy: "
        f"{best_candidate['name']} "
        f"(accuracy={best_candidate['accuracy']:.4f}, "
        f"threshold={best_candidate['threshold']:.2f})"
    )

    predictions = fit_and_predict(best_candidate, models, X, y, X_test)

    submission = pd.DataFrame(
        {
            "PassengerId": test_ids,
            "Survived": predictions.astype(int),
        }
    )
    submission_path = DATA_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSaved submission file: {submission_path}")


if __name__ == "__main__":
    main()
