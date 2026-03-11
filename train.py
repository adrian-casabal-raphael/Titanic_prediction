
from pathlib import Path

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "train_cleaned.csv"
TEST_PATH = DATA_DIR / "test_cleaned.csv"

def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Split features and target
    y = train["Survived"]
    X = train.drop(columns=["Survived"])

    # Do not use PassengerId as a predictive feature
    if "PassengerId" in X.columns:
        X = X.drop(columns=["PassengerId"])
    if "PassengerId" in test.columns:
        X_test = test.drop(columns=["PassengerId"])
    else:
        X_test = test.copy()

    # Try a few baseline models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVC": SVC(kernel="rbf", C=1.0, gamma="scale"),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("5-Fold Cross-Validation Accuracy:")
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        results[name] = scores.mean()
        print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}, folds={scores}")

    # Choose the best model based on CV
    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name}")

    best_model = models[best_name]
    best_model.fit(X, y)

    # Predict on Kaggle test data
    predictions = best_model.predict(X_test).astype(int)

    # Build submission file
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    submission_path = DATA_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSaved submission file: {submission_path}")

if __name__ == "__main__":
    main()
