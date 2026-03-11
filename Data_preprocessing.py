import pandas as pd
import numpy as np
from pathlib import Path


def extract_title(name: str) -> str:
    """Extract a passenger title from the Name field."""
    if pd.isna(name):
        return "Unknown"
    if "," in name and "." in name:
        title = name.split(",", 1)[1].split(".", 1)[0].strip()
    else:
        return "Unknown"

    rare_titles = {
        "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
        "Rev", "Sir", "Jonkheer", "Dona"
    }
    if title in rare_titles:
        return "Rare"
    if title in {"Mlle", "Ms"}:
        return "Miss"
    if title == "Mme":
        return "Mrs"
    return title


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cleaned and engineered features from the raw Titanic dataframe."""
    data = df.copy()

    # Basic engineered features
    data["Title"] = data["Name"].apply(extract_title)
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    data["HasCabin"] = data["Cabin"].notna().astype(int)
    data["TicketGroupSize"] = data.groupby("Ticket")["Ticket"].transform("count")

    # Drop high-cardinality / raw text columns after extracting simpler signals
    data = data.drop(columns=["Name", "Ticket", "Cabin"])

    return data


def fit_clean_transform(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fit preprocessing choices on train, then apply to both train and test."""
    train_features = build_features(train_df)
    test_features = build_features(test_df)

    y_train = train_features["Survived"].copy()
    train_features = train_features.drop(columns=["Survived"])

    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    # Impute numeric columns with median
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = combined.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_fill_values = train_features[numeric_cols].median(numeric_only=True)
    for col in numeric_cols:
        combined[col] = combined[col].fillna(numeric_fill_values[col])

    # Impute categorical columns with train-mode
    for col in categorical_cols:
        mode = train_features[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Missing"
        combined[col] = combined[col].fillna(fill_value)

    # One-hot encode categorical features
    combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=False)

    cleaned_train = combined_encoded.iloc[: len(train_features)].copy()
    cleaned_test = combined_encoded.iloc[len(train_features):].copy()
    cleaned_train.insert(0, "Survived", y_train.values)

    preprocessing_summary = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "train_shape_after_encoding": cleaned_train.shape,
        "test_shape_after_encoding": cleaned_test.shape,
        "missing_values_remaining_train": int(cleaned_train.isna().sum().sum()),
        "missing_values_remaining_test": int(cleaned_test.isna().sum().sum()),
    }

    return cleaned_train, cleaned_test, preprocessing_summary


def main():
    script_dir = Path(__file__).resolve().parent
    candidate_dirs = [script_dir, script_dir / "data", Path.cwd(), Path.cwd() / "data"]

    train_path = None
    test_path = None
    for directory in candidate_dirs:
        candidate_train = directory / "train.csv"
        candidate_test = directory / "test.csv"
        if candidate_train.exists() and candidate_test.exists():
            train_path = candidate_train
            test_path = candidate_test
            break

    if train_path is None or test_path is None:
        searched = ", ".join(str(path) for path in candidate_dirs)
        raise FileNotFoundError(
            "Could not find train.csv and test.csv. "
            f"Searched in: {searched}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    cleaned_train, cleaned_test, summary = fit_clean_transform(train_df, test_df)

    output_dir = script_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_train.to_csv(output_dir / "train_cleaned.csv", index=False)
    cleaned_test.to_csv(output_dir / "test_cleaned.csv", index=False)

    print("Titanic data cleaning + transformation complete.\n")
    print("Files created:")
    print(f"- {output_dir / 'train_cleaned.csv'}")
    print(f"- {output_dir / 'test_cleaned.csv'}\n")
    print("Summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
