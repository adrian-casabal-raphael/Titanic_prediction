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
    if title not in {"Mr", "Miss", "Mrs"}:
        return "Rare"
    return title

def extract_deck(cabin: str) -> str:
    """Extract the deck letter from the Cabin field."""
    if pd.isna(cabin):
        return "Unknown"
    cabin = str(cabin).strip()
    return cabin[0] if cabin else "Unknown"

def extract_ticket_prefix(ticket: str) -> str:
    """Extract the ticket prefix from the Ticket field."""
    if pd.isna(ticket):
        return "None"
    ticket = str(ticket).replace(".", "").replace("/", "").strip()
    parts = ticket.split()
    if len(parts) <= 1:
        return "None"
    prefix = "".join([p for p in parts[:-1]])
    return prefix.upper() if prefix else "None"

def extract_surname(name: str) -> str:
    """Extract the surname from the Name field."""
    if pd.isna(name):
        return "Unknown"
    return name.split(",")[0].strip()

def add_group_age(train_df: pd.DataFrame, test_df: pd.DataFrame):
    combined = pd.concat([train_df.copy(), test_df.copy()], axis=0, ignore_index=True)
    age_medians = (combined.groupby(["Sex", "Pclass", "Title"])["Age"].median().reset_index().rename(columns={"Age": "AgeGroupMedian"}))

    combined = combined.merge(age_medians, on=["Sex", "Pclass", "Title"], how="left")
    combined["Age"] = combined["Age"].fillna(combined["AgeGroupMedian"])
    combined["Age"] = combined["Age"].fillna(combined["Age"].median())
    combined = combined.drop(columns=["AgeGroupMedian"])

    train_out = combined.iloc[:len(train_df)].copy()
    test_out = combined.iloc[len(train_df):].copy()
    return train_out, test_out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cleaned and engineered features from the raw Titanic dataframe."""
    data = df.copy()
    data["Title"] = data["Name"].apply(extract_title)
    data["Surname"] = data["Name"].apply(extract_surname)
    data["Deck"] = data["Cabin"].apply(extract_deck)
    data["TicketPrefix"] = data["Ticket"].apply(extract_ticket_prefix)

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    data["HasCabin"] = data["Cabin"].notna().astype(int)

    data["TicketGroupSize"] = data.groupby("Ticket")["Ticket"].transform("count")
    data["SurnameGroupSize"] = data.groupby("Surname")["Surname"].transform("count")
    data["FamilyGroupSize"] = data.groupby(["Surname", "Fare"])["Surname"].transform("count")

    data["FarePerPerson"] = data["Fare"] / data["FamilySize"].replace(0, 1)
    data["LogFare"] = np.log1p(data["Fare"])

    return data


def fit_clean_transform(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fit preprocessing choices on train, then apply to both train and test."""
    train_features = build_features(train_df)
    test_features = build_features(test_df)
    train_features, test_features = add_group_age(train_features, test_features)

    y_train = train_features["Survived"].copy()
    train_features = train_features.drop(columns=["Survived"])

    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    # Impute numeric/categorical columns using TRAIN columns only
    train_numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
    train_categorical_cols = train_features.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_fill_values = train_features[train_numeric_cols].median(numeric_only=True)
    for col in train_numeric_cols:
        combined[col] = combined[col].fillna(numeric_fill_values[col])

    for col in train_categorical_cols:
        mode = train_features[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Missing"
        combined[col] = combined[col].fillna(fill_value)

    # Add engineered features BEFORE encoding
    combined["IsChild"] = (combined["Age"] < 16).astype(int)

    combined["AgeBin"] = pd.cut(combined["Age"],bins=[-1, 5, 12, 18, 35, 60, 100],labels=["Toddler", "Child", "Teen", "YoungAdult", "Adult", "Senior"],)

    combined["FareBin"] = pd.qcut(combined["Fare"],q=4,labels=["Low", "MidLow", "MidHigh", "High"],duplicates="drop",)

    # Drop raw text columns after extracting useful signals
    combined = combined.drop(columns=["Name", "Ticket", "Cabin", "Surname"], errors="ignore")

    # Recompute column groups after new features were added
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = combined.select_dtypes(exclude=[np.number]).columns.tolist()

    # One-hot encode categorical features
    combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=False)
    if "Survived" in combined_encoded.columns:
        combined_encoded = combined_encoded.drop(columns=["Survived"])

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
