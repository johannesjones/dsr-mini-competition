import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def add_type_information(
    train_values: pd.DataFrame, test_values: pd.DataFrame
) -> pd.DataFrame:
    train_values["type"] = "TRAIN"
    test_values["type"] = "TEST"
    return pd.concat([train_values, test_values], axis=0)


def preprocess_data(df: pd.DataFrame, df_labels: pd.DataFrame = None) -> pd.DataFrame:
    """Merge datasets and preprocess features."""
    # Check for 'building_id' in the main dataframe
    if "building_id" not in df.columns:
        raise ValueError("Missing 'building_id' column in the main dataset.")

    # Merge with labels only if df_labels is provided (for TRAIN data)
    if df_labels is not None:
        if (
            "building_id" not in df_labels.columns
            or "damage_grade" not in df_labels.columns
        ):
            raise ValueError(
                "Missing 'building_id' or 'damage_grade' column in the labels dataset."
            )
        df = df.merge(
            df_labels[["building_id", "damage_grade"]], on="building_id", how="left"
        )

    # Fill missing values
    df.fillna(0, inplace=True)

    # Identify categorical columns for encoding
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    categorical_cols = [
        x for x in categorical_cols if x not in ["type", "damage_grade"]
    ]

    # Preprocess the features
    processed_features = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return processed_features
