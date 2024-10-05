import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    StackingClassifier
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder


def create_stacked_model() -> StackingClassifier:
    """
    Create the stacking classifier model using LGBM, Random Forest, and AdaBoost classifiers.
    """
    # Define the base models
    lgbm_model = XGBClassifier(
        n_estimators=350, learning_rate=0.1, max_bin=95, max_depth=30, num_leaves=200
    )
    rf_model = RandomForestClassifier()
    ada_model = AdaBoostClassifier()

    # Define the logistic regression meta-model
    logreg_model = LogisticRegression()

    # Create and return the stacking classifier
    return StackingClassifier(
        estimators=[("lgbm", lgbm_model), ("rf", rf_model), ("ada", ada_model)],
        final_estimator=logreg_model,
    )


def train_model(
    preprocessed_data: pd.DataFrame
) -> StackingClassifier:
    """
    Train the stacking classifier on the training data.

    Args:
        stacked_model: The stacking classifier.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained stacking model.
    """
    stacked_model = create_stacked_model()
    X_train = preprocessed_data[preprocessed_data["type"] == "TRAIN"].drop(columns=["type", "building_id", "damage_grade"])
    y_train = preprocessed_data[preprocessed_data["type"] == "TRAIN"]["damage_grade"]

    stacked_pipe = Pipeline([("stacked_model", stacked_model)])
    stacked_pipe.fit(X_train, y_train)
    return stacked_pipe


# Node 5: Model Evaluation
def evaluate_model(model, preprocessed_data: pd.DataFrame) -> dict:
    """Evaluate the model on the test set."""
    X_test = preprocessed_data[preprocessed_data["type"] == "TEST"].drop(
        columns=["type", "building_id", "damage_grade"]
    )
    y_test = preprocessed_data[preprocessed_data["type"] == "TEST"]["damage_grade"]

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    # Create a dictionary of metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame([metrics])  # Convert dict to DataFrame

    return metrics_df  # Return the DataFrame instead of the dict


# Node 6: Generate Predictions
def generate_predictions(model_input_data: pd.DataFrame, model) -> pd.DataFrame:
    print("Input Data Shape:", model_input_data.shape)
    print("Input Data Columns:", model_input_data.columns)

    # Separate TRAIN and TEST data
    train_data = model_input_data[model_input_data["type"] == "TRAIN"]
    test_data = model_input_data[model_input_data["type"] == "TEST"]

    print("TRAIN Data Shape:", train_data.shape)
    print("TEST Data Shape:", test_data.shape)

    if test_data.empty:
        print("Warning: No TEST data found. No predictions can be made.")
        return pd.DataFrame(columns=["building_id", "damage_grade"])

    # Remove 'type' column
    test_data = test_data.drop(columns=["type"])

    # Store 'building_id' before dropping it for prediction
    building_ids = test_data["building_id"].values

    # Remove 'building_id' for prediction
    features = test_data.drop(columns=["building_id", "damage_grade"], errors="ignore")

    # Generate predictions
    predictions = model.predict(features)

    # Create a DataFrame with 'building_id' and 'damage_grade'
    output_df = pd.DataFrame(
        {"building_id": building_ids, "damage_grade": predictions.round().astype(int)}
    )

    return output_df
