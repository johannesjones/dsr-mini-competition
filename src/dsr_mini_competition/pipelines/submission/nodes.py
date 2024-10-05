import pandas as pd


def generate_submission(predictions: pd.DataFrame, output_filepath: str) -> None:
    """
    Generate the submission CSV file for the Nepal Earthquake Damage Assessment competition.

    Args:
        predictions (pd.DataFrame): A DataFrame containing `building_id` and `damage_grade`.
        output_filepath (str): The file path where the submission CSV file will be saved.
    """
    # Ensure damage_grade is an integer type without decimal points
    predictions["damage_grade"] = predictions["damage_grade"].astype(int)

    # Validate that damage_grade contains only 1, 2, or 3
    if not predictions["damage_grade"].isin([1, 2, 3]).all():
        raise ValueError("Damage grade values must be either 1, 2, or 3.")

    # Ensure the columns are in the correct order
    submission_df = predictions[["building_id", "damage_grade"]]

    return submission_df
