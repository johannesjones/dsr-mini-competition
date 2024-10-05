from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_model, evaluate_model, generate_predictions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["preprocessed_data"],
                outputs="trained_model",
                name="train_random_forest_node",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "preprocessed_data"],
                outputs="evaluation_metrics",
                name="evaluate_model_node",
            ),
            node(
                func=generate_predictions,
                inputs=[
                    "preprocessed_data",  # The test data to make predictions on
                    "trained_model",  # The trained model from your catalog
                ],
                outputs="predictions",  # Output path for predictions
                name="generate_predictions_node",
            ),
        ]
    )
