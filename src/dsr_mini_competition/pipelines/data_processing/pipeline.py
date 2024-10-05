from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_data, add_type_information


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=add_type_information,
                inputs=["train_values", "test_values"],
                outputs="values",
                name="values_with_type_information",
            ),
            node(
                func=preprocess_data,
                inputs=["values", "train_labels"],
                outputs="preprocessed_data",
                name="preprocess_data_node",
            ),
        ]
    )
