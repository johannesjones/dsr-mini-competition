from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_submission


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_submission,
                inputs=["predictions", "params:output_filepath"],
                outputs="submission",
                name="generate_submission_node",
            ),
        ]
    )
