from dsr_mini_competition.pipelines.data_processing import (
    create_pipeline as dp_pipeline,
)
from dsr_mini_competition.pipelines.data_science import (
    create_pipeline as ds_pipeline,
)

from dsr_mini_competition.pipelines.submission import (
    create_pipeline as sub_pipeline,
)


def register_pipelines() -> dict:
    return {
        "__default__": dp_pipeline() + ds_pipeline(),
        "data_processing": dp_pipeline(),
        "data_science": ds_pipeline(),
        "submission": sub_pipeline(),
    }
