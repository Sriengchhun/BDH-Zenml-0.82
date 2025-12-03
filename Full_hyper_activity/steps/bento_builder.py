from constants import MODEL_NAME
from zenml import __version__ as zenml_version
from zenml.integrations.bentoml.steps import bento_builder_step

bento_builder = bento_builder_step.with_options(
    parameters=dict(
        model_name=MODEL_NAME,
        model_type="sklearn",
        service="service.py:svc",
        labels={
            "framework": "sklearn",
            "dataset": "Infineon_Gesture",
            "zenml_version": zenml_version,
        },
        exclude=["data"],
        python={
            "packages": ["zenml", "scikit-learn"],
        },
    )
)
