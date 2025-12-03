#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from zenml.integrations.bentoml.steps import (
    bento_builder_step
)


bento_builder = bento_builder_step.with_options(
    parameters=dict(
        # model=model,
        # model_name=model_name,
        model_type="pytorch",
        service="service.py:svc",
        labels={
            "framework": "pytorch",
            "dataset": "Base on x-brain",
            "zenml_version": "0.22.0",
        },
        exclude=["data", "augment"],
        docker={
            "system_packages": ["ffmpeg", "libsm6", "libxext6"],
        },
    )
)
    
