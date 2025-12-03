import click
from constants import MODEL_NAME, PIPELINE_NAME, PIPELINE_STEP_NAME
from pipelines.training_model import Deploy_Trained_Infineon_model_pipeline
import argparse


DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default="deploy_and_predict",
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model (`deploy`), or to "
    "only run a prediction against the deployed model "
    "(`predict`). By default both will be run "
    "(`deploy_and_predict`).",
)

@click.option(
    "--test_size",
    type=float,
    default=0.2,
    help="Specify the test size. Default is 0.2.",
)

@click.option(
    "--model_name",
    type=str,
    help="Specify the trained model name.",
)


def main(
    config: str,
    test_size: float,  
    model_name: str,
):
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:

        Deploy_Trained_Infineon_model_pipeline(test_size=test_size, model_name=model_name)
    if predict:
        print("Predicting...")


if __name__ == "__main__":
    main()



##{ "gender": "male", "age": 20, "diseases": ["MDD"], "bodyTemp": 35, "preSpO2": 98, "prePR": 102, "preDyspnea": 5, "fever": "none", "cough": "none", "runnyNose": "none", "soreThroat": "none", "smell": "stable", "diarrhea": "none" }

#  python run.py --test_size 0.2 --select_model Decision_Tree --model_name Trained_model_2024-06-24.pkl
# python run.py --test_size 0.2 --model_name Infineon_Model