import click
from constants import MODEL_NAME, PIPELINE_NAME, PIPELINE_STEP_NAME
from pipelines.training_model import Training_pipeline_for_Cultivation_Condition_Classification
import argparse
from typing import Optional, Union

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


def validate_max_depth(ctx, param, value):
    if value is None:
        return None
    elif value.lower() == 'none':
        return None
    try:
        depth = int(value)
        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")
        return depth
    except ValueError:
        raise click.BadParameter("Depth must be 'none' or a non-negative integer.")


def validate_for_int_and_float(ctx, param, value):
    try:
        # Attempt to convert the value to an integer
        value = int(value)
    except ValueError:
        try:
            # If it fails, attempt to convert the value to a float
            value = float(value)
        except ValueError:
            # If both conversions fail, raise an error
            raise click.BadParameter("Value must be an integer or a float.")
    return value

def validate_max_features(ctx, param, value):
    if value is None:
        return None
    elif value.lower() == 'none':
        return None
    elif value.lower() == 'sqrt':
        return 'sqrt'
    try:
        # Attempt to convert to float first (to allow for float values)
        feature_value = float(value)
        if feature_value > 0:
            return feature_value
        else:
            raise ValueError("max_features must be a positive number or 'sqrt' or 'None'.")
    except ValueError:
        raise click.BadParameter("max_features must be 'sqrt', 'None', or a positive number.")


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
    "(`deploy_and_predict`)."
)
## Parameters for  Train the model
@click.option(
    "--test_size",
    type=float,
    default=0.2,
    help="Specify the test size. Default is 0.2.",
)
@click.option(
    "--select_model",
    type=str,
    default="Random_Forest",
    help="Specify the model to use. Default is 'Random_Forest'.",
)
@click.option(
    "--table_name",
    type=str,
    default="infineon_gesture",
    help="Specify the table name. Default is 'wesafe_ai'.",
)

## Hyper-parameter for Decision Tree model
@click.option(
    "--max_depth_dtc",
    type=str,
    default=None,
    callback=validate_max_depth,
    help="Specify the maximum depth for Decision Tree. Default is None.",
)

@click.option(
    "--min_samples_split",
    type = str,
    default=2,
    callback=validate_for_int_and_float,
    help="The minimum number of samples required to split an internal node. Default is 2. parameter of DecisionTreeClassifier must be an int in the range [2, inf) or a float in the range (0.0, 1.0] ",
)

@click.option(
    "--min_samples_leaf",
    type = str,
    default=1,
    callback=validate_for_int_and_float,
    help="The minimum number of samples required to be at a leaf node. Default is 1.",
)

@click.option(
    "--max_features_dtc",
    type=str,
    default=None,
    callback=validate_max_features,
    help="Specify the max features for Random-Forest. Default is None.",
)
@click.option(
    "--criterion",
    type=str,
    default="gini",
    help="Specify the criterion for Decision Tree. Default is 'gini'.",
)

## Random-Forest
@click.option(
    "--max_depth_rf",
    type=str,
    default=None,
    callback=validate_max_depth,
    help="Specify the maximum depth for Random-Forest. Default is None.",
)
@click.option(
    "--n_estimators_rf",
    type=int,
    default=100,
    help="Specify the number of estimators for Random Forest. Default is 100.",
)
@click.option(
    "--max_features_rf",
    type=str,
    default="sqrt",
    callback=validate_max_features,
    help="Specify the max features for Random-Forest. Default is None.",
)

## Gradient Boosting
@click.option(
    "--n_estimators_gb",
    type=int,
    default=100,
    help="Specify the number of estimators for Random Forest. Default is 100.",
)
@click.option(
    "--max_depth_gb",
    type=int,
    default=3,
    help="Specify the maximum depth for Gradient Boosting. Default is 3.",
)
@click.option(
    "--learning_rate",
    type=float,
    default=0.1,
    help="Specify the learning rate for Gradient Boosting. Default is 0.1.",
)
@click.option(
    "--subsample",
    type=float,
    default=1.0,
    help="Specify the subsample for Gradient Boosting. Default is 1.0",
)
@click.option(
    "--criterion_gb",
    type=str,
    default="squared_error",
    help="Specify the criterion for Gradient_boosting. Default is 'squared_error'.",
)

def main(
    config: str,
    test_size: float,  
    select_model: str,
    table_name: str,
    max_depth_dtc,
    min_samples_split,
    min_samples_leaf,
    max_features_dtc,
    max_depth_rf,
    n_estimators_rf: int,
    max_features_rf,
    n_estimators_gb,
    max_depth_gb,
    learning_rate: float,
    subsample: float,
    criterion: str,
    criterion_gb: str,
    # loss_gb: str,
):
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    print(f'config= {config}, test_size= {test_size}, select_model= {select_model}, table_name= {table_name},  max_depth_dtc= {max_depth_dtc}, max_depth_rf= {max_depth_rf}, n_estimators_rf= {n_estimators_rf}, max_depth_gb= {max_depth_gb}, learning_rate= {learning_rate}, criterion= {criterion}')

    if deploy:
        Training_pipeline_for_Cultivation_Condition_Classification(
            test_size=test_size, 
            selected_model=select_model, 
            table_name=table_name, 
            max_depth_dtc=max_depth_dtc,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features_dtc=max_features_dtc,
            max_depth_rf=max_depth_rf,
            n_estimators_rf=n_estimators_rf,
            max_features_rf=max_features_rf,
            n_estimators_gb=n_estimators_gb,
            max_depth_gb=max_depth_gb,
            learning_rate=learning_rate,
            subsample=subsample,
            criterion=criterion,
            criterion_gb=criterion_gb,
            # loss=loss_gb
        )
    if predict:
        print("Predicting...")

if __name__ == "__main__":
    main()
