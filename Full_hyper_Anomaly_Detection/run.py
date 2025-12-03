import click
from typing import Optional, Union

from constants import MODEL_NAME, PIPELINE_NAME, PIPELINE_STEP_NAME  # keep if used elsewhere
from pipelines.training_model import Training_pipeline_for_Anomaly_Classification

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

# -----------------------
# Validators
# -----------------------
def _noneable_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s == "none":
        return None
    try:
        return int(s)
    except ValueError:
        raise click.BadParameter("Value must be an integer or 'none'.")

def _noneable_int_any(ctx, param, value):
    return _noneable_int(value)

def validate_max_samples(ctx, param, value):
    """
    Accepts:
      - 'auto'
      - int >= 1
      - float in (0, 1]
    """
    if value is None:
        return "auto"
    s = str(value).strip().lower()
    if s == "auto":
        return "auto"
    # try int
    try:
        iv = int(s)
        if iv < 1:
            raise click.BadParameter("max_samples int must be >= 1.")
        return iv
    except ValueError:
        pass
    # try float
    try:
        fv = float(s)
        if not (0.0 < fv <= 1.0):
            raise click.BadParameter("max_samples float must be in (0, 1].")
        return fv
    except ValueError:
        raise click.BadParameter("max_samples must be 'auto', an int >= 1, or a float in (0, 1].")

def validate_contamination(ctx, param, value):
    """
    Accepts:
      - 'auto'
      - float in (0, 0.5]
    """
    if value is None:
        return "auto"
    s = str(value).strip().lower()
    if s == "auto":
        return "auto"
    try:
        fv = float(s)
        if not (0.0 < fv <= 0.5):
            raise click.BadParameter("contamination must be in (0, 0.5] or 'auto'.")
        return fv
    except ValueError:
        raise click.BadParameter("contamination must be a float in (0, 0.5] or 'auto'.")

def validate_max_features(ctx, param, value):
    """
    Accepts:
      - int >= 1
      - float in (0, 1]
    """
    if value is None:
        return 1.0
    s = str(value).strip().lower()
    # try int
    try:
        iv = int(s)
        if iv < 1:
            raise click.BadParameter("max_features int must be >= 1.")
        return iv
    except ValueError:
        pass
    # try float
    try:
        fv = float(s)
        if not (0.0 < fv <= 1.0):
            raise click.BadParameter("max_features float must be in (0, 1].")
        return fv
    except ValueError:
        raise click.BadParameter("max_features must be an int >= 1 or a float in (0, 1].")

# -----------------------
# CLI
# -----------------------
@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default="deploy_and_predict",
    help="Run only deployment (`deploy`), only prediction (`predict`), or both (`deploy_and_predict`).",
)
@click.option(
    "--test_size",
    type=float,
    default=0.2,
    help="Test split size (fraction). Default: 0.2",
)
@click.option(
    "--table_name",
    type=str,
    default="infineon_gesture",
    help="Source table name. Default: 'infineon_gesture'.",
)
# ---- IsolationForest hyperparameters ----
@click.option(
    "--n_estimators",
    type=int,
    default=100,
    help="Number of trees for IsolationForest. Default: 100",
)
@click.option(
    "--max_samples",
    default="auto",
    callback=validate_max_samples,
    help="Samples per tree: 'auto', int>=1, or float in (0, 1]. Default: 'auto'",
)
@click.option(
    "--contamination",
    default="auto",
    callback=validate_contamination,
    help="Expected anomaly proportion: 'auto' or float in (0, 0.5]. Default: 'auto'",
)
@click.option(
    "--max_features",
    default=1.0,
    callback=validate_max_features,
    help="Features per tree: int>=1 or float in (0, 1]. Default: 1.0",
)
@click.option(
    "--bootstrap/--no-bootstrap",
    default=False,
    help="Enable bootstrap sampling per tree. Default: False",
)
@click.option(
    "--n_jobs",
    default=None,
    callback=_noneable_int_any,
    help="Parallel jobs (int) or 'none' for single-thread. Default: none",
)
@click.option(
    "--random_state",
    default=None,
    callback=_noneable_int_any,
    help="Random seed (int) or 'none'. Default: none",
)
@click.option(
    "--verbose",
    type=int,
    default=0,
    help="scikit-learn verbosity. Default: 0",
)
def main(
    config: str,
    test_size: float,
    table_name: str,
    n_estimators: int,
    max_samples: Union[int, float, str],
    contamination: Union[float, str],
    max_features: Union[int, float],
    bootstrap: bool,
    n_jobs: Optional[int],
    random_state: Optional[int],
    verbose: int,
):
    deploy = config in (DEPLOY, DEPLOY_AND_PREDICT)
    predict = config in (PREDICT, DEPLOY_AND_PREDICT)

    print(
        f"config={config}, test_size={test_size}, table_name={table_name}, "
        f"n_estimators={n_estimators}, max_samples={max_samples}, contamination={contamination}, "
        f"max_features={max_features}, bootstrap={bootstrap}, n_jobs={n_jobs}, "
        f"random_state={random_state}, verbose={verbose}"
    )

    if deploy:
        Training_pipeline_for_Anomaly_Classification(
            table_name=table_name,
            test_size=test_size,
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    if predict:
        print("Predicting...")

if __name__ == "__main__":
    main()




# python run.py --config deploy --table_name nexpie_sensors --contamination 0.05 --n_estimators 200 --n_jobs -1
# python run.py --config deploy --table_name nexpie_sensors --contamination 0.05 --n_estimators 200 --n_jobs -1
