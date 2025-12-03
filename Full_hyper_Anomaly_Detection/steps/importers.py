from typing_extensions import Annotated  # or `from typing import Annotated on Python 3.9+
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from typing import Optional
from zenml import pipeline, step
from sqlalchemy import create_engine
import steps.Cleasing_fn as cf
import psycopg2
import logging
from sklearn.preprocessing import RobustScaler


# --- your function pasted here (import from your module if you have one) ---
def preprocess_and_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    soil_thresh = df['soil'].quantile(0.05)
    lux_thresh = df['lux'].quantile(0.05)

    df['anomaly'] = df.apply(
        lambda row: 1 if row['temperature'] > 45 or row['soil'] < soil_thresh or row['lux'] < lux_thresh else 0, axis=1
    )
    df['label'] = df.apply(
        lambda row: 'hot' if row['temperature'] > 45 else
                    'dry' if row['soil'] < soil_thresh else
                    'dark' if row['lux'] < lux_thresh else
                    'normal',
        axis=1
    )
    df['recommend'] = df.apply(
        lambda row: 'water' if row['soil'] < soil_thresh and row['temperature'] > 40 else 'none',
        axis=1
    )
    return df

# --- helpers ---
def _ensure_datetime(series: pd.Series, colname: str = "ts") -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", utc=True)
    except Exception:
        logging.warning(f"Could not parse {colname} to datetime; leaving as-is.")
        return series


@step
def query_database(table_name: str) -> pd.DataFrame:
    # Replace with your PostgreSQL connection details
    connection_string = 'postgresql://airflow:airflow@10.115.1.18:5433/postgres'
    engine = create_engine(connection_string)
    query = f"SELECT * FROM {table_name}"
    with engine.connect() as connection:
        result = pd.read_sql_query(query, connection)
    return result

@step
def training_data_loader(table_name: str, test_size: float = 0.2) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    logging.info("Loading IoT dataset...")
    # df = query_database(table_name)
    df = pd.read_csv("steps/df_AIoT_Sensor.csv")

    # Normalize columns
    df.columns = ['idx', '_id', 'id', 'type', 'ts', 'crt', 'mdt', 'dts',
                  'temperature', 'humidity', 'lux', 'soil',
                  'data_key1', 'data_key2', 'data_key3', 'data_led0', 'data_led1',
                  'data_led2', 'data_led3']

    # Filter to target devices
    target_ids = [f"NEXIIOT0000{i}" for i in range(1, 7)]
    df = df[df['id'].isin(target_ids)].copy()
    if df.empty:
        raise ValueError("No rows left after ID filtering. Check target_ids or table content.")

    # Keep needed cols and clean
    cols = ['ts', 'id', 'temperature', 'humidity', 'lux', 'soil']
    df = df[cols].copy()
    logging.info(f"Missing before dropna:\n{df.isna().sum()}")
    df = df.dropna(subset=['ts', 'temperature', 'humidity', 'lux', 'soil']).drop_duplicates()

    # Parse timestamp & sort (prevents leakage)
    df['ts'] = _ensure_datetime(df['ts'], 'ts')
    df = df[df['ts'].notna()].sort_values('ts').reset_index(drop=True)

    # Label using your logic
    df = preprocess_and_label(df)

    # Ground truth for evaluation
    y_all = df['anomaly'].astype(int).clip(0, 1)

    # Features
    feature_cols = ['temperature', 'humidity', 'lux', 'soil']
    X_all = df[feature_cols].astype(float)

    # Time-aware split
    n = len(df)
    if n < 10:
        logging.warning(f"Very small dataset (n={n}). Consider collecting more data.")
    split_idx = int((1.0 - test_size) * n)
    split_idx = max(1, min(split_idx, n - 1))

    X_train_full = X_all.iloc[:split_idx].copy()
    y_train_full = y_all.iloc[:split_idx].copy()
    X_test = X_all.iloc[split_idx:].copy()
    y_test = y_all.iloc[split_idx:].copy()

    # Normal-only training set for IsolationForest
    normal_mask = (y_train_full == 0)
    if normal_mask.sum() == 0:
        logging.warning("No NORMAL rows in training split; using full train for scaler (suboptimal).")
        X_train_norm = X_train_full
        y_train = y_train_full  # not used by model; returned for completeness
    else:
        X_train_norm = X_train_full[normal_mask].copy()
        y_train = y_train_full[normal_mask].copy()  # all zeros

    # Robust scaling (fit on train-normal only)
    scaler = RobustScaler()
    scaler.fit(X_train_norm)

    X_train = pd.DataFrame(
        scaler.transform(X_train_norm),
        columns=feature_cols,
        index=X_train_norm.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    logging.info(
        f"Prepared for IsolationForest -> "
        f"X_train={X_train.shape} (normal-only), X_test={X_test.shape}, "
        f"train_norm={normal_mask.sum()}/{len(y_train_full)}, test_anom={(y_test==1).sum()}"
    )

    return X_train, X_test, y_train, y_test