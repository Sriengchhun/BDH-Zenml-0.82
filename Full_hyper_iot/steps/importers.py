from typing_extensions import Annotated  # or `from typing import Annotated on Python 3.9+
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from typing import Optional
from zenml import pipeline, step
from sqlalchemy import create_engine
import steps.Cleasing_fn as cf
import psycopg2
import logging

@step
def query_database(table_name: str) -> pd.DataFrame:
    # Replace with your PostgreSQL connection details
    connection_string = 'postgresql://airflow:airflow@10.90.1.4:5433/postgres'
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
    """Load data from a file and split into train and test sets."""
    logging.info("Loading IoT data set...")
    df = query_database(table_name)
    df.columns = ['idx', '_id', 'id', 'type', 'ts', 'crt', 'mdt', 'dts',
       'temperature', 'humidity', 'lux', 'soil',
       'data_key1', 'data_key2', 'data_key3', 'data_led0', 'data_led1',
       'data_led2', 'data_led3']
    df.info()
    target_ids = [f"NEXIIOT0000{i}" for i in range(1, 7)]
    filtered_df = df[df['id'].isin(target_ids)]

    df_for_train = filtered_df[['idx', '_id', 'id', 'type', 'ts', 'crt', 'mdt', 'dts', 'temperature',
       'humidity', 'lux', 'soil']]
    print(df_for_train.isnull().sum())
    df_for_train.dropna(inplace=True)
    df_for_train.isnull().sum()
    print(df_for_train.isnull().sum())
    # df_prepared = df_for_train.copy()

    df_labeled = df_for_train.copy()
    df_labeled = cf.preprocess_and_label(df_for_train)
    print(df_labeled.shape)


    # df_labeled = df_labeled.drop(['index'], axis=1) 
    df_cls = df_labeled[df_labeled['label'].isin(['normal', 'hot', 'dry', 'dark'])].copy()

    label_map = {'normal': 1, 'hot': 2, 'dry': 3, 'dark': 4}
    # Map the gesture labels to numerical values in the DataFrame
    df_cls['label'] = df_cls['label'].map(label_map)

    X = df_cls[['temperature', 'humidity', 'lux', 'soil']]
    y = df_cls['label']

    logging.info("Logging functions and Splitting train and test ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=42)
    
    print(f"X.shape = {X_train.shape}, X_test.shape = {X_test.shape}")
    return X_train, X_test, y_train, y_test

