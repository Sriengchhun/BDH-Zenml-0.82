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
def query_database(table_name: str, limit_num: int) -> pd.DataFrame:
    # Replace with your PostgreSQL connection details
    connection_string = 'postgresql://airflow:airflow@10.90.1.4:5433/postgres'
    engine = create_engine(connection_string)
    query = f"SELECT * FROM {table_name} LIMIT {limit_num}"
    with engine.connect() as connection:
        result = pd.read_sql_query(query, connection)
    return result

@step
def training_data_loader(table_name: str, limit_num: int, test_size: float = 0.2) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Load data from a file and split into train and test sets."""
    logging.info("Loading Covid 19 data set...")
    df = query_database(table_name, limit_num)
    df = df.drop(['index'], axis=1) 
    df.rename(columns={"badgecolor": "badgeColor", "runnynose": "runnyNose", "sorethroat":"soreThroat"}, inplace=True)
    print(df.info())  
    dataset = cf.Drop_unrequired_Color(df)
    dataset = cf.Convert_obj(dataset)
    dataset = cf.extract_feature(dataset)
    dataset = cf.Prepare_columns(dataset)
    
    X, y = cf.X_and_y(dataset)

    logging.info("Logging functions and Splitting train and test ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=42)
    
    print(f"X.shape = {X_train.shape}, X_test.shape = {X_test.shape}")
    return X_train, X_test, y_train, y_test

