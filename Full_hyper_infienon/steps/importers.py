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
from sqlalchemy import create_engine, text


import logging
@step
def query_database(table_name: str, limit_num: int) -> pd.DataFrame:
    connection_string = 'postgresql://airflow:airflow@10.90.1.4:5433/postgres'
    engine = create_engine(connection_string)
    
    # Define the labels
    labels = ['Circle', 'Side-to-Side', 'Square', 'negative']
    # Initialize an empty list to store DataFrames
    data_frames = []
    
    # Iterate over each label
    for label in labels:
        # Use parameterized queries to avoid SQL injection
        query = text(f"SELECT * FROM {table_name} WHERE gesture = :gesture LIMIT :limit")
        
        # Execute the query with parameters
        with engine.connect() as connection:
            label_data = pd.read_sql_query(query, connection, params={"gesture": label, "limit": limit_num})
            
            # Check if data is available
            if not label_data.empty:
                data_frames.append(label_data)
            else:
                print(f"No data found for label: {label}")
    
    # Concatenate the list of DataFrames into a single DataFrame
    selected_data = pd.concat(data_frames, ignore_index=True)
    
    return selected_data


@step
def training_data_loader(table_name: str, limit_num: int, test_size: float = 0.2) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Load data from a file and split into train and test sets."""
    logging.info("Loading infineon gesture data set...")
    df = query_database(table_name, limit_num)
    df = df.drop(columns=['index'], axis=1)
    print(df.info())
    print(df.gesture.value_counts())    
    gesture_mapping = {'Circle': 1, 'Side-to-Side': 2, 'Square': 3, 'negative': 4}
    # Map the gesture labels to numerical values in the DataFrame
    df['gesture'] = df['gesture'].map(gesture_mapping)
    df.gesture.value_counts()
    
    print(df.gesture.value_counts())
    X = df.drop(columns=['gesture'], axis=1)    
    y = df['gesture']
    logging.info("Logging functions and Splitting train and test ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=42)
    
    logging.info("Data loaded and split successfully.")
    print(f"X.shape = {X_train.shape}, X_test.shape = {X_test.shape}")
    return X_train, X_test, y_train, y_test

