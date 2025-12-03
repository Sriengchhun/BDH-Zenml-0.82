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
import psycopg2
import logging

@step
def query_database(table_name: str, limit_num: int) -> pd.DataFrame:
    conn = psycopg2.connect(database="postgres",
                        host="10.115.1.18",
                        user="airflow",
                        password="airflow",
                        port="5433")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit_num}")

    tuples_list = cursor.fetchall()
    result = pd.DataFrame(tuples_list)  
    print(f"result.shape = {result.shape}")
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
    print(df.info())
    # Preprocessing steps from the original code
    df.drop([0], axis=1, inplace=True)
    df.rename(columns ={1:'Heart',2:'Calories',3:'Steps',4:'Distance',5:'Age',6:'Gender',7:'Weight',8:'Height',9:'Activity'}, inplace=True)
    
    df.drop(['Height'],axis=1,inplace=True)
    df['Activity'] = df['Activity'].replace("0.Sleep", 0)
    df['Activity'] = df['Activity'].replace("1.Sedentary", 1)
    df['Activity'] = df['Activity'].replace("2.Light", 2)
    df['Activity'] = df['Activity'].replace("3.Moderate", 3)
    df['Activity'] = df['Activity'].replace("4.Vigorous", 4)
    df['Gender'] = df['Gender'].replace("M", 0)
    df['Gender'] = df['Gender'].replace("F", 1)
    #drop values
    df.drop(df[df.Activity > 1].index, inplace=True)
    X=df.drop('Activity',axis=1)
    y=df['Activity']
    
    logging.info("Splitting train and test based on 80:20 ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=42)
    
    print(f"X.shape = {X_train.shape}, X_test.shape = {X_test.shape}")
    return X_train, X_test, y_train, y_test