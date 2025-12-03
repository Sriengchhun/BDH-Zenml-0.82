## Import library
import numpy as np
import pandas as pd


# Function for drop the color that not required in badgecolor and convert obj type to float type.

def preprocess_and_label(df):
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

def print_X_y(X, y):
    print(X.shape)
    print(y.shape)


if __name__ == '__main__':
    print("Done")