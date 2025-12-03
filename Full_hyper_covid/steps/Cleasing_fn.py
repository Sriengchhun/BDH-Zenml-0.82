## Import library
import numpy as np
import pandas as pd


# Function for drop the color that not required in badgecolor and convert obj type to float type.
def Drop_unrequired_Color(data):
    new_df = data.copy()
    new_df=new_df.drop(new_df[(new_df['badgeColor'] == 0) | (new_df['badgeColor'] == 1) |
                              (new_df['badgeColor'] == 4) | (new_df['badgeColor'] == 5) |
                              (new_df['badgeColor'] == 6) | (new_df['badgeColor'] == 7)
                             ].index)
    return new_df

def Convert_obj(data):
    
    data=data.drop(data[(data['gender']=='other') | \
                                 (data['gender']=='undefined')].index)

    data = data.replace(to_replace=['none','stable', 'decrease','increase','male', 'female','ชาย','หญิง'],\
                            value=[1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 0.0, 1.0])
    data = data.dropna()
    new_df = data.copy()
    return new_df




# Extract diseases to become new columns
def extract_feature(data):
    df = data.copy()
    df = df.dropna()
    # Convert str to list 
    df["list_diseases"] = df["diseases"].apply(eval)
    df['HT_d'] = df['list_diseases'].apply(lambda x: 1 if "HT" in x else 0)
    df['BW_d'] = df['list_diseases'].apply(lambda x: 1 if "BW>90" in x else 0)
    df['DM_d'] = df['list_diseases'].apply(lambda x: 1 if "DM" in x else 0)
    df['DLP_d'] = df['list_diseases'].apply(lambda x: 1 if "DLP" in x else 0)
    df['Obesity'] = df['list_diseases'].apply(lambda x: 1 if "Obesity" in x else 0)
    df['HIV'] = df['list_diseases'].apply(lambda x: 1 if "HIV" in x else 0)
    df['HTN'] = df['list_diseases'].apply(lambda x: 1 if "HTN" in x else 0)
    df['hypothyroid'] = df['list_diseases'].apply(lambda x: 1 if "hypothyroid" in x else 0)
    df['PhumPer'] = df['list_diseases'].apply(lambda x: 1 if "ภูมิแพ้อากาศ" in x else 0)
    return df.copy()

# Recolumn to be perfect for train the model
def Prepare_columns(data):
    data = data[['gender','age','bodytemp','oxygen','pulse','dyspnea','cough','diarrhea','fever','runnyNose','smell',
            'soreThroat','HT_d','BW_d','DM_d','DLP_d','HIV','hypothyroid','PhumPer', 'badgeColor']]
    return data

# Prepare X and y for next split step
def X_and_y(data):
    X = data[['gender','age','bodytemp','oxygen','pulse','dyspnea','cough','diarrhea','fever','runnyNose','smell',
                'soreThroat','HT_d','BW_d','DM_d','DLP_d','HIV','hypothyroid','PhumPer']]
    y = data['badgeColor']
    y = y.replace(to_replace=[2,3],value=[0,1])
    
    return X, y

def print_X_y(X, y):
    print(X.shape)
    print(y.shape)


if __name__ == '__main__':
    print("Done")