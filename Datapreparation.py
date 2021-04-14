import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split

# data = pd.read_csv('B:\\Desktop\\Studium Gruppenarbeiten\\LabellingWithSnorkel\\weatherAUS.csv')
# data.dropna()
# data["RainToday"] = data["RainToday"].map({"Yes":1,"No":0})
# data["RainTomorrow"]= data["RainTomorrow"].map({"Yes":1,"No":0})
# data.corr()

def load_dataset(load_train_labels: bool = False):

    df = pd.read_csv('B:\\Desktop\\Studium Gruppenarbeiten\\LabellingWithSnorkel\\weatherAUS.csv')
    df= df.dropna()
    df["RainToday"] = df["RainToday"].map({"Yes":1,"No":0})
    df["RainTomorrow"]= df["RainTomorrow"].map({"Yes":1,"No":0})
    # Lowercase column names
    df.columns = map(str.lower, df.columns)
    # Remove several fields
    df = df.drop("location", axis=1)
    df = df.drop("date", axis=1)
    df = df.rename(columns={"raintomorrow": "label"})
    # Shuffle order
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    df_train =df

    if not load_train_labels:
        df_train, df_test = train_test_split(df_train, test_size=1250, random_state=123, stratify=df_train.label)
        df_train["label"] = np.ones(len(df_train["label"])) * -1
        #df_train.loc[:,"label"]= -1
        
    return df_train, df_test, df