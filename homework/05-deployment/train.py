##### goal: train a logistic regression classifier for churn prediction #####

# dependencies
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# paths
PATH_DATA = Path("../data/bank_marketing/bank/bank-full.csv")

# parameters

# FIXME: add the parameters here
C

# data preparation

# read data
# FIXME: actually, we need the CHURN data
data_churn_full = pd.read_csv(PATH_DATA, sep=";")

# not all columns are used in exercise
# select those that are

# make list containing used columns

# numerical columns
numerical = ['tenure', 'monthlycharges', 'totalcharges']

# categorical columns
categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

# select features from full data
X = data_churn_full[feature_columns]

# select labels from full data
y = data_churn_full["y"]

# get full train and test sets from full data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# get train and validation sets from full train set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# apply one hot encoding

# done like this when
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)


# has to be applied to
X_train
X_val
X_test






# training


# trainig the final model


# save the model



