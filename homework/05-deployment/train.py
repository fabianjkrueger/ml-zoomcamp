##### goal: train a logistic regression classifier for churn prediction #####

# parameters
C = 1.0 # regularization for logistic regression
n_splits = 5

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
PATH_DATA = Path("../../data/churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# functions
def one_hot_encode(data, columns):
    """
    One hot encode data using a dict vectorizer.
    
    Arguments:
        - data: Input data.
        - columns: Columns to consider and encode.
        
    Returns:
        - dv: Fitted dict vectorizer.
        - data: One hot encoded version of input data.    
    """
    # get dictionary of data
    dicts = data[columns].to_dict(orient='records')
    # initialize a dict vectorizer
    dv = DictVectorizer(sparse=False)
    # use the dict vectorizer to encode the data
    data = dv.fit_transform(dicts)

    return dv, data

def train_model(features_train, labels_train, C=1.0):
    """
    Train a model and get a dict vectorizer.
    
    Arguments:
        - features_train: Features used for training.
        - labels_train: Labels used for training.
        - C: Regularization parameter. Defaults to 1.0.
    """
    
    # apply one hot encoding to train features and get dict vectorizer
    dv, features_train = one_hot_encode(
        data=features_train,
        columns=feature_columns
    )
    
    # convert labels to numpy array
    #labels_train = np.array(labels_train)
    
    # initialize, then train a model
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(features_train, labels_train)
    
    # return both model and dict vectorizer
    return dv, model

def predict(data, dv, model, columns):
    dicts = data[columns].to_dict(orient="records")
    
    features = dv.transform(dicts)
    predictions = model.predict_proba(features)[:, 1]

    return predictions

# data preparation

# read data
data_churn_full = pd.read_csv(PATH_DATA, sep=",")

# change column names to lowercase
data_churn_full.columns = data_churn_full.columns.str.lower()

# convert total charges column to numeric, because it was interpreted as object
data_churn_full.totalcharges = pd.to_numeric(data_churn_full.totalcharges, errors='coerce')
data_churn_full.totalcharges = data_churn_full.totalcharges.fillna(0)

# convert label column to numeric
data_churn_full.churn = (data_churn_full.churn == 'Yes').astype(int)

# make list containing columns that will be used as features

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

# join both to get all feature columns
feature_columns = numerical + categorical

# also make a list containing the label column
label = ["churn"]


# select features from full data
X = data_churn_full[feature_columns]

# select labels from full data
y = data_churn_full[label]


# get full train and test sets from full data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# training

# get dict vectorizer and train the model
dict_vectorizer, model = train_model(
    features_train=X_train,
    labels_train=y_train,
    C=C
)


def do_kfold_evaluation(X_t, y_t, C=C):

    # 5-fold cross evaluation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(X_t):
        
        X_train_kfold = X_t.iloc[train_idx]
        X_val_kfold = X_t.iloc[val_idx]
        
        y_train_kfold = y_t.iloc[train_idx]
        y_val_kfold = y_t.iloc[val_idx]
        
        dv_kfold, model_kfold = train_model(
            features_train=X_train_kfold,
            labels_train=y_train_kfold, C=C
        )
        
        y_pred_kfold = predict(data=X_val_kfold,
                            dv=dv_kfold,
                            model=model_kfold,
                            columns=feature_columns
        )

        auc = roc_auc_score(y_val_kfold, y_pred_kfold)
        scores.append(auc)
        
    print('C=%s %.3f +- %.3f')

do_kfold_evaluation(
    X_t=X_train,
    y_t=y_train
)

# save the model



