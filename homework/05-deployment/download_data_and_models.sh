#!/usr/bin/env bash

# URL or rather path to data and directory to which it will be saved to
DATA="https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUTDIR="data/churn"

# create output directory
mkdir "$OUTPUTDIR"

# download data to output directory
wget -P "$OUTPUTDIR" "$DATA"

# download model and dict vectorizer
PREFIX=https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/cohorts/2024/05-deployment/homework
wget $PREFIX/model1.bin
wget $PREFIX/dv.bin