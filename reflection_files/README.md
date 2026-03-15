# Titanic - Machine Learning From Disaster
This project was completed as part of the Kaggle competition using the Titanic - Machine Learning From Disaster dataset. The goal of the competition was to predict whether a passenger would survive the Titanic disaster using supervised machine learning techniques.

## Deliverable
The main deliverable was a submission csv file that was produced by:

1. preprocessing the dataset
2. feature engineering
3. training and evaluating multiple learning models
4. selecting the best-performing model

The output of the csv file was in the format:

1. PassengerID
2. Survived

## Files Included
- train_cleaned.csv
    - cleaned training dataset produced by preprocessing script
- test_cleaned.csv
    - cleaned test dataset produced by proprocessing script
- train.py
    - training script that trains and evaluates multiple models, using the cleaned datasets.
- Data_preprocessing.py
    - preprocessing script that cleans raw Titanic data and performs feaure engineering techniques
- submission_1/2/3/4.csv
    - submission files uploaded to Kaggle

## How To Run

1. run the preprocessing script in the main project directory
    - python .\Data_preprocessing.py
2. run the training script
    - python .\train.py
