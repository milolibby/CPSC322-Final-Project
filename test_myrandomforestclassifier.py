from sklearn.metrics import label_ranking_average_precision_score
from mysklearn.myrandomforestclassifier import MyRandomForestClassifier
import pandas as pd
import csv
import numpy as np
from tabulate import tabulate
from mysklearn.myevaluation import stratified_kfold_cross_validation
from mysklearn import myutils
from sklearn.ensemble import RandomForestClassifier

N = 20
M = 7
F = 2
X_train = X_test = y_train = y_test = []


def import_nba_dataset():
    AGE = 3
    MIN = 8
    PTS = 9
    y_labels = []
    dataset = []
    with open("./nba_ratings.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = []
        header.append(next(csv_reader))
        for i, row in enumerate(csv_reader):
            y_labels.append([row[-1]])
            dataset.append([row[AGE], row[MIN], row[PTS]])

    return dataset, y_labels


#dataset = pd.read_csv("./nba_ratings.csv")

# interview dataset
interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
interview_table = [
    ["Senior", "Java", "no", "no", "False"],
    ["Senior", "Java", "no", "yes", "False"],
    ["Mid", "Python", "no", "no", "True"],
    ["Junior", "Python", "no", "no", "True"],
    ["Junior", "R", "yes", "no", "True"],
    ["Junior", "R", "yes", "yes", "False"],
    ["Mid", "R", "yes", "yes", "True"],
    ["Senior", "Python", "no", "no", "False"],
    ["Senior", "R", "yes", "no", "True"],
    ["Junior", "Python", "yes", "no", "True"],
    ["Senior", "Python", "yes", "yes", "True"],
    ["Mid", "Python", "no", "yes", "True"],
    ["Mid", "Java", "yes", "no", "True"],
    ["Junior", "Python", "no", "yes", "False"]
]
# remove the classifications from each row in interview table and
# add them to y_labels
interview_y_labels = [row[-1] for row in interview_table]
interview_table = [row[0:-1] for row in interview_table]
"""
Prepend the attribute labels
Ex: 'Senior' converted to 'level=Senior' 
"""
# for row in interview_table:
#    for i in range(len(row)):
#        row[i] = header[i] + "=" + str(row[i])


# nba dataset
nba_table, nba_labels = import_nba_dataset()

X_train = [['low', 'moderate', 'mid'], ['average', 'low', 'bottom'], ['average', 'moderate', 'bottom'], ['low', 'moderate', 'mid'], ['low', 'low', 'bottom'], ['low', 'moderate', 'mid'], ['low', 'low', 'bottom'], [
    'low', 'high', 'top'], ['high', 'high', 'top'], ['low', 'low', 'bottom'], ['average', 'moderate', 'mid'], ['average', 'high', 'mid'], ['low', 'moderate', 'mid'], ['low', 'low', 'bottom'], ['low', 'moderate', 'mid']]

y_train = ['special', 'basic', 'special', 'basic', 'basic', 'special', 'basic',
           'special', 'special', 'basic', 'special', 'special', 'special', 'basic', 'basic']

myRFC = MyRandomForestClassifier(6, 5, 3)


def test_fit():
    myRFC.fit(X_train, y_train)

    assert len(myRFC.trees) == 6


def test_predict():
    y_pred = myRFC.predict([["average", "high", "top"]])
    assert y_pred == ["special"]


def get_train_test_sets(ds: list, y_labels):
    X_train_folds, X_test_folds = stratified_kfold_cross_validation(
        ds, y_labels)

    # create the train and test data based on the index we got from the stratified_kfold_cross_validation
    for group in X_train_folds:
        for index in group:
            X_train.append(ds[index])
            y_train.append(y_labels[index])

    for group in X_test_folds:
        for index in group:
            X_test.append(ds[index])
            y_test.append(y_labels[index])


if __name__ == '__main__':
    test_fit()
    test_predict()
