from tkinter import Y
import numpy as np
from mysklearn import myutils
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.myevaluation import accuracy_score
from mysklearn.myutils import *
from random import randint

from mysklearn.mypytable import MyPyTable


def bootstrap(table):
    return [table[randint(0, len(table)-1)] for _ in len(table)]


class MyRandomForestClassifier:

    def __init__(self, N, M, F):
        self.N = N
        self.M = M
        self.F = F
        self.trees = []
        # parallel to trees (stores the trees accuracy scores)
        self.accuracy = []

    def fit(self, X_train, y_train):

        for i in range(self.N):
            tree = MyDecisionTreeClassifier(self.F)
            X_sample, X_remaining, y_sample, y_true = myevaluation.bootstrap_sample(
                X_train, y_train, n_samples=int(len(X_train) * .66), random_state=i)
            tree.fit(X_sample, y_sample)
            y_pred = tree.predict(X_remaining)
            accuracy_score = myevaluation.accuracy_score(y_true, y_pred)
            self.trees.append(tree)
            self.accuracy.append(accuracy_score)

    def predict(self, X_test):
        y_preds = []

        best_trees = []

        for i in range(self.M):
            index = self.accuracy.index(max(self.accuracy))
            best_trees.append(self.trees[index])
            self.trees.pop(index)
            self.accuracy.pop(index)

        for test_instance in X_test:
            predictions = []
            for tree in best_trees:
                predictions.append(tree.predict([test_instance]))
            predictions = [prediction[0] for prediction in predictions]
            values, cts = myutils.get_frequencies(predictions)
            winner = myutils.majority_vote_winner(values, cts)
            y_preds.append(winner)

        return y_preds
