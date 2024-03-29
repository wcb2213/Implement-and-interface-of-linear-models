#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/10/7


from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt


def try_Pipeline():
    # Import some data to play with
    X, y = load_iris(return_X_y=True)
    # Add non-informative features
    np.random.seed(0)
    X = np.hstack((X, 2 * np.random.random((X.shape[0], 36))))

    # Create a feature-selection transform, a scaler and an instance of SVM that we
    # combine together to have an full-blown estimator
    clf = Pipeline([('anova', SelectPercentile(chi2)),
                    ('scaler', StandardScaler()),
                    ('svc', SVC(gamma="auto"))])

    # Plot the cross-validation score as a function of percentile of features
    score_means = list()
    score_stds = list()
    percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

    for percentile in percentiles:
        clf.set_params(anova__percentile=percentile)
        this_scores = cross_val_score(clf, X, y, cv=5)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())

    plt.errorbar(percentiles, score_means, np.array(score_stds))
    plt.title(
        'Performance of the SVM-Anova varying the percentile of features selected')
    plt.xticks(np.linspace(0, 100, 11, endpoint=True))
    plt.xlabel('Percentile')
    plt.ylabel('Accuracy Score')
    plt.axis('tight')
    plt.show()

if __name__=='__main__':
    try_Pipeline()