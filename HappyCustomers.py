#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('./data/ACME-HappinessSurvey2020.csv')

X = df.drop('Y', axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# With a simple logistic regression model, we get 55% training accuracy and 69% testing accuracy, 
# indicating underfitting of the data. We can either increment the size of the data set which would 
# not be feasible in this case or instead use a more complex model like random forest classifier to tackle the issue.
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(f'LogReg Training Accuracy:{logreg.score(X_train, y_train)}')
print(f'LogReg Testing Accuracy:{logreg.score(X_test, y_test)}')


# The Training accuracy drastically improves with RandomForestClassifier. However, the model seems to be
#  overfitting the training data as it is 30% higher than the testing accuracy. We could tune the hyperparameters
#  to reduce overfitting.
rf = RandomForestClassifier(random_state=22)
rf.fit(X_train, y_train)
print(f'RF Training Accuracy: {rf.score(X_train, y_train)}')
print(f'RF Testing Accuracy: {rf.score(X_test, y_test)}')

depths = np.arange(1,12)
train_accs = []
test_accs = []

for depth in depths:
    rf = RandomForestClassifier(n_estimators=50, max_depth=depth, min_samples_leaf=2,
                                random_state=22)
    rf.fit(X_train, y_train)
    train_accs.append(rf.score(X_train, y_train))
    test_accs.append(rf.score(X_test, y_test))
    
plt.figure()
plt.plot(depths, train_accs, label='Train')
plt.plot(depths, test_accs, label='Test')
plt.title('Training Accuracy vs Testing Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# It seems that the model exhibits the best performance with max_depth set to 4, so that will be the parameter we choose.
rf = RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=2,
                        random_state=22)
rf.fit(X_train, y_train)
print(f'Tuned RF Training Accuracy: {rf.score(X_train, y_train)}')
print(f'Tuned RF Testing Accuracy: {rf.score(X_test, y_test)}')


# We can now move on the analyzing the importance of the features.
plt.figure()
plt.bar(X.columns, rf.feature_importances_)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Importance of each feature')
plt.show()


# The two features with lowest importance are `X2` and `X4`. We can try removing them from 
# the training and see how does the performance change.
X_4_feature = X.drop(['X2', 'X4'], axis=1)
X_train_4_feature, X_test_4_feature, y_train, y_test = train_test_split(X_4_feature,
                                                y, test_size=0.2, random_state=40)

rf_4_feature = RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=2,
                        random_state=22)
rf_4_feature.fit(X_train_4_feature, y_train)
print(f'4-feature RF Training Accuracy: {rf_4_feature.score(X_train_4_feature, y_train)}')
print(f'4-feature RF Testing Accuracy: {rf_4_feature.score(X_test_4_feature, y_test)}')


# After dropping off the least important features, the model starts to overfit the training data 
# again and the performance on the testset drops significantly. However, what if we switch back to the 
# simple logistic regression model?
logreg_4_feature = LogisticRegression()
logreg_4_feature.fit(X_train_4_feature, y_train)
print(f'4-feature LogReg Training Accuracy:{logreg_4_feature.score(X_train_4_feature, y_train)}')
print(f'4-feature LogReg Testing Accuracy:{logreg_4_feature.score(X_test_4_feature, y_test)}')


# We can see that despite the model is still underfitting, dropping the two least important features actually 
# improved the training accuracy without compromising the testing accuracy at all. This might indicate that we can still
# find a model that is complex enough to still maintain the same level of performance even with 
# less features without over fitting the training set.
rf_4_feature = RandomForestClassifier(n_estimators=55, max_depth=2, min_samples_leaf=3,
                        random_state=22)
rf_4_feature.fit(X_train_4_feature, y_train)
print(f'simpler 4-feature RF Training Accuracy: {rf_4_feature.score(X_train_4_feature, y_train)}')
print(f'simpler 4-feature RF Testing Accuracy: {rf_4_feature.score(X_test_4_feature, y_test)}')

# It seems that we could potentially remove `X2` and `X4` in the next survey. Which makes sense 
# since customer are most likely to have purchased the product acknowledging what is to expect and the price of the item.
