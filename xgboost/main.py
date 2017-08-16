# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# you can replace to your data path
train_df = pd.read_csv('./data/user_log.csv', sep='|')

X, y = train_df.values[:, 1:], train_df[:, :1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# set all parameters
parameters = {
                'objective': 'binary:logistic',
                'learning_rate': 0.01,
                'max_depth': 6,
                'n_estimators': 1000,
                'seed': 1033,
                'subsamples': 0.9,
                'colsample_bytree': 0.8
             }


gbm = xgb.XGBClassifier(**parameters)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)], eval_metric="logloss", early_stopping_rounds=30)


y_pred = gbm.predict(X_test)

# making the confusion matrix
result = precision_recall_fscore_support(y_test, y_pred, average="binary")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(result)

# save your model
joblib.dump(gbm, 'gbm.pkl')
