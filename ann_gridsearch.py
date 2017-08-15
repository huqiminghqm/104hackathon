# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from tensorflow.contrib.keras.api.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout
from tensorflow.contrib.keras import backend
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np

#匯入user_log.csv
train_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/user_log.csv', sep='|', dtype={'uid': str, 'jobNo': str})
#匯入文字向量vector.csv
vector_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/jieba_result/wiki_seg_job_0728_true_vector.txt', sep=',', error_bad_lines=False,header=None)
vector_df = vector_df.rename(index=str, columns={300: "jobNo"})
vector_df.loc[:,"jobNo"] = vector_df.jobNo.map(lambda x: str(x).replace(' ', '').replace('.0', ''))

#將文字向量合併到原本的'viewJob', 'saveJob', 'applyJob'變數
grouped_df = train_df[train_df.action != 'viewCust'].groupby(['uid', 'jobNo', 'action']).size().reset_index(name='count')
pivot_df = grouped_df.pivot_table(index=['uid', 'jobNo'], columns='action', values='count').fillna(0)
pivot_df = pivot_df.reindex_axis(['viewJob', 'saveJob', 'applyJob'], axis=1)
pivot_df_1 = pivot_df.reset_index().merge(vector_df,on = 'jobNo', how = 'inner').set_index(['uid', 'jobNo'])

#設置Independent variables與 Dependent variables
X = pd.np.hstack([pivot_df_1.values[:, :2], pivot_df_1.values[:, 3:]])
y = pivot_df.values[:, 2]
y[y >= 1] = 1

#切割training data 與 testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 設置相關參數
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=24, kernel_initializer='uniform', activation='relu', input_dim=2))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units=24, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# 以GridSearchCV尋找最佳帶寬
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [512],
              'epochs': [100],
              'optimizer': ['adam']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=5, n_jobs = 3)
grid_search = grid_search.fit(X_train, y_train)

y_pred = grid_search.best_estimator_.predict(X_test)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("Best parameters:", best_parameters)
print("Best accuracy:", best_accuracy)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)

y_pred = model.predict(X_test)
precision_recall_fscore_support(y_test, y_pred, average='macro')
print (precision_recall_fscore_support)
