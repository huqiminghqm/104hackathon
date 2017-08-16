import pandas as pd
import xgboost as xgb
from sklearn.externals import joblib

# load your model
gbm = joblib.load('gbm.pkl')

# read test data
test_df = pd.read_csv('./data/user_log_test.csv', sep='|')
X_test = test_df.values
y_pred = gbm.predict(X_test)

# output your result
y_pred.to_csv("predict.csv", index=False, sep='|')
