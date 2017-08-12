#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 11:07:50 2017

@author: liouscott
"""

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
import pandas as pd

train_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/user_log.csv', sep='|', dtype={'uid': str, 'jobNo': str})
#train_df.head()
train_df.describe()


job_structured_info_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/job/job_structured_info.csv', sep='|', dtype={'invoice': str, 'jobno': str, 'job': str, 'jobcat1': str, 'jobcat2': str, 'jobcat3': str, 'edu': str, 'salary_low': str, 'salary_high': str, 'role': str, 'language1': str, 'language2': str, 'language3': str, 'period': str, 'major_cat': str, 'major_cat2': str, 'major_cat3': str, 'industry': str, 'worktime': str, 'role_status': str, 's2': str, 's3': str, 'addr_no': str, 's9': str, 'need_emp': str, 'need_emp1': str, 'startby': str, 'exp_jobcat1': str, 'exp_jobcat2': str, 'exp_jobcat3': str})
#job_structured_info_df.head()
job_structured_info_df.describe()

job_description_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/job/job_description.csv', sep='|', dtype={'invoice': str, 'jobno': str, 'job': str, 'description': str, 'others': str}, error_bad_lines=False)
#job_description_df.head()
job_description_df.describe()

companies_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/companies.csv', sep=',')
#companies_df.head()
companies_df.describe()

department_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/category/department.csv', sep=',')
department_df.head()
department_df.describe()

district_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/category/district.csv', sep=',', error_bad_lines=False)
#district_df.head()
district_df.describe()

industry_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/category/industry.csv', sep=',', error_bad_lines=False)
#industry_df.head()
industry_df.describe()

job_category_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/category/job_category.csv', sep=',')
#job_category_df.head()
job_category_df.describe()






