#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:04:17 2017

@author: liouscott
"""

import pandas as pd
import numpy as np
#將companies.csv讀入並且去除任何一欄有空直的欄位
companies_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/companies.csv', sep=',', skiprows=[0], header=None, names = ["invoice","name","stock","head500"] , dtype={'invoice': str, 'name': str, 'stock':int, 'head500':int})
companies_df = companies_df.dropna(axis=1, how='any')
companies_df = companies_df.drop(['name'],axis=1)


#將user_log.csv讀入
train_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/user_log.csv', sep='|', dtype={'uid': str, 'jobNo': str})
#train_df.head()
train_df_head = train_df.head(200000)
train_df = None


#扣除viewCust，得到使用裝置的比例
train_df_noviewcust = train_df_head[train_df_head.action != 'viewCust']
train_df_noviewcust_s = train_df_noviewcust[['uid', 'jobNo', 'source']].groupby(['uid', 'jobNo', 'source']).size().reset_index(name='count')
train_df_noviewcust_p = train_df_noviewcust_s.pivot_table(index=['uid', 'jobNo'], columns='source', values = 'count').fillna(0)
train_df_noviewcust_p.loc[:,"sum_device"] = train_df_noviewcust_p['app'] + train_df_noviewcust_p['mobileWeb'] + train_df_noviewcust_p['web']
train_df_noviewcust_p.loc[:,"app_ratio"] =  train_df_noviewcust_p['app'] / train_df_noviewcust_p['sum_device']
train_df_noviewcust_p.loc[:,"mobileWeb_ratio"] =  train_df_noviewcust_p['mobileWeb'] / train_df_noviewcust_p['sum_device']
train_df_noviewcust_p.loc[:,"web_ratio"] =  train_df_noviewcust_p['web'] / train_df_noviewcust_p['sum_device']
train_df_noviewcust_p.reset_index(inplace=True)
train_df_noviewcust_p = train_df_noviewcust_p[['uid','jobNo', 'app_ratio', 'mobileWeb_ratio', 'web_ratio']]



#扣除viewcust，得到viewJob,saveJob,applyJob，並將saveJob歸戶成0和1，將viewJob大於10的等於10
grouped_df = train_df_noviewcust.groupby(['uid', 'jobNo', 'invoice', 'action']).size().reset_index(name='count')
pivot_df = grouped_df.pivot_table(index=['uid', 'jobNo', 'invoice'], columns='action', values='count').fillna(0)
pivot_df = pivot_df.reindex_axis(['viewJob', 'saveJob', 'applyJob'], axis=1)
pivot_df.reset_index(inplace=True)
pivot_df["saveJob"]= np.where(pivot_df['saveJob'] > 0, 1, 0)
pivot_df.loc[pivot_df['viewJob'] > 9, 'viewJob'] = 10

#算出每個工作的瀏覽數/（所有工作的瀏覽總和/看過幾個工作）
weight_view_df = pivot_df.groupby('uid')["viewJob"].agg(['sum','count']).reset_index().rename(columns={'sum':'view_sum', 'count':'view_count'})
weight_view_df_merge = pd.merge(pivot_df, weight_view_df, 'left', on = ["uid"] )
weight_view_df_merge.loc[:,"view_ratio"] = weight_view_df_merge["viewJob"] / (weight_view_df_merge["view_sum"] / weight_view_df_merge["view_count"] )

train_df_viewcust = train_df_head[train_df_head.action == 'viewCust']
train_df_viewcust_s = train_df_viewcust.groupby(['uid', 'invoice']).size().reset_index(name='viewcust_count')
train_df_viewcust_s.loc[train_df_viewcust_s['viewcust_count'] > 9, 'viewcust_count'] = 10

train_df_viewjob = train_df_head[train_df_head.action == 'viewJob']
train_df_viewjob.loc[:,"date"] = train_df_viewjob.dateTime.map(lambda x: x[:10])
train_df_dt = train_df_viewjob.groupby(['uid','jobNo','date']).size().reset_index(name='viewdt_count')
train_df_dt_u = train_df_dt.groupby(['uid','jobNo']).size().reset_index(name='jobdt_count')





#language 只取 language1 檢查是不是 "1111" 是的話代表沒有要求與言能力，反之代表有要求語言能力
# 管理責任 (s2) 只分成有或無 是否出差 (s3) 只分成有或無
job_structured_info_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/job/job_structured_info.csv', sep='|', dtype={'invoice': str, 'jobNo': str, 'job': str, 'jobcat1': str, 'jobcat2': str, 'jobcat3': str, 'edu': int, 'salary_low': str, 'salary_high': str, 'role': str, 'language1': str, 'language2': str, 'language3': str, 'period': str, 'major_cat': str, 'major_cat2': str, 'major_cat3': str, 'industry': str, 'worktime': str, 'role_status': str, 's2': int, 's3': int, 'addr_no': str, 's9': str, 'need_emp': str, 'need_emp1': str, 'startby': int, 'exp_jobcat1': str, 'exp_jobcat2': str, 'exp_jobcat3': str})
job_structured_info_df["language1"]= np.where(job_structured_info_df['language1']=='1111', 0, 1)
job_structured_info_df["s2"]= np.where(job_structured_info_df['s2'] > 0, 1, 0)
job_structured_info_df["s3"]= np.where(job_structured_info_df['s3'] > 0, 1, 0)
job_variable_finial = job_structured_info_df[['invoice','jobno','language1','s2','s3','period']]
job_variable_finial = job_variable_finial.rename(index=str, columns={"jobno": "jobNo"})



#edu role industry addr_no startby處理為onehotencoder
job_structured_info_df.loc[job_structured_info_df['startby'] < 2, 'startby'] = 1
job_structured_info_df.loc[job_structured_info_df['startby'] > 2, 'startby'] = 3
job_structured_info_df.loc[:,"addr_no_c"] = job_structured_info_df.addr_no.map(lambda x: x[:4])
job_structured_info_df.loc[job_structured_info_df.addr_no_c.map(lambda x: x != "6001") & job_structured_info_df.addr_no_c.map(lambda x: x != "6002") , 'addr_no_c'] = 0
job_structured_info_df.loc[:,"industry_c"] = job_structured_info_df.industry.map(lambda x: x[:4])
job_structured_info_df.loc[job_structured_info_df.role.map(lambda x: int(x) > 3),'role'] = 0
job_structured_info_df.loc[:,"job_c"] = job_structured_info_df.jobcat1.map(lambda x: x[:4])
job_structured_info_df.loc[job_structured_info_df.jobcat1.map(lambda x: int(x[:1])) != 2, 'job_c'] = 0
job_structured_info_df.loc[job_structured_info_df.edu.map(lambda x: int(x) <= 4),'edu'] = 4
job_structured_info_df.loc[job_structured_info_df.edu.map(lambda x: 4 < int(x) <= 8),'edu'] = 8
job_structured_info_df.loc[job_structured_info_df.edu.map(lambda x: 8 < int(x) <= 16),'edu'] = 16
job_structured_info_df.loc[job_structured_info_df.edu.map(lambda x: 16 < int(x)),'edu'] = 32

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
job_structured_info_df_main = job_structured_info_df[['invoice','jobno']]
job_structured_onehotencoder = job_structured_info_df[['edu','role','startby','job_c','industry_c','addr_no_c']].values
labelencoder = LabelEncoder()
job_structured_onehotencoder[:, 0] = labelencoder.fit_transform(job_structured_onehotencoder[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
job_structured_onehotencoder = onehotencoder.fit_transform(job_structured_onehotencoder).toarray()

job_structured_onehotencoder[:, 5] = labelencoder.fit_transform(job_structured_onehotencoder[:,5])
onehotencoder = OneHotEncoder(categorical_features = [5])
job_structured_onehotencoder = onehotencoder.fit_transform(job_structured_onehotencoder).toarray()

job_structured_onehotencoder[:, 7] = labelencoder.fit_transform(job_structured_onehotencoder[:,7])
onehotencoder = OneHotEncoder(categorical_features = [7])
job_structured_onehotencoder = onehotencoder.fit_transform(job_structured_onehotencoder).toarray()

job_structured_onehotencoder[:, 11] = labelencoder.fit_transform(job_structured_onehotencoder[:,11])
onehotencoder = OneHotEncoder(categorical_features = [11])
job_structured_onehotencoder = onehotencoder.fit_transform(job_structured_onehotencoder).toarray()

job_structured_onehotencoder[:, 30] = labelencoder.fit_transform(job_structured_onehotencoder[:,29])
onehotencoder = OneHotEncoder(categorical_features = [30])
job_structured_onehotencoder = onehotencoder.fit_transform(job_structured_onehotencoder).toarray()

job_structured_onehotencoder[:, 32] = labelencoder.fit_transform(job_structured_onehotencoder[:,32])
onehotencoder = OneHotEncoder(categorical_features = [32])
job_structured_onehotencoder = onehotencoder.fit_transform(job_structured_onehotencoder).toarray()

job_structured_e = pd.DataFrame(job_structured_onehotencoder)
job_onehotencoder_finial = pd.concat([job_structured_info_df_main, job_structured_e], axis=1)
job_onehotencoder_finial = job_onehotencoder_finial.rename(index=str, columns={"jobno": "jobNo"})


#將所有變數merge
all_data_one = pd.merge(pivot_df, train_df_viewcust_s, 'left', on = ["uid", "invoice"] )
all_data_two = pd.merge(all_data_one, train_df_noviewcust_p, 'left', on = ["uid", "jobNo"] )
all_data_three = pd.merge(all_data_two, companies_df, 'left', on = ["invoice"] )
all_data_four = pd.merge(all_data_three, train_df_dt_u, 'left', on = ["uid", "jobNo"] )
all_data_five = pd.merge(all_data_four, job_variable_finial, 'left', on = ["invoice", "jobNo"] )
all_data_six = pd.merge(all_data_five, job_variable_finial, 'left', on = ["invoice", "jobNo"] ).fillna(0)
all_data_seven = pd.merge(all_data_six, job_onehotencoder_finial, 'left', on = ["invoice", "jobNo"] )


