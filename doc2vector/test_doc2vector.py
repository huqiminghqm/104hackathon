#coding:utf-8
"""
Created on Tue Jul 18 11:31:07 2017

@author: liouscott
"""
from gensim import models,corpora,similarities
import jieba.posseg as pseg
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import os
import pandas as pd
import jieba
import csv
import gensim, logging
import numpy as np

#job_description_df = pd.read_csv('/home/ubuntu/job_description.csv', sep='|', dtype={'invoice': str, 'jobno': str, 'job': str, 'description': str, 'others': str}, error_bad_lines=False)
#job_description_df.head()
#job_description_df.describe()
#description = job_description_df['description']
#description_in = description.tolist()

job_description_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/job/job_description.csv', sep='|', dtype={'invoice': str, 'jobno': str, 'job': str, 'description': str, 'others': str}, error_bad_lines=False)
job_description_df.describe()
job = job_description_df['job']
jobno = job_description_df['jobno']
job_in_now = job.tolist()
jobno_now = jobno.tolist()
yy = job_in_now[9210]
print (len(job_in_now))
print (yy)
#載入training好的wiki_seg_job_0728_true_model.txt，接著由test_data_1放入需要測試的文字，可找到
model = Doc2Vec.load('/Users/liouscott/Documents/scott/104_competition/model/wiki_seg_job_0728_true_model.txt')
test_data_1 = '工讀生'
test_cut_raw_1 =[]
item2=(pseg.cut(test_data_1))
for k in list(item2):
	test_cut_raw_1.append(k.word)
inferred_vector = model.infer_vector(test_cut_raw_1)
sims = model.docvecs.most_similar([inferred_vector], topn=20)
sims_two = np.dot(model.docvecs[6682], model.docvecs[50021])
print(sims)  #sims是一个tuples,(index_of_document, similarity)
print(sims_two)
print (content[9210])
print (len(description_in))
print (len(model.docvecs))
print (model.docvecs)



