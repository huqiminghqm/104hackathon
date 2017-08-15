#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:19:41 2017

@author: liouscott
"""

from tensorflow.contrib.keras.api.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout
from tensorflow.contrib.keras import backend
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import jieba
import logging

#讀入job_description
job_description_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/job/job_description.csv', sep='|', dtype={'invoice': str, 'jobno': str, 'job': str, 'description': str, 'others': str}, error_bad_lines=False)
job = job_description_df['job']

job_in_now = job.tolist()


#增加dict.txt.big詞庫，去除stopwords.txt，注意要去除斷行符號。
def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # jieba custom setting.
    jieba.set_dictionary('/Users/liouscott/Documents/scott/104_competition/jieba_dict/dict.txt.big')

    # load stopwords set
    stopwordset = set()
    with open('/Users/liouscott/Documents/scott/104_competition/jieba_dict/stopwords.txt','r',encoding='utf-8') as sw:
        for line in sw:
            stopwordset.add(line.strip('\n'))

    output = open('/Users/liouscott/Documents/scott/104_competition/jieba_result/wiki_seg_job_0728_true.txt','a')
    texts_num = 0
    error_num = 0
    #print ("hi")
    for line in job_in_now:
        
        #print (line)
        #print ("hi")
        words = jieba.cut(line, cut_all=True)
        
        
        word_f = ''
        try:
            for word in words:
            
                
                word = word.replace('\n', ' ').replace('\r', '')
                if word not in stopwordset:
                    word_f = word_f + ' ' + word
            texts_num += 1
            if texts_num % 10000 == 0:
                logging.info("已完成前 %d 行的斷詞" % texts_num)


        except:
            error_num += 1
        output.write(word_f +'\n')

            

        
        
    print (error_num)
    output.close()
    
if __name__ == '__main__':
	main()