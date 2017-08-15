#coding:utf-8
"""
Created on Tue Jul 18 11:31:07 2017

@author: liouscott
"""
import gensim, logging
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import os
import pandas as pd

#將斷完詞的檔案wiki_seg_job_0728_true.txt讀入，接著Doc2Vec去training，最後使用model.save得到最後的model
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
sentences = gensim.models.doc2vec.TaggedLineDocument('/Users/liouscott/Documents/scott/104_competition/jieba_result/wiki_seg_job_0728_true.txt')
model = gensim.models.Doc2Vec(sentences, size = 30, window = 250, workers=3)
model.save('/Users/liouscott/Documents/scott/104_competition/model/wiki_seg_job_0728_true_model.txt')
#將Doc2Vec.load
model = Doc2Vec.load('/Users/liouscott/Documents/scott/104_competition/model/wiki_seg_job_0728_true_model.txt')
#確認資料行數，才能知道資料有沒有跑掉，蠻重要的。
print (len(model.docvecs))


#讀入job_description.csv，
job_description_df = pd.read_csv('/Users/liouscott/Documents/scott/104_competition/job/job_description.csv', sep='|', dtype={'invoice': str, 'jobno': str, 'job': str, 'description': str, 'others': str}, error_bad_lines=False)
jobno = job_description_df['jobno']
#由於TaggedLineDocument並沒有存在對回原本table的key值，所以抓取其中的jobno，在模型輸出成向量時增加一個jobno的tag，生成wiki_seg_job_0728_true_vector.txt
out = open('/Users/liouscott/Documents/scott/104_competition/jieba_result/wiki_seg_job_0728_true_vector.txt', 'a')
for idx, docvec in enumerate(model.docvecs):

    for value in docvec:
        out.write('%.3f' % value + ',')
    out.write(str(jobno[idx]).replace(" ","").replace(".0",""))
    out.write('\n')
    print (idx)
    #print (docvec)
out.close()
