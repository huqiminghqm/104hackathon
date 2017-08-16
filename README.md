### 2017 104 Hackathon Recommendation
## 資料夾
*ANN<br />
*使用Gridsearch來尋找最佳training參數。*
```
ann_gridsearch.py
```
*放置參數到ANN Model。*
```
ann_savemodel.py
```
*Dataprocessing<br />
讀入所有104 hackathon data
```
read_data.py
```
處理建模變數
```
merge_data.py
```
*Decisiontree<br />
以觀察資料特性，來找出每一層的規則
```
Cpc_predict.ipynb
```
*Doc2vector<br />
針對資料中的description，以jieba斷詞。
```
job_description_jieba.py
```
將斷完詞檔案，建置doc2vec模型。
```
doc2vec.py
```
放入要測試的文字可找到相似的文章。
```
test_doc2vector.py
```
*Word2vector<br />
針對資料中的description，以jieba斷詞。
```
job_description_jieba.py
```
將斷完詞檔案，建置word2vector模型。
```
word2vector_model.py
```
有三種模式測試找到相似的文章
```
word2vector_test.py
```
## 參考網址：
*Text Mining
*1.http://zake7749.github.io/2016/08/28/word2vec-with-gensim/<br />
*2.https://www.bbsmax.com/A/l1dy8rax5e/<br />
*3.https://read01.com/xoGyG4.html#.WZPTUHcjFE4<br />
*4.http://blog.csdn.net/xiaomuworld/article/details/52461045<br />
*5.https://github.com/iamxiaomu/doc2vec/blob/master/doc2vct.py<br />
*6.http://www.cnblogs.com/kaituorensheng/p/3600137.html<br />
*7.http://blog.csdn.net/eastmount/article/details/50473675<br />
*8.https://radimrehurek.com/gensim/models/word2vec.html#id4<br />
*9.http://zake7749.github.io/2016/08/28/word2vec-with-gensim/<br />
