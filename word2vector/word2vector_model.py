#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:13:46 2017

@author: liouscott
"""

# -*- coding: utf-8 -*-

from gensim.models import word2vec
from gensim.models import doc2vec
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus("/Users/liouscott/Documents/scott/104_competition/jieba_result/wiki_seg_job_0728_true.txt")
    model = word2vec.Word2Vec(sentences, size=250, alpha=0.025, window=30, min_count=5, workers=3)
    # Save our model.
    model.save("/Users/liouscott/Documents/scott/104_competition/text_mining/word2vec_0728.model.bin")
if __name__ == "__main__":
    main()