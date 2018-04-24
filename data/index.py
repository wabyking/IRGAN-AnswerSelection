# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:47:10 2018

@author: wabywang
"""
import pandas as pd

answers = pd.read_csv("data/raw_answers.txt",sep="\t",names=["answer"])["answer"]  
validate = pd.read_csv("data/validate.txt",sep="\t",names=["split","question","good","bad"]) 
train = pd.read_csv("data/train.txt",sep="\t",names=["question","answer","answer_index"])


d = dict()
for i,answer in enumerate(answers):
    tokens= " ".join([item for item in answer.strip().split()][:200])
    d[tokens]=i

def aplly_test(group):

    return d[group["answer"].strip()]
    
    
ansers_index = train.apply(aplly_test,axis=1)
train["answer_index"]=ansers_index

train.to_csv("data/train.csv",index=False,sep="\t",header=None,encoding="utf-8")
