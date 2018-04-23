import os
import numpy as np

from collections import Counter
from tqdm import tqdm
import random
import time
import pandas as pd
import tensorflow as tf

from functools import wraps
def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco  


from codecs import open
try:
    import cPickle as pickle
except ImportError:
    import pickle
    

    
class Alphabet(dict):
    def __init__(self, start_feature_id = 0, alphabet_type="text"):
        self.fid = start_feature_id
        if alphabet_type=="text":
            self.add('[PADDING]')
            self.add('[UNK]')
#            self.add('[END]')
            self.unknow_token = self.get('[UNK]')
#            self.end_token = self.get('[END]')
            self.padding_token = self.get('[PADDING]')

    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
      # self[idx] = item
            self.fid += 1
        return idx
    
    def addAll(self,words):
        for word in words:
            self.add(word)
            
    def dump(self, fname,path="temp"):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path,fname), "w",encoding="utf-8") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))
                

class BucketIterator(object):
    def __init__(self,data,opt=None,batch_size=2,shuffle=True,test=False,position=False):
        self.shuffle=shuffle
        self.data=data
        self.batch_size=batch_size
        self.test=test        
        if opt is not None:
            self.setup(opt)
    
    def transform(self,data):

        return np.array(data)    

    def __iter__(self):
        if self.shuffle:
#            self.data = self.data.sample(frac=1).reset_index(drop=True)
            random.shuffle(self.data)
        batch_nums = int(len(self.data)/self.batch_size)
        for  i in range(batch_nums):
            yield self.transform(self.data[i*self.batch_size:(i+1)*self.batch_size])
        yield self.transform(self.data[-1*self.batch_size:])
        
 

     

class InsuranceQA(object):
    def __init__(self,opt):
        self.opt=opt
        
        
        self.answers,self.validate,self.train,self.dev,self.test1,self.test2 = self.loadData()
        self.alphabet = self.buildVocab()
        self.opt.vocab_size=len(self.alphabet)

    def buildVocab(self): 
        alphabet = Alphabet()
        for sentences in ( self.answers,self.train["question"],self.validate["question"]):
            for sentence in sentences:
                for token in sentence.split():
                    alphabet.add(token)
        return alphabet
                    
    def loadData(self):
        answers = pd.read_csv("data/raw_answers.txt",sep="\t",names=["answer"])["answer"]  
        validate = pd.read_csv("data/validate.txt",sep="\t",names=["split","question","good","bad"]) 
        train = pd.read_csv("data/train.txt",sep="\t",names=["question","answer","answer_index"])
        dev =   validate[validate.split ==0 ] [["question","good","bad"]]
        test1 =   validate[validate.split ==1 ] [["question","good","bad"]] 
        test2 =   validate[validate.split ==2 ] [["question","good","bad"]] 
        return answers,validate,train,dev,test1,test2
    
    @log_time_delta
    def generate_uniform_pair(self):        
        samples=[]        
        for i,pair in self.train.iterrows():
            q=pair["question"]
            a=pair["answer"]
            index=random.randint(0, len(self.answers) - 1)
            neg= self.answers[index]            
            samples.append([self.encode(item) for item in [q,a,neg]])
        return BucketIterator(samples,batch_size=self.opt.batch_size)
    @log_time_delta    
    def generate_dns_pair(self,sess, model):
        samples=[]
        for i,row in self.train.iterrows():
            q=row["question"]
            a=row["answer"]
            if i %100==0:
                print( "have sampled %d pairs" % i)        
        
            pools=np.random.choice(self.answers,size=[self.opt.pools_size])
            
            subsamples=[]
            for neg in pools:
                subsamples.append([self.encode(item) for item in [q,neg]])
   
            predicteds=[]
            for batch in BucketIterator(subsamples,batch_size=self.opt.batch_size):                            
                predicted=model.predict(batch,sess)
                predicteds.extend(predicted)        
            index=np.argmax(predicteds)
            samples.append([self.encode(item) for item in [q,a,pools[index]]])
        return BucketIterator(samples,batch_size=self.opt.batch_size)
    
    @log_time_delta
    def generate_gan(self,sess, model,loss_type="pair",negative_size=3,sampled_temperature=20):
        samples=[]
        for i,row in self.train.iterrows():
            q=row["question"]
            a=row["answer"]
            if i %100==0:
                print( "have sampled %d pairs" % i) 
            neg_alist_index=[i for i in range(len(self.answers))] 
            neg_alist_index.remove(int(row["answer_index"]))                 #remove the positive index
            sampled_index=np.random.choice(neg_alist_index,size=[self.opt.pools_size],replace= False)
        		pools=np.array(answers)[sampled_index]
            subsamples=[]
            for neg in pools:
                subsamples.append([self.encode(item) for item in [q,neg]])
   
            predicteds=[]
            for batch in BucketIterator(subsamples,batch_size=self.opt.batch_size):                            
                predicted=model.predict(batch,sess)
                predicteds.extend(predicted)        
            exp_rating = np.exp(np.array(predicteds)*sampled_temperature)
        		prob = exp_rating / np.sum(exp_rating)
            neg_samples = np.random.choice(pools, size= negative_size,p=prob,replace=False) 
            for neg in neg_samples:
                samples.ppend([encode_sent(vocab,item, FLAGS.max_sequence_length) for item in [q,a,neg]])
            
            samples.append([self.encode(item) for item in [q,a,pools[index]]])
        return samples
            
        


    def encode(self,sentence):    
        tokens = sentence.split()
        return [self.alphabet.get(word,self.alphabet.unknow_token)  for word in tokens[:self.opt.max_sequence_length]] + [self.alphabet.padding_token] *int(self.opt.max_sequence_length-len(tokens))

    def dev_step(self,sess,model,dataset="dev",dev_size=None):
        dataset=dataset.strip().lower()
        if dataset=="test1":
            testData= self.test1
        elif dataset == "test2":
            testData= self.test2
        else:
            testData=self.dev
        if dev_size is not None:
            testData=testData[:dev_size]
        
        def evaluate_aplly(group):
#            group= self.dev.iloc[0,:]
            good = [int(i)-1 for i in group["good"].split()]
            bad = [int(i)-1 for i in group["bad"].split()]
            data = good + bad
            
            question = group["question"]
            samples = [(self.encode(question) , self.encode( self.answers[anaser_index] )) for anaser_index in data]
            scores =[]
            for batch in BucketIterator(samples,batch_size=self.opt.batch_size,shuffle=False) :
                scores.extend(model.predict(batch,sess))            
            if max(scores) > max(scores[:len(good)]):
                return 0
            else:
                return 1
            
        return testData.apply(evaluate_aplly,axis=1).mean()



    @log_time_delta
    def evaluate(self,sess,model,dataset="dev"):
        current_step = tf.train.global_step(sess, model.global_step)
        precision_current=self.dev_step(sess,model,dataset)

        print( model.save_model(sess,precision_current))
        return precision_current,current_step
        
if __name__ =="__main__":
    from config import Params
    opts= Params()
    opts.parseArgs()     
    dataset=InsuranceQA(opts) 

    items =  dataset.generate_uniform_pair()    



    




 
    
    
