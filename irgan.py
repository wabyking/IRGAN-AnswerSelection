#coding=utf-8
#! /usr/bin/env python3.4
#coding=utf-8
#! /usr/bin/env python3.4

import tensorflow as tf
import numpy as np
import time
from codecs import open
import os
from Discriminator import Discriminator
from Generator import Generator
import pickle,datetime
from dataHelper import BucketIterator

timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time()) ))
log_precision = 'log/test1.dns'+timeStamp +".log"


from config import Params
opts= Params()
opts.parseArgs() 

if not os.path.exists("log"):
    os.mkdir("log")
if not os.path.exists("model"):
    os.mkdir("model")
    
from dataHelper import InsuranceQA        
dataset=InsuranceQA(opts) 
    
def main():
    pass
    


                                        
if __name__ == '__main__':

    g1 = tf.Graph()
    g2 = tf.Graph()
    sess1 = tf.InteractiveSession(graph=g1)        
    sess2 = tf.InteractiveSession(graph=g2)
    with g1.as_default():
        generator = Generator(opts)
        saver1 = tf.train.Saver(max_to_keep=50)
        sess1.run(tf.global_variables_initializer())
    with g2.as_default():
        discriminator = Discriminator(opts)    
        saver2 = tf.train.Saver(max_to_keep=50)
        sess2.run(tf.global_variables_initializer())

        # evaluation(sess,discriminator,log,0)
    for i in range(opts.num_epochs):
        if i>0:
            samples=dataset.generate_gan(sess1,generator) 
            for j in range(opts.d_epochs_num):                            
                for _index,batch in enumerate(samples):    # try:                     
                    discriminator.train_step(batch,sess2)      
                    
            percision,step = dataset.evaluate(sess2,discriminator)    

    
        for g_epoch in range(opts.g_epochs_num):     
            
            
            for i,row in dataset.train.iterrows():
                q=row["question"]
                a=row["answer"]
                if i %100==0:
                    print( "have sampled %d pairs" % i) 
                
                
                
                # build a candidate pool
                neg_alist_index=[i for i in range(len(dataset.answers))] 
                sampled_index=np.random.choice(neg_alist_index,size=[opts.pools_size-1],replace= False)
#                neg_alist_index.remove(int(row["answer_index"]))
                sampled_index=list(sampled_index)
                sampled_index.append(int(row["answer_index"]))          

                pools=np.array(dataset.answers)[sampled_index]
                samples=[]
                for neg in pools:
                    samples.append([dataset.encode(item) for item in [q,a,neg]])
                
                
                # G sample 
                predicteds=[]
                for batch in BucketIterator(samples,batch_size=opts.batch_size):                            
                    predicted=generator.neg_over_pos_score(batch,sess1,score="gan")
                    predicteds.extend(predicted)   
                    
                exp_rating = np.exp(np.array(predicteds)*opts.sampled_temperature)
                prob = exp_rating / np.sum(exp_rating)                
                neg_index = np.random.choice(np.arange(len(pools)) , size=opts.gan_k, p=prob ,replace=False)
               
                
                # fetch reward from D
                subsamples=[]
                for neg in neg_index:
                    subsamples.append([dataset.encode(item) for item in [q,a,pool[neg]]])
                rewards=[]
                for batch in BucketIterator(subsamples,batch_size=opts.batch_size): 
                    reward=discriminator.reward(batch,sess2)
                    rewards.extend(reward)
                
                # feed back to G  

                step,current_loss,positive,negative = generator.feedback(samples,sess,neg_index,rewards)                                                                                                   #self.gan_loss = -tf.reduce_mean(tf.log(self.prob) * self.reward) 
    
                line=("%s: GEN step %d, loss %f  positive %f negative %f"%(datetime.datetime.now().isoformat(), step, current_loss,positive,negative))
                if _index %100==0:
                    print(line)

                
            
            percision,step = dataset.evaluate(sess1,generator)


