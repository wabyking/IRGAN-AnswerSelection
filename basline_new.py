#coding=utf-8
#! /usr/bin/env python3.4

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle
import math
from Discriminator import Discriminator


timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time()) ))

precision = 'log/test1.dns'+timeStamp



from config import Params
opts= Params()
opts.parseArgs() 

from dataHelper import InsuranceQA        
dataset=InsuranceQA(opts) 




def main():
    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            session_conf = tf.ConfigProto(allow_soft_placement=True,  log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default() ,open(precision,"w") as log: 
                model = Discriminator(opts)
                sess.run(tf.global_variables_initializer())
                # evaluation(sess,discriminator,log,0)
                for i in range(opts.num_epochs):
                    # x1,x2,x3=generate_dns(sess,discriminator)
                    # samples=generate_dns(sess,discriminator)#generate_uniform_pair() #generate_dns(sess,discriminator) #generate_uniform() #                        
                    samples=dataset.generate_uniform_pair() #generate_dns_pair(sess,discriminator) #generate_uniform() # generate_uniform_pair() #                     
                    for j in range(1):
                        for batch in samples:    # try:                        
                                
                            model.train_step(batch,sess,model)
                            percision,step = dataset.evaluate(sess,model)
                            print("%d percision @1 : %.4f"%(step,percision))
                    
if __name__ == '__main__':
    main()
    # print (embeddings)
                 