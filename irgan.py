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
import pickle
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


#
#                discriminator = Discriminator.Discriminator(
#                        sequence_length=FLAGS.max_sequence_length,
#                        batch_size=FLAGS.batch_size,
#                        vocab_size=len(vocab),
#                        embedding_size=FLAGS.embedding_dim,
#                        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
#                        num_filters=FLAGS.num_filters,
#                        learning_rate=FLAGS.learning_rate,
#                        l2_reg_lambda=FLAGS.l2_reg_lambda,
#                        # embeddings=embeddings,
#                        embeddings=None,
#                        paras=param,
#                        loss=loss_type)
#
#                generator = Generator.Generator(
#                        sequence_length=FLAGS.max_sequence_length,
#                        batch_size=FLAGS.batch_size,
#                        vocab_size=len(vocab),
#                        embedding_size=FLAGS.embedding_dim,
#                        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
#                        num_filters=FLAGS.num_filters,
#                        learning_rate=FLAGS.learning_rate*0.1,
#                        l2_reg_lambda=FLAGS.l2_reg_lambda,
#                        # embeddings=embeddings,
#                        embeddings=None,
#                        paras=param,
#                        loss=loss_type)
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
        
                    line=("%s: DIS step %d, loss %f with acc %f "%(datetime.datetime.now().isoformat(), step, current_loss,accuracy))
                    if _index%10==0:
                        print(line)
                    loss_log.write(line+"\n")
                    loss_log.flush()
                
            dataset.evaluate(sess2,discriminator,log,i)
    
        for g_epoch in range(opts.g_epochs_num):
            
            for i,row in self.train.iterrows():
                q=row["question"]
                a=row["answer"]
                if i %100==0:
                    print( "have sampled %d pairs" % i) 
                    
                neg_alist_index=[i for i in range(len(dataset.answers))] 
                neg_alist_index.remove(int(row["answer_index"]))                 #remove the positive index
                sampled_index=np.random.choice(neg_alist_index,size=[opts.pools_size],replace= False)
                pools=np.array(dataset.answers)[sampled_index]
                subsamples=[]
                for neg in pools:
                    subsamples.append([dataset.encode(item) for item in [q,neg]])
       
                predicteds=[]
                for batch in BucketIterator(subsamples,batch_size=self.opt.batch_size):                            
                    predicted=model.predict(batch,sess)
                    predicteds.extend(predicted)        
                exp_rating = np.exp(np.array(predicteds)*sampled_temperature)
                prob = exp_rating / np.sum(exp_rating)
                neg_samples = np.random.choice(pools, size= negative_size,p=prob,replace=False) 
                for neg in neg_samples:
                    samples.append([self.encode(item) for item in [q,a,neg]])
                
                
            for _index,pair in enumerate(raw):
    
                q=pair[2]
                a=pair[3]    
    
                neg_alist_index=[item for item in range(len(alist))] 
                sampled_index=np.random.choice(neg_alist_index,size=[FLAGS.pools_size-1],replace= False)
                sampled_index=list(sampled_index)
                sampled_index.append(_index)
                pools=np.array(alist)[sampled_index]
    
                samples=insurance_qa_data_helpers.loadCandidateSamples(q,a,pools,vocab)
                predicteds=[]
                for batch in insurance_qa_data_helpers.batch_iter(samples,batch_size=FLAGS.batch_size):                            
                    feed_dict = {generator.input_x_1: batch[:,0],generator.input_x_2: batch[:,1],generator.input_x_3: batch[:,2]}
                    
                    predicted=sess.run(generator.gan_score,feed_dict)
                    predicteds.extend(predicted)                                                         
                
                exp_rating = np.exp(np.array(predicteds)*FLAGS.sampled_temperature)
                prob = exp_rating / np.sum(exp_rating)
    
                neg_index = np.random.choice(np.arange(len(pools)) , size=FLAGS.gan_k, p=prob ,replace=False)    # 生成 FLAGS.gan_k个负例
    
                subsamples=np.array(insurance_qa_data_helpers.loadCandidateSamples(q,a,pools[neg_index],vocab))    
                feed_dict = {discriminator.input_x_1: subsamples[:,0],discriminator.input_x_2: subsamples[:,1],discriminator.input_x_3: subsamples[:,2]}
                reward = sess.run(discriminator.reward,feed_dict)                 # reward= 2 * (tf.sigmoid( score_13 ) - 0.5)
    
                samples=np.array(samples)
                feed_dict = {
                                generator.input_x_1: samples[:,0],
                                generator.input_x_2: samples[:,1],
                                generator.neg_index: neg_index,
                                generator.input_x_3: samples[:,2],
                                generator.reward: reward
                            }
                _, step,    current_loss,positive,negative = sess.run(                                                                                    #应该是全集上的softmax    但是此处做全集的softmax开销太大了
                        [generator.gan_updates, generator.global_step, generator.gan_loss, generator.positive,generator.negative],         #     self.prob= tf.nn.softmax( self.cos_13)
                        feed_dict)                                                                                                                    #self.gan_loss = -tf.reduce_mean(tf.log(self.prob) * self.reward) 
    
                line=("%s: GEN step %d, loss %f  positive %f negative %f"%(datetime.datetime.now().isoformat(), step, current_loss,positive,negative))
                if _index %100==0:
                    print(line)
                loss_log.write(line+"\n")
                loss_log.flush()
                
            
            evaluation(sess,generator,log,i*FLAGS.g_epochs_num + g_epoch)
            log.flush()


