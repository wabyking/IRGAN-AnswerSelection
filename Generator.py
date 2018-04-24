#coding=utf-8
import tensorflow as tf 
import numpy as np 

from QACNN import QACNN
class Generator(QACNN):
    
    def __init__(self, opt,paras=None,learning_rate=1e-2,embeddings=None,loss="pair",trainable=True):
        QACNN.__init__(self, opt,paras=paras,embeddings=embeddings,loss=loss,trainable=trainable)
        self.model_type="Gen"
        self.reward  =tf.placeholder(tf.float32, shape=[None], name='reward')
        self.neg_index  =tf.placeholder(tf.int32, shape=[None], name='neg_index')

        self.batch_scores =tf.nn.softmax( self.score13-self.score12) #~~~~~
        # self.all_logits =tf.nn.softmax( self.score13) #~~~~~
        self.prob = tf.gather(self.batch_scores,self.neg_index)
        self.gan_loss =  -tf.reduce_mean(tf.log(self.prob) *self.reward) +self.opt.l2_reg_lambda * self.l2_loss
        
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.opt.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.gan_loss)
        self.gan_updates = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)


    def feedback(self,batch,sess,neg_index,rewards,verbose=True):


        feed_dict = {
                    self.input_x_1: batch[:,0],
                    self.input_x_2: batch[:,1],
                    self.neg_index: neg_index,
                    self.input_x_3: batch[:,2],
                    self.reward: rewards,
                    self.dropout_keep_prob_holder:self.opt.dropout_keep_prob
                }
        _, step,    current_loss,positive,negative = sess.run(                                                                                    #应该是全集上的softmax    但是此处做全集的softmax开销太大了
            [self.gan_updates, self.global_step, self.gan_loss, self.positive,self.negative],         #     self.prob= tf.nn.softmax( self.cos_13)
            feed_dict)  
        return current_loss,step,positive,negative
  



