#coding=utf-8
import tensorflow as tf 
import numpy as np 
import datetime

from QACNN import QACNN           
class Discriminator(QACNN):
  

    def __init__(self,opt, paras=None,embeddings=None,loss="pair",trainable=True):
        
        QACNN.__init__(self, opt,paras=paras,embeddings=embeddings,loss=loss,trainable=trainable)
        self.model_type="Dis"

    
        with tf.name_scope("output"):

            self.losses = tf.maximum(0.0, tf.subtract(0.05, tf.subtract(self.score12, self.score13)))
            self.loss = tf.reduce_sum(self.losses) + self.opt.l2_reg_lambda * self.l2_loss
            
            self.reward = 2.0*(tf.sigmoid(tf.subtract(0.05, tf.subtract(self.score12, self.score13))) -0.5) # no log
            self.positive= tf.reduce_mean(self.score12)
            self.negative= tf.reduce_mean( self.score13)

            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")


        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.opt.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)

        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
    
    def train_step(self,batch,sess,verbose=True):
        feed_dict = {
                self.input_x_1: batch[:,0],
                self.input_x_2: batch[:,1],
                self.input_x_3: batch[:,2],
                self.dropout_keep_prob_holder:1.0
                }
        _, step,    current_loss,accuracy = sess.run(
                [self.train_op, self.global_step, self.loss,self.accuracy],
                feed_dict)
        
        time_str = datetime.datetime.now().isoformat()
        print(("%s: DIS step %d, loss %f with acc %f "%(time_str, step, current_loss,accuracy)))
        
        

if __name__ =="__main__":

    
    from config import Params
    opts= Params()
    opts.parseArgs() 
    
    from dataHelper import InsuranceQA        
    dataset=InsuranceQA(opts) 
    
    model = Discriminator(opts)
    sess= tf.Session()
    sess.run(tf.global_variables_initializer())
    batch = next(iter( dataset.generate_uniform_pair()))
    
    
    
