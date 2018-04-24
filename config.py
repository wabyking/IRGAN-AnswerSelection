# -*- coding: utf-8 -*-


import numpy as np
import configparser
import argparse
class Params(object):
    def __init__(self, embedding_dim = 100, max_sequence_length = 200, filter_sizes ='1,2,3,5', num_filters = 500,
                 dropout_keep_prob = 1.0, l2_reg_lambda = 0.0000001,learning_rate = 0.1, batch_size = 128,
                 num_epochs = 500000, evaluate_every = 500, pools_size= 100,checkpoint_every=500,
                 sampled_size=100,g_epochs_num=1,d_epochs_num=1,sampled_temperature=20,gan_k=5):
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate =learning_rate
        self.batch_size = batch_size
        self.num_epochs=num_epochs
        self.evaluate_every = evaluate_every
        self.pools_size = pools_size
        self.checkpoint_every=checkpoint_every
        self.sampled_size=sampled_size
        self.g_epochs_num=g_epochs_num
        self.d_epochs_num=d_epochs_num
        self.sampled_temperature=sampled_temperature
        self.gan_k=gan_k


    def parse_config(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)
        config_common = config['COMMON']
        

        if 'embedding_dim' in config_common:
            self.embedding_dim = int(config_common['embedding_dim'])

        if 'max_sequence_length' in config_common:
            self.max_sequence_length = int(config_common['max_sequence_length'])

        if 'filter_sizes' in config_common:
            self.filter_sizes = config_common['filter_sizes']

        if 'num_filters' in config_common:
            self.num_filters = int(config_common['num_filters'])

        if 'dropout_keep_prob' in config_common:
            self.dropout_keep_prob = float(config_common['dropout_keep_prob'])

        if 'l2_reg_lambda' in config_common:
            self.l2_reg_lambda = float(config_common['l2_reg_lambda'])

        if 'learning_rate' in config_common:
            self.learning_rate = float(config_common['learning_rate'])

        if 'batch_size' in config_common:
            self.batch_size = int(config_common['batch_size'])

        if 'num_epochs' in config_common:
            self.num_epochs =int( config_common['num_epochs'])

        if 'evaluate_every' in config_common:
            self.evaluate_every =int( config_common['evaluate_every'])

        if 'pools_size' in config_common:
            self.pools_size = int(config_common['pools_size'])
        if 'checkpoint_every' in config_common:
            self.checkpoint_every =int( config_common['checkpoint_every'])
        if 'g_epochs_num' in config_common:
            self.g_epochs_num =int( config_common['g_epochs_num'])
        if 'd_epochs_num' in config_common:
            self.d_epochs_num =int( config_common['d_epochs_num'])
        if 'sampled_size' in config_common:
            self.sampled_size =int( config_common['sampled_size'])
        if 'sampled_temperature' in config_common:
            self.sampled_temperature =int( config_common['sampled_temperature'])
        if 'gan_k' in config_common:
            self.gan_k =int( config_common['gan_k'])



    def export_to_config(self, config_file_path):
        config = configparser.ConfigParser()
        config['COMMON'] = {}
        config_common = config['COMMON']
        config_common['embedding_dim'] = str(self.embedding_dim)
        config_common['max_sequence_length'] = str(self.max_sequence_length)
        config_common['filter_sizes'] = str(self.filter_sizes)
        config_common['num_filters'] = str(self.num_filters)
        config_common['dropout_keep_prob'] = str(self.dropout_keep_prob)
        config_common['l2_reg_lambda'] = str(self.l2_reg_lambda)
        config_common['learning_rate'] = str(self.learning_rate)
        config_common['batch_size'] = str(self.batch_size)
        config_common['evaluate_every'] = str(self.evaluate_every)
        config_common['num_epochs'] = str(self.num_epochs)
        config_common['pools_size'] = str(self.pools_size)
        config_common['checkpoint_every'] = str(self.checkpoint_every)
        config_common['sampled_size'] = str(self.sampled_size)
        config_common['g_epochs_num'] = str(self.g_epochs_num)
        config_common['d_epochs_num'] = str(self.d_epochs_num)
        config_common['sampled_temperature'] = str(self.sampled_temperature)
        config_common['gan_k'] = str(self.gan_k)

        with open(config_file_path, 'w') as configfile:
            config.write(configfile)
        return 

    def parseArgs(self):
        #required arguments:
        parser = argparse.ArgumentParser(description='run irgan in qa')
        parser.add_argument('-config', action = 'store', dest = 'config_file_path', help = 'The configuration file path.',default="config.ini")
        args = parser.parse_args()
        self.parse_config(args.config_file_path)
if __name__=="__main__":
    opts= Params()
    opts.parseArgs()