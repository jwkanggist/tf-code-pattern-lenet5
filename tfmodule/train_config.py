# Copyright 2018 Jaewook Kang (jwkang10@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-

from os import getcwd
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime



class TrainConfig(object):
    def __init__(self):

        self.learning_rate      = 0.01
        self.is_learning_rate_decay = True
        self.learning_rate_decay_rate =0.99
        self.opt_type='Adam'

        self.batch_size         = 1000


        # the number of step between evaluation
        self.display_step   = 50
        self.train_data_size      = 5000
        self.test_data_size       = 1000

        self.training_epochs = int(float(self.train_data_size/self.batch_size) * 10.0)

        # batch norm config
        self.batch_norm_decay   =  0.999
        self.batch_norm_fused   =  True
        self.FLAGS              = None

        # FC layer config
        self.dropout_keeprate       = 0.8
        self.fc_weights_initializer = tf.contrib.layers.xavier_initializer
        self.fc_weights_regularizer = tf.contrib.layers.l2_regularizer(4E-5)


        # conv layer config
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer = None
        self.biases_initializer  = slim.init_ops.zeros_initializer()

        self.is_trainable       = True
        self.activation_fn      = tf.nn.relu
        self.normalizer_fn      = slim.batch_norm
        self.multiprocessing_num = 1


        self.random_seed        = 66478

        # tensorboard config
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.root_logdir = getcwd() + '/export/'

        self.ckptdir  = self.root_logdir + '/pb_and_ckpt/'
        self.tflogdir = "{}/run-{}/".format(self.root_logdir+'/tf_logs', now)

