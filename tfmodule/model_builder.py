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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from model_config import model_config
from model_config import model_chout_num


def get_model(model_in,
              dropout_keeprate_node,
              train_config,
              scope):


    net = model_in
    with tf.variable_scope(name_or_scope=scope,values=[model_in]):

        # batch norm arg_scope
        with slim.arg_scope([train_config.normalizer_fn],
                            decay=train_config.batch_norm_decay,
                            fused=train_config.batch_norm_fused,
                            is_training=train_config.is_trainable,
                            activation_fn=train_config.activation_fn):

            if train_config.normalizer_fn == None:
                conv_activation_fn = train_config.activation_fn
            else:
                conv_activation_fn = None
            # max_pool arg_scope
            with slim.arg_scope([slim.max_pool2d],
                                stride      = model_config['maxpool_stride'],
                                kernel_size = model_config['maxpool_ksize'],
                                padding     = 'VALID'):

                # convolutional layer arg_scope
                with slim.arg_scope([slim.conv2d],
                                        kernel_size = model_config['conv_ksize'],
                                        stride      = model_config['conv_stride'],
                                        weights_initializer = train_config.weights_initializer,
                                        weights_regularizer = train_config.weights_regularizer,
                                        biases_initializer  = train_config.biases_initializer,
                                        trainable           = train_config.is_trainable,
                                        activation_fn       = conv_activation_fn,
                                        normalizer_fn       = train_config.normalizer_fn):


                    net = slim.conv2d(inputs=    net,
                                     num_outputs= model_chout_num['c1'],
                                     padding    = 'SAME',
                                     scope      = 'c1_conv')

                    net = slim.max_pool2d(inputs=   net,
                                          scope ='s2_pool')

                    net = slim.conv2d(inputs        = net,
                                      num_outputs   = model_chout_num['c3'],
                                      padding       = 'VALID',
                                      scope         = 'c3_conv')

                    net = slim.max_pool2d(inputs    = net,
                                          scope     = 's4_pool')

                    net  = slim.conv2d(inputs       = net,
                                       num_outputs  = model_chout_num['c5'],
                                       padding      = 'VALID',
                                       scope        = 'c5_conv')


        # output layer by fully-connected layer
        with slim.arg_scope([slim.fully_connected],
                            trainable=      train_config.is_trainable):

            with slim.arg_scope([slim.dropout],
                                keep_prob   =dropout_keeprate_node,
                                is_training=train_config.is_trainable):

                net = slim.fully_connected(inputs        =net,
                                           num_outputs  = model_chout_num['f6'],
                                           activation_fn= train_config.activation_fn,
                                           scope        ='f6_fc')

                net = slim.dropout(inputs=net,
                                   scope='f6_dropout')

                net = slim.fully_connected(inputs       =net,
                                           num_outputs  =model_chout_num['out'],
                                           activation_fn=None,
                                           scope        ='out_fc')

                out_logit = slim.dropout(inputs=net,
                                         scope='out_dropout')

                out_logit = tf.reshape(out_logit,
                                       shape=[-1,
                                              model_chout_num['out']])

        return out_logit

