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
import numpy as np
import time

from train_config import TrainConfig
from data_loader import MnistDataLoader
from data_loader import FileManager

from model_builder import get_model
from model_config import model_config


def train(dataloader_train,dataloader_test,trainconfig_worker):


    # model building =========================
    model_in = tf.placeholder(dtype=model_config['image_dtype'],
                                shape=[None,
                                       dataloader_train.IMAGE_SIZE,
                                       dataloader_train.IMAGE_SIZE,
                                       dataloader_train.NUM_CHANNELS])
    labels   = tf.placeholder(dtype=model_config['label_dtype'],
                              shape=[None])



    with tf.variable_scope(name_or_scope='model',values=[model_in, labels]):
        dropout_keeprate_node = tf.placeholder(dtype=model_config['image_dtype'])

        model_out = get_model(model_in              =model_in,
                              dropout_keeprate_node =dropout_keeprate_node,
                              train_config          =trainconfig_worker,
                              scope                 ='model')




    # tf data loading ===================================================
    with tf.name_scope(name='dataloader'):
        images_placeholder  = tf.placeholder(   dtype=model_config['image_dtype'],
                                                shape=dataloader_train.image_shape)
        label_placeholder   = tf.placeholder(   dtype=model_config['label_dtype'],
                                                shape=[None])
        train_dataset   = dataloader_train.input_fn(images_placeholder,label_placeholder)
        train_iterator  = train_dataset.make_initializable_iterator()



    # traning ops =============================================
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=model_out))
    train_op = tf.train.AdamOptimizer(learning_rate=trainconfig_worker.learning_rate).minimize(loss=loss_op)


    with tf.name_scope('model_out'):
        model_pred = tf.nn.softmax(model_out)
    with tf.name_scope('eval_performance'):
        error             = tf.equal(tf.argmax(model_pred,1),labels)
        tf_pred_accuracy     = tf.reduce_mean(tf.cast(error,tf.float32))


    # For Tensorboard ===========================================
    file_writer = tf.summary.FileWriter(logdir=trainconfig_worker.tflogdir)
    file_writer.add_graph(tf.get_default_graph())

    tb_summary_accuracy_train = tf.summary.scalar('accuracy_train', tf_pred_accuracy)
    tb_summary_accuracy_test = tf.summary.scalar('accuracy_test', tf_pred_accuracy)

    tb_summary_cost         = tf.summary.scalar('loss', loss_op)




    # training ==============================

    train_error_rate        = np.zeros(shape=np.ceil(trainconfig_worker.training_epochs/trainconfig_worker.display_step).astype(np.int16),
                                       dtype=np.float32)
    test_error_rate         = np.zeros(shape=np.ceil(trainconfig_worker.training_epochs/trainconfig_worker.display_step).astype(np.int16),
                                       dtype=np.float32)
    init_var = tf.global_variables_initializer()

    print('[train] training_epochs = %s' % trainconfig_worker.training_epochs)
    print('------------------------------------')




    with tf.Session() as sess:
        # Run the variable initializer
        sess.run(init_var)

        # importing data
        image_train_numpy, label_train_numpy = \
            dataloader_train.import_data(imagefilename=fm.train_images_filename,
                                         labelfilename=fm.train_labels_filename)
        image_test_numpy, label_test_numpy = \
            dataloader_test.import_data(imagefilename=fm.test_images_filename,
                                         labelfilename=fm.test_labels_filename)

        sess.run(train_iterator.initializer, feed_dict={images_placeholder: image_train_numpy,
                                                         label_placeholder: label_train_numpy})
        images_train_op, labels_train_op = train_iterator.get_next()


        avg_cost = 0.
        rate_record_index = 0

        for epoch in range(trainconfig_worker.training_epochs):

            start_time = time.time()
            image_train_batch, label_train_batch    = sess.run([images_train_op, labels_train_op])
            _, minibatch_cost = sess.run([train_op,loss_op],
                                         feed_dict={model_in:  image_train_batch,
                                                    labels:     label_train_batch,
                                                    dropout_keeprate_node: trainconfig_worker.dropout_keeprate})

            # compute average cost and error rate
            avg_cost += minibatch_cost


            if trainconfig_worker.display_step == 0:
                continue
            elif epoch  % trainconfig_worker.display_step == 0:
                elapsed_time = time.time() - start_time

                train_error_rate[rate_record_index] = (1.0 - tf_pred_accuracy.eval(feed_dict={model_in: image_train_numpy,
                                                                                              labels: label_train_numpy,
                                                                                              dropout_keeprate_node: 1.0})) *100.0

                test_error_rate[rate_record_index] = (1.0 - tf_pred_accuracy.eval(feed_dict={model_in: image_test_numpy,
                                                                                             labels: label_test_numpy,
                                                                                             dropout_keeprate_node: 1.0})) * 100.0



                tb_summary_cost_result, tb_summary_accuracy_train_result  = sess.run([tb_summary_cost,tb_summary_accuracy_train],
                                                                                       feed_dict={model_in: image_train_numpy,
                                                                                                  labels: label_train_numpy,
                                                                                                  dropout_keeprate_node:1.0})

                tb_summary_accuracy_test_result  = sess.run(tb_summary_accuracy_test,
                                                               feed_dict={model_in: image_test_numpy,
                                                                          labels: label_test_numpy,
                                                                          dropout_keeprate_node:1.0})
                
                file_writer.add_summary(tb_summary_cost_result,epoch)
                file_writer.add_summary(tb_summary_accuracy_train_result,epoch)
                file_writer.add_summary(tb_summary_accuracy_test_result,epoch)


                print('At epoch = %d, elapsed_time = %.1f ms' % (epoch, elapsed_time))
                # print("Training set avg cost (avg over minibatches)=%.2f" % avg_cost)
                print("Training set Err rate (avg over minibatches)= %.2f %%  " % (train_error_rate[rate_record_index]))
                print("Test set Err rate (total batch)= %.2f %%" % (test_error_rate[rate_record_index]))
                print("--------------------------------------------")

                rate_record_index += 1

        print("Training finished!")

    file_writer.close()





if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    trainconfig_worker = TrainConfig()
    fm = FileManager()

    # dataloader instance gen
    dataloader_train = MnistDataLoader(is_training        =trainconfig_worker.is_trainable,
                                         datasize           =trainconfig_worker.train_data_size,
                                         batch_size         =trainconfig_worker.batch_size,
                                         multiprocessing_num=trainconfig_worker.multiprocessing_num,
                                         is_image_scaling   =True)

    dataloader_test    = MnistDataLoader(is_training        =False,
                                         datasize           =trainconfig_worker.test_data_size)

    # model tranining
    with tf.name_scope(name='trainer',values=[dataloader_train,dataloader_test]):

        train(dataloader_train=dataloader_train,
              dataloader_test = dataloader_test,
              trainconfig_worker=trainconfig_worker)