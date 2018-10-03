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

import sys
from os import getcwd
from os import chdir

chdir('..')
sys.path.insert(0,getcwd())
print ('getcwd() = %s' % getcwd())


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# image processing tools
import cv2

# from train_config import train_config
from data_loader import MnistDataLoader
from data_loader import FileManager
from train_config import TrainConfig
from model_config import model_config



class DataLoaderTest(tf.test.TestCase):

    def test_data_loader_coco(self):
        '''
            This test checks below:
        '''

        fm = FileManager()
        train_config = TrainConfig()

        dataloader = MnistDataLoader(is_training    = train_config.is_trainable,
                                     datasize       = train_config.data_size,
                                     batch_size     = train_config.batch_size,
                                     multiprocessing_num     = train_config.multiprocessing_num,
                                     is_image_scaling = False)


        images_placeholder = tf.placeholder(dtype=model_config['image_dtype'],
                                            shape=dataloader.image_shape)

        label_placeholder  = tf.placeholder(dtype=model_config['label_dtype'],
                                            shape=[None])


        dataset                 = dataloader.input_fn(images_placeholder,
                                                      label_placeholder)

        iterator                = dataset.make_initializable_iterator()

        image_numpy, label_numpy = dataloader.import_data(imagefilename=fm.train_images_filename,
                                                          labelfilename=fm.train_labels_filename)


        with self.test_session() as sess:

            sess.run(iterator.initializer,feed_dict={images_placeholder:image_numpy,
                                                     label_placeholder:label_numpy})

            images_op, labels_op = iterator.get_next()


            for n in range(0,50):
                image_numpy_batch, label_numpy_batch = sess.run([images_op,labels_op])

                image_index = 0
                image_pick = image_numpy_batch[image_index,:,:,0]
                label_pick = label_numpy_batch[image_index]

                plt.figure(n)
                plt.imshow(image_pick.astype(np.uint8))
                plt.title('True = %d' % label_pick)
                plt.show()




if __name__ == '__main__':
    tf.test.main()

