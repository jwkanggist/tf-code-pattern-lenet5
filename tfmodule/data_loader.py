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


import os
import gzip
from os import getcwd
import numpy as np
from six.moves import urllib
import tensorflow as tf
import functools


# data filename =====================================================
'''
    we splits the LeCun's training data to training and validation data.
    we use the LeCun's test data as it is.
'''

class FileManager(object):

    def __init__(self):
        self.train_images_filename = 'train-images-idx3-ubyte.gz'
        self.train_labels_filename = 'train-labels-idx1-ubyte.gz'
        self.test_images_filename = 't10k-images-idx3-ubyte.gz'
        self.test_labels_filename = 't10k-labels-idx1-ubyte.gz'



class MnistDataLoader(object):

    def __init__(self,is_training,
                 datasize,
                 batch_size=1,
                 multiprocessing_num=1,
                 is_image_scaling=True):


        self.SOURCE_URL     = 'http://yann.lecun.com/exdb/mnist/'
        self.WORK_DIRECTORY = getcwd() + '/data/mnist'

        # The size of mnist image
        # The image size of the first convolutional layer in LeNet5 is 28 X 28
        self.IMAGE_SIZE     = 28
        # gray scsle image
        self.NUM_CHANNELS   = 1
        self.PIXEL_DEPTH    = 255
        # 0 to 9 char images
        self.NUM_LABELS     = 10

        self.image_shape = [None,
                            self.IMAGE_SIZE,
                            self.IMAGE_SIZE,
                            self.NUM_CHANNELS]

        self.batch_size             = batch_size
        self.datasize               = datasize
        self.is_training            = is_training
        self.multiprocessing_num    = multiprocessing_num
        self.is_image_scaling       = is_image_scaling

        self.images_path = None
        self.label_path  = None




    def _set_shapes(self,batch_size,images,labels):


        images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))

        # below codes must be modified after applying preprocessing
        labels.set_shape(labels.get_shape().merge_with(
            tf.TensorShape([batch_size])))

        return images, labels



    def _preprocessing_fn(self):
        # currently no preprocessing used
        pass



    def input_fn(self,image_placeholder,label_placeholder):

        dataset = tf.data.Dataset.from_tensor_slices((image_placeholder,
                                                      label_placeholder))

        if self.is_training:
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=self.datasize))


        # dataset = dataset.map(map_func=self._preprocessing_fn)
        dataset = dataset.batch(batch_size=self.batch_size)

        # if we have preprocessing for data,
        # use the below functions instead of map() and batch() separately
        ##----------------------------------------------------------------
        # USE tf.contrib.data.map_and_batch()
        # dataset = dataset.apply(
        #     tf.contrib.data.map_and_batch(map_func=self.parse_fn,
        #                                   batch_size=self.batch_size,
        #                                   drop_remainder=True))
        ##----------------------------------------------------------------

        # set batch size to placeholder
        dataset = dataset.map(
            functools.partial(self._set_shapes,
                              self.batch_size),
            num_parallel_calls=self.multiprocessing_num)



        # prefetch for parallel data fetching
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        tf.logging.info('[Input_fn] dataset pipeline building complete')


        return dataset




    def import_data(self,imagefilename,
                         labelfilename):

        tf.logging.info('[Input_fn] is_training = %s' % self.is_training)
        tf.logging.info('[Input_fn] download if the files does not exist')

        self.images_path = self._download(filename=imagefilename)
        self.label_path  = self._download(filename=labelfilename)

        image_numpy = self._extract_data()
        label_numpy = self._extract_label()

        return image_numpy, label_numpy





    # function module for data set loading  ============================
    def _download(self,filename):
        '''
            check whether we have the mnist dataset in the given WORK_DIRECTORY,
            otherwise, download the data from YANN's website,
        '''

        if not tf.gfile.Exists(self.WORK_DIRECTORY):
            tf.gfile.MakeDirs(self.WORK_DIRECTORY)
            tf.logging.info(" %s is not exist" % self.WORK_DIRECTORY)

        filepath = os.path.join(self.WORK_DIRECTORY,filename)

        tf.logging.info('filepath = %s' % filepath)

        if not tf.gfile.Exists(filepath):
            filepath, _ = urllib.request.urlretrieve(self.SOURCE_URL+ filename, filepath)
            with tf.gfile.GFile(filepath) as f:
                size = f.size()
                tf.logging.info('Successfully downloaded',filename,size,'bytes.')

            tf.logging.info('[download_mnist_dataset] filepath = %s' % filepath)
        return filepath




    def _extract_data(self):
        '''
        Extract the image into 4D tensor [image index, height,weight, channels]
        values are rescaled from [ 0, 255] down to [-0.5, 0.5]

        LeCun provides training set in a type of BMP format (No compression)
        One pixel value is from 0 to 255

        For representation, this needs 1byte = 8 bits ==> 2^8 =256

        TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel
        '''
        print ('[extract_data] Extracting gzipped data from %s' % self.images_path)

        with gzip.open(self.images_path) as bytestream:
            # threw out the header which has 16 bytes
            bytestream.read(16)

            # extract image data
            buf     = bytestream.read(self.IMAGE_SIZE * self.IMAGE_SIZE * self.datasize * self.NUM_CHANNELS)

            # type cast from uint8 to np.float32 to work in tensorflow framework
            data    = np.frombuffer(buffer=buf,
                                    dtype =np.uint8).astype(np.float32)
            if self.is_image_scaling:
                # rescaling data set over [-0.5 0.5]
                data    = (data - (self.PIXEL_DEPTH / 2.0) ) / self.PIXEL_DEPTH

            # reshaping to 4D tensors
            data    = data.reshape(self.datasize, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS)
            return data




    def _extract_label(self):
        '''
            Extract the lable into vector of int64 label IDs
        '''
        print ('[extract_label] Extracting gzipped data from %s' % self.label_path)

        with gzip.open(filename=self.label_path) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * self.datasize)
            # type cast from uint8 to np.int64 to work in tensorflow framework
            labels = np.frombuffer(buffer=buf,
                                   dtype=np.uint8).astype(np.int64)
        print('[extract_label] label= %s'% labels)
        return labels
