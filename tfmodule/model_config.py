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

import tensorflow as tf

model_config = {
    'image_dtype': tf.float32,
    'label_dtype': tf.int64,
    'maxpool_stride':   [2,2],
    'maxpool_ksize':    [2,2],
    'conv_stride':      [1,1],
    'conv_ksize':       [5,5]
}

model_chout_num = \
{
    'c1': 6,
    'c3': 16,
    'c5': 120,
    'f6': 84,
    'out':10
}


