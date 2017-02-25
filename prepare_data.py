# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts EM data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

import hyperparams
import src.data as d

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data, hparams, num_examples, name):
  """Converts a dataset to tfrecords."""

  s_rows = hparams.source_width
  t_rows = hparams.template_width

  filename = os.path.join(hparams.data_dir, name + '.tfrecords')

  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)

  for index in range(num_examples):
    if index%20 == 0:
        print(str(100*index/float(num_examples))+"%")
    #Get images
    t, s = data.getSample([t_rows, t_rows], [s_rows, s_rows], hparams.resize, data.metadata)

    search_raw = np.asarray(s*255, dtype=np.uint8).tostring()
    temp_raw = np.asarray(t*255, dtype=np.uint8).tostring()

    ex = tf.train.Example(features=tf.train.Features(feature={
        'search_raw': _bytes_feature(search_raw),
        'template_raw': _bytes_feature(temp_raw),}))
    writer.write(ex.SerializeToString())

  writer.close()


def main(unused_argv):
  # Get the data.
  hparams = hyperparams.create_hparams()
  data = d.Data(hparams, prepare = True )

  # Convert to Examples and write the result to TFRecords.
  convert_to(data, hparams, 64000, 'train_100K')
  convert_to(data, hparams, 1000, 'validation_1K')
  convert_to(data, hparams, 1000, 'test_1K')


if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
