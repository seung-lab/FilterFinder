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

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import src.helpers as helpers
import src.loss as loss
import hyperparams

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

import hyperparams
import src.data as d
import src.model as models

import tensorflow as tf
import src.model as models
import src.data as d
import src.metrics as metrics
import src.training as training
import src.loss as loss
import src.visual as vis
import hyperparams
import numpy as np
import src.helpers as helpers
import time
import util

FLAGS = None
nccnet = True

pathset = [ (120,9900, 11000), (20, 9900, 11000),
            (60, 16000, 17000),(70, 16000, 17000),
            (400, 8500, 27000),(400, 7000, 27000),
            (300, 7000, 21500),(151, 4500, 5000),
            (51, 18000, 9500), (52, 18000, 7500),
            (55, 18000, 7500), (60, 18100, 8400)]


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data, hparams, num_examples, name):
  """Converts a dataset to tfrecords."""

  s_rows = hparams.in_source_width
  t_rows = hparams.in_template_width

  filename = os.path.join(hparams.data_dir, name + '.tfrecords')

  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)

  if nccnet:
      model = models.create_model(hparams, data, train = False)
      sess = model.sess
      g = model

  else: #Simple NCC
    sess = tf.Session()
    model = models.Graph()

    model.image = tf.placeholder(tf.float32, shape=[hparams.in_source_width,hparams.in_source_width])
    model.template = tf.placeholder(tf.float32, shape=[hparams.in_template_width,hparams.in_template_width])

    search_dim = tf.expand_dims(tf.expand_dims(search, dim=0), dim=3)
    template_dim = tf.expand_dims(tf.expand_dims(template, dim=0), dim=3)

    model.source_alpha = [search_dim]
    model.template_alpha = [template_dim]
    model.similar = tf.constant(np.ones((8)))

    model = models.normxcorr(g, hparams)
    model = loss.loss(g, hparams)

  index = 0
  while(index < num_examples):

    if nccnet:
        t, s = data.getBatch(hparams)
    else:
        t, s = data.getSample([t_rows, t_rows], [s_rows, s_rows], hparams.resize, data.metadata)

    ct = hparams.in_template_width/2-hparams.template_width/2
    st = hparams.in_source_width/2-hparams.source_width/2

    t_cropped = t[:, int(ct):int(ct+hparams.template_width), int(ct):int(ct+hparams.template_width)]
    s_cropped = s[:, int(st):int(st+hparams.source_width), int(st):int(st+hparams.source_width)]

    results = sess.run(model.full_loss, feed_dict={model.template: t_cropped, model.image: s_cropped, model.similar: np.ones((8))})

    for i in range(8):
        result = results[i,0,0,0]
        print(result)
        if(result> -0.14) or result<-0.90:
            print('done', index)

            search_raw = np.asarray(s[i]*255, dtype=np.uint8).tostring()
            temp_raw = np.asarray(t[i]*255, dtype=np.uint8).tostring()

            ex = tf.train.Example(features=tf.train.Features(feature={
                'search_raw': _bytes_feature(search_raw),
                'template_raw': _bytes_feature(temp_raw),}))

            writer.write(ex.SerializeToString())
            index += 1
  writer.close()


def main(unused_argv):
  # Get the data.
  hparams = hyperparams.create_hparams()
  data = d.Data(hparams, prepare = True )

  # Convert to Examples and write the result to TFRecords.
  convert_to(data, hparams, 1000, 'hardest_examples')
  #convert_to(data, hparams, 1000, 'validation_1K')
  #convert_to(data, hparams, 1000, 'test_1K')


if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
