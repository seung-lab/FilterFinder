import os
import time
import itertools
import tensorflow as tf

import src.model as models
import src.data as d
import src.data_toy as toy_d
import src.metrics as metrics
import src.training as training
import src.loss as loss
import hyperparams

TIMESTAMP = int(time.time())

if hyperparams.FLAGS.model_dir:
  MODEL_DIR = hyperparams.FLAGS.model_dir
else:
  MODEL_DIR = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))

tf.logging.set_verbosity(hyperparams.FLAGS.loglevel)

import numpy as np


def main(unusedargs):
    hparams = hyperparams.create_hparams()

    print("Loading the data model...")
    if hparams.toy:
        data = toy_d.Data(hparams)
    else:
        data = d.Data(hparams)

    print("Creating the computational graph...")
    model = models.create_model(hparams, data, train= True)

    print("Pretrain network...")

    # Add Pretrain
    if hparams.toy:
        model = models.premodel_mnist(model, hparams)
        training.pretrain(model, hparams)

    print("Starting to train...")
    training.train(model, hparams, data)

if __name__ == "__main__":
  tf.app.run()
