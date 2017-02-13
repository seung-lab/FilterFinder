import os
import time
import itertools
import tensorflow as tf

import src.model as models
import src.data as d
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

def main(unusedargs):
  hparams = hyperparams.create_hparams()
  data = d.Data(hparams)

  model = models.create_model(hparams)
  training.train(model, hparams, data)


if __name__ == "__main__":
  tf.app.run()
