import tensorflow as tf
import numpy as np
from collections import namedtuple
import json

# Model Parameters
tf.flags.DEFINE_string("exp_name", None, "Name of the run")
tf.flags.DEFINE_integer("source_width", 512, "The width of source image 512") #
tf.flags.DEFINE_integer("template_width", 224, "The width of template image 224") #
tf.flags.DEFINE_boolean("identity_init", False, "Initialize as Identity")
tf.flags.DEFINE_integer("resize", 3, "Resize Images")
tf.flags.DEFINE_integer("dropout", 1, "Global probability for dropout")
tf.flags.DEFINE_integer("dialation_rate", 1, "Global dilation rate ")
tf.flags.DEFINE_integer("aligned", 0, "Define the data type")
tf.flags.DEFINE_boolean("linear", False, "Decide if the convolution is linear or not")

# Loss Parameters
tf.flags.DEFINE_integer("radius", 10, "Maximum radius for finding second")
tf.flags.DEFINE_boolean("mean_over_batch", True, "Take the mean over the batch otherwise min")
tf.flags.DEFINE_float("lambd", -0.5, "Lambda for mixed loss")
tf.flags.DEFINE_float("eps", 0.0001, "small number")
tf.flags.DEFINE_string("loss_type", "dist", "Define the loss format either 'dist' or 'ratio' ")
tf.flags.DEFINE_string("loss_form", "minus", "Define the loss formulae to minimize over {'minus', 'inverse', 'log'}")
tf.flags.DEFINE_boolean("softmax", False, "Use Softmax")

# Data paths
tf.flags.DEFINE_string("loging_dir", "/FilterFinder/logs/", "Path for logging the data")
tf.flags.DEFINE_string("metadata_dir", "/FilterFinder/data/prealigned/registry.txt", "Path to registry.txt file")
tf.flags.DEFINE_string("prealigned_dir", "/FilterFinder/data/prealigned/", "Path to prealigned data")
tf.flags.DEFINE_string("aligned_dir", "/FilterFinder/data/aligned/pinky_aligned_11184-11695_25018-25529_1-260.h5", "Path to aligned fie")
tf.flags.DEFINE_string("model_dir", "/FilterFinder/model/", "Path to model files")
tf.flags.DEFINE_string("data_dir", "/FilterFinder/data/prepared", "Path to prepared data")

# Training Parameters
tf.flags.DEFINE_float("pretrain", False, "Pretrain convnet")
tf.flags.DEFINE_float("toy", False, "Train on toy example")
tf.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate")
tf.flags.DEFINE_float("momentum", 0.9, "Learning momentum")
tf.flags.DEFINE_float("decay", 0.75, "Learning momentum")
tf.flags.DEFINE_float("decay_steps", 1000, "Learning momentum")
tf.flags.DEFINE_integer("steps", 10000, "Number of steps to complete the training")
tf.flags.DEFINE_integer("batch_size", 8, "Batch size during training")
tf.flags.DEFINE_integer("epoch_size", 16, "Epoch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 2, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")

kernel_shape = [[3, 3, 1, 32],
                [3, 3, 32, 64],
                [3, 3, 64, 128],
                [3, 3, 128, 256],
                #[3, 3, 128, 256],
                ]


pathset = [ (120,9900, 11000), (20, 9900, 11000),
            (60, 16000, 17000),(70, 16000, 17000),
            (400, 8500, 27000),(400, 7000, 27000),
            (300, 7000, 21500),(151, 4500, 5000),
            (51, 18000, 9500), (52, 18000, 7500),
            (55, 18000, 7500), (60, 18100, 8400),

            (120,9900, 11000), (20, 9900, 11000),
            (60, 16000, 17000),(70, 16000, 17000),
            (400, 8500, 27000),(400, 7000, 27000),
            (300, 7000, 21500),(151, 4500, 5000),
            (51, 18000, 9500), (52, 18000, 7500),
            (55, 18000, 7500), (60, 18100, 8400),

            (120,9900, 11000), (20, 9900, 11000),
            (60, 16000, 17000),(70, 16000, 17000),
            (400, 8500, 27000),(400, 7000, 27000),
            (300, 7000, 21500),(151, 4500, 5000),
            (51, 18000, 9500), (52, 18000, 7500),
            (55, 18000, 7500), (60, 18100, 8400),

            ]

tf.flags.DEFINE_string("kernel_shape", json.dumps(kernel_shape), "The convolutional model")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  "HParams",
  [
    "exp_name",
    "source_width",
    "template_width",
    "identity_init",
    "resize",
    "dropout",
    "dialation_rate",
    "aligned",
    "radius",
    "lambd",
    "loss_type",
    "loss_form",
    "loging_dir",
    "metadata_dir",
    "prealigned_dir",
    "aligned_dir",
    "learning_rate",
    "momentum",
    "steps",
    "batch_size",
    "epoch_size",
    "eval_batch_size",
    "optimizer",
    "loglevel",
    "softmax",
    "kernel_shape",
    "pathset",
    "model_dir",
    "eps",
    "mean_over_batch",
    "data_dir",
    "toy",
    "decay",
    "decay_steps",
    "pretrain",
    "linear",
  ])

def create_hparams():
  print(FLAGS.kernel_shape)
  return HParams(
    pretrain = FLAGS.pretrain,
    exp_name = FLAGS.exp_name,
    source_width = FLAGS.source_width,
    template_width = FLAGS.template_width,
    identity_init = FLAGS.identity_init,
    resize = FLAGS.resize,
    dropout = FLAGS.dropout,
    dialation_rate = FLAGS.dialation_rate,
    aligned = FLAGS.aligned,
    radius = FLAGS.radius,
    lambd = FLAGS.lambd,
    loss_type =  FLAGS.loss_type,
    loss_form =  FLAGS.loss_form,
    metadata_dir = FLAGS.metadata_dir,
    prealigned_dir = FLAGS.prealigned_dir,
    aligned_dir = FLAGS.aligned_dir,
    learning_rate = FLAGS.learning_rate,
    momentum = FLAGS.momentum,
    steps = FLAGS.steps,
    batch_size = FLAGS.batch_size,
    epoch_size = FLAGS.epoch_size,
    eval_batch_size = FLAGS.eval_batch_size,
    optimizer = FLAGS.optimizer,
    loglevel = FLAGS.loglevel,
    softmax = FLAGS.softmax,
    kernel_shape = np.array(json.loads(FLAGS.kernel_shape)),
    pathset = pathset,
    model_dir = FLAGS.model_dir,
    loging_dir= FLAGS.loging_dir,
    eps = FLAGS.eps,
    mean_over_batch = FLAGS.mean_over_batch,
    data_dir = FLAGS.data_dir,
    toy = FLAGS.toy,
    decay = FLAGS.decay,
    decay_steps = FLAGS.decay_steps,
    linear = FLAGS.linear,
    )
