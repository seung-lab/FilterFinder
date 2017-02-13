import tensorflow as tf
import numpy as np
from collections import namedtuple

# Model Parameters
tf.flags.DEFINE_integer("source_width", 512, "The width of source image")
tf.flags.DEFINE_integer("template_width", 224, "The width of template image")
tf.flags.DEFINE_integer("identity_init", 0, "Initialize as Identity")
tf.flags.DEFINE_integer("resize", 3, "Resize Images")
tf.flags.DEFINE_integer("dropout", 1, "Global probability for dropout")
tf.flags.DEFINE_integer("dialation_rate", 2, "Global dilation rate ")
tf.flags.DEFINE_integer("aligned", 0, "Define the data type")

# Loss Parameters
tf.flags.DEFINE_integer("radius", 20, "Maximum radius for finding second")
tf.flags.DEFINE_float("lambd", -0.5, "Lambda for mixed loss")
tf.flags.DEFINE_float("eps", 0.001, "small number")
tf.flags.DEFINE_string("loss_type", "dist", "Define the loss format Could be dist, ratio, dist_ratio, inv_dist")
tf.flags.DEFINE_boolean("softmax", False, "Use Softmax")

# Data paths
tf.flags.DEFINE_string("loging_dir", "/FilterFinder/logs/", "Path for logging the data")
tf.flags.DEFINE_string("metadata_dir", "/FilterFinder/data/prealigned/registry.txt", "Path to registry.txt file")
tf.flags.DEFINE_string("prealigned_dir", "/FilterFinder/data/prealigned/", "Path to prealigned data")
tf.flags.DEFINE_string("aligned_dir", "/FilterFinder/data/aligned/pinky_aligned_11184-11695_25018-25529_1-260.h5", "Path to aligned fie")
tf.flags.DEFINE_string("model_dir", "/FilterFinder/model/", "Path to model files")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_float("momentum", 0.5, "Learning momentum")
tf.flags.DEFINE_integer("steps", 1000, "Number of steps to complete the training")
tf.flags.DEFINE_integer("batch_size", 8, "Batch size during training")
tf.flags.DEFINE_integer("epoch_size", 16, "Epoch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 16, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")

kernel_shape =  np.array([  [5,5,1,8],
                            [3,3,8,1],
                            #[2,2,48,96],
                            #[3,3,96,7],
                            #[3,3,7,1],
                            #[16,16]
                            ])

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  "HParams",
  [
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
    "model_dir",
    "eps"
  ])

def create_hparams():
  return HParams(
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
    kernel_shape = kernel_shape,
    model_dir = FLAGS.model_dir,
    loging_dir= FLAGS.loging_dir,
    eps = FLAGS.eps
    )
