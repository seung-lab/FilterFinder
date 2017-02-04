
# %%
%matplotlib inline
%config InlineBackend.figure.format = 'svg'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab as pl
import tensorflow as tf
import numpy as np
from IPython import display
import h5py
import time
from math import floor
from scipy.misc import imresize
import numpy.matlib

source_shape = [600,600]
template_shape = [250,250]
kernel_shape = np.array([32,32])
kernel_identity = False
radius = 20
learing_rate = 0.0001
lambd = -0.5
loss_type = 'diff'
num_steps = 250
epoch_size = 10
num_test_steps = 30
n_slices = 100
resize = 3

# %%
metadata = getMetadata()
#train_set = getAlignedData(train=True)
#test_set = getAlignedData(train=False)

# %%
# Write normalised cross-correlation and loss function
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
image = tf.placeholder(tf.float32, shape=source_shape)
temp = tf.placeholder(tf.float32, shape=template_shape)

source_alpha = tf.placeholder(tf.float32, shape=source_shape)
template_alpha = tf.placeholder(tf.float32, shape=template_shape)

# Build the model and
kernel, kernel_2, source_alpha, template_alpha  = model(image, temp, kernel_shape, kernel_identity)

p = normxcorr2FFT(source_alpha, template_alpha)
l, p_max, p_max_2, mask_p  = loss(p, radius, ltype=loss_type)
train_step = tf.train.AdamOptimizer(learing_rate).minimize(l)

sess.run(tf.global_variables_initializer())

# %%
train(num_steps, source_shape, template_shape, False)

# Experiment 1 (16x16) - Non Linear
# 0.126 - 0.851 0.725 Identity Training 1
# 0.127 - 0.655 0.528 Identity
# 0.11 - 0.84 0.73 Identity Training 2 (with one layer)
# 0.143 -  0.53 0.387 Random
# 0.17 - 0.77 0.60 Random Initialized trained (with one layer)

# Experiment 2 (32x32) - Non Linear
# --- Identity Training 1
# 0.128 Identity
# 0.23 Identity Training 2 (with one layer)
# 0.19 -  0.52 0.329 Random
# 0.15 - 0.84 0.69 Random Initialized trained (with one layer)

evaluate(60, source_shape, template_shape, aligned=False, pos=(16000, 17000))
# To Do - Evaluate
# - EVALUATE: Get pathological cases - Done
# - FEATURE: At more features at each layer
# - FEATURE: Do striding
# - EXPERIMENT: Make smaller the strides
# - EVALUATE: Set Bandpass and compare
# - EVALUATE: Experiment with embedded images (such as Template and Image are the same)
# - Experiment: Area under the curve - elementwise multiply with softmax
# - EXPERIMENT: Try to run kernel on after normxcor (Ashwin)
# - EXPERIMENT: Run on the same layer
# - EXPERIMENT: Skip one layer

# Next Steps
# Decide the experiments of convolution and do grid search


# Blows up
# - DEBUG: Check what blows up Seems source and template become 0.999 and afterwards it blows up
# - On bigger images the problem does not exists if the initializatin is identity
# - it is not due to FFT (might be delaying cause. but not solving it)
# - affect by changing the std of weight distribution and the learning rate (heavily)
# - still exists
