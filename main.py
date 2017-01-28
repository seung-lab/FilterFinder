
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

source_shape = [512,512]
template_shape = [250,250]
kernel_shape = np.array([32,32])
kernel_identity = False
radius = 20
learing_rate = 0.01
lambd = -0.5
loss_type = 'diff'
num_steps = 5000
epoch_size = 10
num_test_steps = 60
n_slices = 100
resize = 2
#data = getData()

# %%
#metadata = getMetadata()
train_set = getAlignedData(train=True)
test_set = getAlignedData(train=False)

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
train(num_steps, source_shape, template_shape)


evaluate(50, source_shape, template_shape)


# To Do - Evaluate
# - DEBUG: Check what blows up Seems source and template become 0.999 and afterwards it blows up
# - EVALUATE: Get pathological cases
# - EVALUATE: Set Bandpass and compare
# - EVALUATE: Experiment with embedded images (such as Template and Image are the same)
# - FEATURE: elementwise multiply with softmax
