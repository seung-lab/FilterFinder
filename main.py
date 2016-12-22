
# %%
#%matplotlib inline
#%config InlineBackend.figure.format = 'svg'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab as pl
import tensorflow as tf
import numpy as np
from IPython import display
#from utils import show, getData, normxcorr2, loss, model
import time
from math import floor

source_shape = [512,512]
template_shape = [250,250]
kernel_shape = [128,128]
kernel_identity = False
radius = 20
learing_rate = 0.02
lambd = -0.5
loss_type = 'ratio'
num_steps = 250
epoch_size = 10
num_test_steps = 60
n_slices = 100
resize = 2
#data = getData()

# %%
metadata = getMetadata()
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
kernel, source_alpha, template_alpha  = model(image, temp, kernel_shape)
p = normxcorr2FFT(source_alpha, template_alpha)
l, p_max, p_max_2, mask_p  = loss(p, radius, ltype=loss_type)
train_step = tf.train.AdamOptimizer(learing_rate).minimize(l)

sess.run(tf.initialize_all_variables())

train(num_steps, source_shape, template_shape)


evaluate(50)


# To Do - Evaluate
# - Get edge cases
# - Set Bandpass and compare
