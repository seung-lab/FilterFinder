%matplotlib inline
%config InlineBackend.figure.format = 'svg'
import matplotlib.pyplot as plt
import pylab as pl
import tensorflow as tf
import numpy as np

#from utils import show, getData, normxcorr2, loss, model
import time
from math import floor

metadata = getMetadata()
train_set = getAlignedData(train=True)
test_set = getAlignedData(train=False)

source_shape = [512,512]
template_shape = [400,400]
kernel_shape = [32,32]
kernel_identity = False
radius = 10
learing_rate = 0.0001
num_steps = 1000
epoch_size = 10
num_test_steps = 60
n_slices = 200
resize = 2
#data = getData()


# Write normalised cross-correlation and loss function
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
image = tf.placeholder(tf.float32, shape=source_shape)
temp = tf.placeholder(tf.float32, shape=template_shape)

source_alpha = tf.placeholder(tf.float32, shape=source_shape)
template_alpha = tf.placeholder(tf.float32, shape=template_shape)

#construct the model, init variables and return the computation
def model(img, tmp, kernel_shape):
    #dim housekeeping

    #Build Convolution layer
    if kernel_identity:
        kernel_shape = np.array(kernel_shape)
        kernel_init = np.zeros(kernel_shape)
        kernel_init[kernel_shape[0]/2,kernel_shape[1]/2] = 1
    else:
        kernel_init = tf.truncated_normal(kernel_shape, stddev=0.1)

    kernel = tf.Variable(kernel_init)
    source_alpha = fftconvolve2d(image, kernel)
    template_alpha = fftconvolve2d(tmp, kernel)

    return  kernel, source_alpha, template_alpha

# Build the model and
kernel, source_alpha, template_alpha  = model(image, temp, kernel_shape)
p = normxcorr2(source_alpha, template_alpha)
l, p_max, p_max_2, mask_p  = loss(p, radius)
train_step = tf.train.AdamOptimizer(learing_rate).minimize(l)

sess.run(tf.initialize_all_variables())

def test(num_test_steps, source_shape, template_shape, aligned=True):
    sum_dist = 0
    for i in range(num_test_steps):
        if aligned:
            t,s = getAlignedSample(template_shape, source_shape, test_set)
        ls = sess.run(l, feed_dict={image: s, temp: t})
        sum_dist = sum_dist - ls

    print ("test set average: %g"%(sum_dist/num_test_steps))

def train(num_steps, source_shape, template_shape, aligned=True):
    loss = np.zeros(num_steps)
    for i in range(num_steps):
        a = time.time()
        #Check id data is aligned
        if aligned:
            t,s = getAlignedSample(template_shape, source_shape, train_set)
        else:
            t,s = getSample(template_shape, source_shape, resize, metadata)

        #Train step
        _, ls, p_max_c, p_max_c_2 = sess.run([train_step,l, p_max, p_max_2], feed_dict={image: s, temp: t})
        loss[i] = -ls
        #Show Learning curve
        b = time.time()

        if i%epoch_size==0:
            test(num_test_steps, source_shape, template_shape)

        #print("step %d, maximizing %g: max_p %g max_p_2 %g, time %g"%(i, -ls, p_max_c, p_max_c_2, b-a))

    return loss

loss_data = train(num_steps, source_shape, template_shape)


evaluate()

evaluate()

print(loss_data)

# To Do - Evaluate
# - Optimization - post on slack the code
# - GPU implementation V
# - Plot Loss curve
# - Plot 3D surface
# - Get edge cases
# - Construct Test set (Quality vs Quantity)
# - Set Bandpass and compare
