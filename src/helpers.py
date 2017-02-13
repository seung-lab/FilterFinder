import tensorflow as tf
import metrics
import numpy as np

def bias_variable(identity = False, name = 'bias'):
    if identity:
        initial = tf.constant(0.0)
    else:
        initial = tf.constant(-0.5)
    b = tf.Variable(initial)
    #metrics.variable_summaries(b)
    return b

def weight_variable(shape, identity = False, name = 'conv'):
    #Build Convolution layer
    if identity:
        kernel_shape = np.array(shape)
        kernel_init = np.zeros(shape)
        kernel_init[kernel_shape[0]/2,shape[1]/2] = 1.0
    else:
        kernel_init = tf.random_normal(shape, stddev=0.1)
    weight = tf.Variable(kernel_init, name=name)
    metrics.kernel_summary(weight, name)
    return weight

def convolve2d(x,y, padding = "VALID", strides=[1,1,1,1], rate = 1):

    #Dim corrections
    if(len(x.get_shape())<4):
        x = tf.expand_dims(x, dim=0)

    if(len(x.get_shape())<4):
        x = tf.expand_dims(x, dim=3)

    if (len(y.get_shape())==2):
        y = tf.expand_dims(tf.expand_dims(y,  dim=2), dim=3)

    y = tf.to_float(y, name='ToFloat')
    if rate>1:
        o = tf.nn.atrous_conv2d(x, y, rate=rate, padding=padding)
    else:
        o = tf.nn.conv2d(x, y, strides=strides, padding=padding)
    return tf.squeeze(o)

def softmax2d(image):
    # ASSERT:  if 0 is softmax 0 under all conditions
    shape = tuple(image.get_shape().as_list())
    image = tf.reshape(image, [shape[0]*shape[1]], name=None)
    soft_1D = tf.nn.softmax(image)
    soft_image = tf.reshape(soft_1D, shape, name=None)
    return soft_image

def fftconvolve2d(x, y, padding="VALID"):
    #return convolve2d(x,y)
    """
    x and y must be real 2-d tensors.

    mode must be "SAME" or "VALID".

    need to add custom striding
    """
    #Read shapes
    x_shape = tuple(x.get_shape().as_list())
    y_shape = tuple(y.get_shape().as_list())

    #Construct paddings and pad
    x_shape = np.array(x_shape)[0:2]-1
    y_pad =  [[0, x_shape[0]],[0, x_shape[1]]]
    y_shape = np.array(y_shape)[0:2]-1
    x_pad = [[0, y_shape[0]],[0, y_shape[1]]]

    x = tf.pad(x, x_pad)
    y = tf.pad(y, y_pad)

    # Go to FFT domain
    y = tf.cast(y, tf.complex64, name='complex_Y')
    x = tf.cast(x, tf.complex64, name='complex_X')

    y_fft = tf.fft2d(y, name='fft_Y')
    x_fft = tf.fft2d(x, name='fft_X')

    # Do elementwise multiplication
    convftt = tf.multiply(x_fft, y_fft, name='fft_mult')

    # Come back
    z = tf.ifft2d(convftt, name='ifft_z')
    z = tf.real(z)

    #Slice correctly based on requirements
    if padding == 'VALID':
        begin = [y_shape[0], y_shape[1]]
        size  = [x_shape[0]-y_shape[0], x_shape[1]-y_shape[0]]

    if padding == 'SAME':
        begin = [y_shape[0]/2-1, y_shape[1]/2-1]
        size  = [x_shape[0], x_shape[1]]

    z = tf.slice(z, begin, size)
    return z

def normxcorr2FFT(img, template, strides=[1,1,1,1], padding='VALID', eps = 0.01):

    #normalize and get variance
    dt = template - tf.reduce_mean(template)
    templatevariance = tf.reduce_sum(tf.square(dt))

    t1 = tf.ones(tf.shape(dt))
    tr = tf.reverse(dt, [0, 1])
    numerator = fftconvolve2d(img, tr, padding=padding) #tf.nn.conv2d (img, tr, strides=strides, padding=padding)

    localsum2 = fftconvolve2d(tf.square(img), t1, padding=padding)
    localsum = fftconvolve2d(img, t1, padding=padding)
    localvariance = localsum2-tf.square(localsum)/tf.reduce_prod(tf.to_float(tf.shape(template)))
    denominator = tf.sqrt(localvariance*templatevariance)

    #zero housekeeping
    numerator = tf.where(denominator<=tf.zeros(tf.shape(denominator)), tf.zeros(tf.shape(numerator), tf.float32), numerator)
    denominator = tf.where(denominator<=tf.zeros(tf.shape(denominator))+tf.constant(eps), tf.zeros(tf.shape(denominator), tf.float32)+tf.constant(eps), denominator)

    #Compute Pearson
    p = tf.div(numerator,denominator)
    p = tf.where(tf.is_nan(p, name=None), tf.zeros(tf.shape(p), tf.float32), p, name=None)

    return p

def normxcorr2(img, template, strides=[1,1,1,1], padding='SAME', eps = 0.001):

    #Do dim housekeeping
    img = tf.expand_dims(tf.expand_dims(img, 0),3)
    template = tf.expand_dims(tf.expand_dims(template,2),2)

    #normalize and get variance
    dt = template - tf.reduce_mean(template)
    templatevariance = tf.reduce_sum(tf.square(dt))

    t1 = tf.ones(tf.shape(dt))
    numerator = tf.nn.conv2d (img, dt, strides=strides, padding=padding)

    localsum2 = tf.nn.conv2d(tf.square(img), t1, strides=strides, padding=padding)
    localsum = tf.nn.conv2d(img, t1, strides=strides, padding=padding)
    localvariance = localsum2-tf.square(localsum)/tf.reduce_prod(tf.to_float(tf.shape(template)))
    denominator = tf.sqrt(localvariance*templatevariance)

    #zero housekeeping
    numerator = tf.where(denominator<=tf.zeros(tf.shape(denominator)), tf.zeros(tf.shape(numerator), tf.float32), numerator)
    denominator = tf.where(denominator<=tf.zeros(tf.shape(denominator)), tf.zeros(tf.shape(denominator), tf.float32)+tf.constant(eps), denominator)

    #Compute Pearson
    p = tf.div(numerator,denominator)
    p = tf.where(tf.is_nan(p, name=None), tf.zeros(tf.shape(p), tf.float32), p, name=None)
    p = tf.squeeze(p)
    return p
