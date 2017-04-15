import tensorflow as tf
import metrics
import numpy as np
from random import randint

def bias_variable(identity = False, shape=(), name = 'bias'):
    if identity:
        initial = tf.constant(0.0, shape=shape)
    else:
        initial = tf.constant(0.0, shape=shape)
    b = tf.Variable(initial)
    #metrics.variable_summaries(b)
    return b

def weight_variable(shape, identity = False, xavier = True,  name = 'conv', summary=True):
    #Build Convolution layer
    if identity:
        kernel_shape = np.array(shape)
        kernel_init = np.zeros(shape)
        kernel_init[kernel_shape[0]/2,shape[1]/2] = 1.0
        weight = tf.Variable(kernel_init, name=name)
    elif xavier:
        weight = tf.get_variable(name, shape=tuple(shape),
            initializer=tf.contrib.layers.xavier_initializer())
    else:
        kernel_init = tf.random_normal(shape, stddev=0.01)
        weight = tf.Variable(kernel_init, name=name)

    if summary:
        metrics.kernel_summary(weight, name)
    return weight

def add_conv_weight_layer(kernels, bias, kernel_shape, identity_init= False):
    # Set variables
    stringID = str(len(kernels))+'_'+str(randint(10000,99999))
    bias.append(bias_variable(identity_init, shape=[kernel_shape[3]], name='bias_layer_'+stringID))
    kernels.append(weight_variable(kernel_shape, identity_init, name='layer_'+stringID, summary=False))
    return kernels, bias

def convolve2d(x,y, padding = "VALID", strides=[1,1,1,1], rate = 1):

    #Dim corrections
    if(len(x.get_shape())==2):
        x = tf.expand_dims(x, dim=0)
        x = tf.expand_dims(x, dim=3)

    elif(len(x.get_shape())==3 and x.get_shape()[0].value == x.get_shape()[1].value ):
        x = tf.expand_dims(x, dim=0)
    elif(len(x.get_shape())==3):
        x = tf.expand_dims(x, dim=3)

    if (len(y.get_shape())==2):
        y = tf.expand_dims(tf.expand_dims(y,  dim=2), dim=3)
    elif(len(y.get_shape())==3):
        y = tf.expand_dims(y, dim=2)

    y = tf.to_float(y, name='ToFloat')
    if rate>1:
        o = tf.nn.atrous_conv2d(x, y, rate=rate, padding=padding)
    else:
        o = tf.nn.conv2d(x, y, strides=strides, padding=padding)
    return o

def deconv2d(x, W, stride=2, padding = "SAME"):
    x_shape = x.get_shape().as_list()
    print('deconv2d')
    output_shape =[x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2]
    print(output_shape)

    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

def crop(x, shape):
    old_shape = x.get_shape().as_list()
    pad = (old_shape[1] - shape[1])/2
    return tf.slice(x, [0,pad,pad,0], [-1, shape[1], shape[2],-1])

def concat(x, y):
    shape = y.get_shape().as_list()
    x = crop(x, shape)
    print('concat')
    print(x.get_shape())
    print(y.get_shape())
    return tf.concat([x, y], axis= 3)

def softmax2d(image):
    # ASSERT:  if 0 is softmax 0 under all conditions
    shape = tuple(image.get_shape().as_list())
    image = tf.reshape(image, [-1, shape[0]*shape[1]], name=None)
    soft_1D = tf.nn.softmax(image)
    soft_image = tf.reshape(soft_1D, shape, name=None)
    return soft_image

def max_pool_2x2(x):
    if(len(x.get_shape())==3):
        x = tf.expand_dims(x, dim=3)
    o = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    return o

### FusionNet
def conv_block(x, y, kernels, bias, kernel_shape):
    kernels, bias = add_conv_weight_layer(kernels, bias, kernel_shape)

    x_out = tf.tanh(convolve2d(x, kernels[-1], padding='SAME')+bias[-1])
    y_out = tf.tanh(convolve2d(y, kernels[-1], padding='SAME')+bias[-1])

    return x_out, y_out

def deconv_block(x, y, kernels, bias, kernel_shape):
    kernels, bias = add_conv_weight_layer(kernels, bias, kernel_shape)

    x_out = deconv2d(x, kernels[-1], padding='SAME') #tf.tanh( ... +bias[-1)
    y_out = deconv2d(y, kernels[-1], padding='SAME') #tf.tanh( ... +bias[-1])

    return x_out, y_out

def residual_block(x, y, kernels, bias, kernel_shape):
    x_1, y_1 = conv_block(x, y, kernels, bias, kernel_shape)
    kernel_shape[2] = kernel_shape[3]
    x_2, y_2 = conv_block(x_1, y_1, kernels, bias, kernel_shape)
    x_3, y_3 = conv_block(x_2, y_2, kernels, bias, kernel_shape)
    x_4, y_4 = conv_block(x_3, y_3, kernels, bias, kernel_shape)

    x_5, y_5 = conv_block(x_4+x_1, y_4+y_1, kernels, bias, kernel_shape)
    return x_5, y_5

def fftconvolve2d(x, y, padding="VALID"):
    #return convolve2d(x,y)
    """
    x and y must be real 2-d tensors.

    mode must be "SAME" or "VALID".
    Input is x=[batch, width, height] and kernel is [batch, width, height]

    need to add custom striding
    """
    # Read shapes
    x_shape = np.array(tuple(x.get_shape().as_list()), dtype=np.int32)
    y_shape = np.array(tuple(y.get_shape().as_list()), dtype=np.int32)

    # Check if they are 2D add one artificial batch layer
    # Do the same for kernel seperately

    # Construct paddings and pad
    x_shape[1:3] = x_shape[1:3]-1
    y_pad =  [[0,0], [0, x_shape[1]],[0, x_shape[2]]]
    y_shape[1:3] = y_shape[1:3]-1
    x_pad = [[0,0], [0, y_shape[1]],[0, y_shape[2]]]

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
        begin = [0, y_shape[1], y_shape[2]]
        size  = [x_shape[0], x_shape[1]-y_shape[1], x_shape[2]-y_shape[1]]

    if padding == 'SAME':
        begin = [0, y_shape[1]/2-1, y_shape[2]/2-1]
        size  = x_shape #[-1, x_shape[0], x_shape[1]]

    z = tf.slice(z, begin, size)
    return z

def normxcorr2FFT(img, template, strides=[1,1,1,1], padding='VALID', eps = 0.01):

    #normalize and get variance
    dt = template - tf.reduce_mean(template, axis = [1,2], keep_dims = True)
    templatevariance = tf.reduce_sum(tf.square(dt), axis = [1,2], keep_dims = True)

    t1 = tf.ones(tf.shape(dt))
    tr = tf.reverse(dt, [1, 2])
    numerator = fftconvolve2d(img, tr, padding=padding)

    localsum2 = fftconvolve2d(tf.square(img), t1, padding=padding)
    localsum = fftconvolve2d(img, t1, padding=padding)

    shape = template.get_shape()[1].value*template.get_shape()[2].value
    localvariance = localsum2-tf.square(localsum)/shape
    denominator = tf.sqrt(localvariance*templatevariance)

    #zero housekeeping
    numerator = tf.where(denominator<=tf.zeros(tf.shape(denominator)),
                            tf.zeros(tf.shape(numerator), tf.float32),
                            numerator)

    denominator = tf.where(denominator<=tf.zeros(tf.shape(denominator))+tf.constant(eps),
                            tf.zeros(tf.shape(denominator),
                            tf.float32)+tf.constant(eps),
                            denominator)

    #Compute Pearson
    p = tf.div(numerator,denominator)
    p = tf.where(tf.is_nan(p, name=None), tf.zeros(tf.shape(p), tf.float32), p, name=None)

    return p
