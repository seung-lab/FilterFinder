import tensorflow as tf
import metrics
import numpy as np
from random import randint


def bias_variable(identity = False, initial=0.0, shape=(), name = 'bias'):
    if identity:
        initial = tf.constant(0.0, shape=shape)
    else:
        initial = tf.constant(initial, shape=shape)
    b = tf.Variable(initial)
    #metrics.variable_summaries(b)
    return b

def weight_variable(shape, identity = False, xavier = True,  name = 'conv', summary=True):
    #Build Convolution layer
    if identity:
        kernel_shape = np.array(shape)
        kernel_init = np.zeros(shape)
        kernel_init[kernel_shape[0]/2,shape[1]/2] = 0.125
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
    stringID = str(len(kernels))+'_'+str(len(kernels))
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
    #print('deconv2d')
    output_shape =[x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2]
    #print(output_shape)

    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

def resizeconv2d(x, W, stride=2, padding = "SAME"):
    x_shape = x.get_shape().as_list()
    x = tf.image.resize_images(x, size=[x_shape[1]*2, x_shape[2]*2], method=1, align_corners=True)

    x_out = convolve2d(x, W, padding="SAME")
    return x_out

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


### 1x1 Convolution
def conv_one_by_one(x):
    stringID = 'last'
    x_shape = x.get_shape().as_list()
    shape = [1,1,x_shape[-1], 1]
    identity_init = False

    b = bias_variable(identity_init, shape=[1], name='bias_layer_'+stringID)
    kernel = weight_variable(shape, identity_init, name='layer_'+stringID, summary=False)

    out = tf.sigmoid(convolve2d(x, kernel, padding='SAME')+b)
    #out = tf.squeeze(out)
    return out

def batch_normalization(x, y):


    return x, y

### FusionNet
def conv_block(x, y, kernels, bias, kernel_shape):
    kernels, bias = add_conv_weight_layer(kernels, bias, kernel_shape)

    x_out = tf.tanh(convolve2d(x, kernels[-1], padding='SAME')+bias[-1])
    y_out = tf.tanh(convolve2d(y, kernels[-1], padding='SAME')+bias[-1])

    #normalization
    #x_out, y_out = batch_normalization(x_out, y_out)

    return x_out, y_out

def deconv_block(x, y, kernels, bias, kernel_shape):

    kernels, bias = add_conv_weight_layer(kernels, bias, kernel_shape)

    kernels[-1] = tf.transpose(kernels[-1], [0,1,3,2])
    x_out = tf.tanh(deconv2d(x, kernels[-1], padding='SAME')+bias[-1])
    y_out = tf.tanh(deconv2d(y, kernels[-1], padding='SAME')+bias[-1])

    #x_out = tf.tanh(resizeconv2d(x, kernels[-1], padding='SAME')+bias[-1])
    #y_out = tf.tanh(resizeconv2d(y, kernels[-1], padding='SAME')+bias[-1])
    #x_out, y_out = batch_normalization(x_out, y_out)

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
        size  = [x_shape[0], x_shape[1]-y_shape[1], x_shape[2]-y_shape[2]]

    if padding == 'SAME':
        begin = [0, y_shape[1]/2-1, y_shape[2]/2-1]
        size  = x_shape #[-1, x_shape[0], x_shape[1]]

    z = tf.slice(z, begin, size)
    return z

def fftconvolve2d3d(x,y, padding):
    x_shape = x.get_shape().as_list()
    y_shape = y.get_shape().as_list()

    x = tf.transpose(x, [0,3,1,2])
    y = tf.transpose(y, [0,3,1,2])

    x = tf.reshape(x, [x_shape[0]*x_shape[3], x_shape[1], x_shape[2]])
    y = tf.reshape(y, [y_shape[0]*y_shape[3], y_shape[1], y_shape[2]])

    o = fftconvolve2d(x, y, padding)
    o_shape = o.get_shape().as_list()

    o = tf.reshape(o, [x_shape[0], x_shape[3], o_shape[1], o_shape[2]])
    o = tf.reduce_mean(o, axis=[1])
    o = tf.squeeze(o)
    return o

def fftconvolve3d(x, y, padding):
    # FIXME SAME will not work correctly
    # FIXME specifically designed for normxcorr (need to work more to make it general)
    # Read shapes
    x_shape = np.array(tuple(x.get_shape().as_list()), dtype=np.int32)
    y_shape = np.array(tuple(y.get_shape().as_list()), dtype=np.int32)
    # Construct paddings and pad
    x_shape[1:4] = x_shape[1:4]-1
    y_pad =  [[0,0], [0, x_shape[1]],[0, x_shape[2]], [0, x_shape[3]]]
    y_shape[1:4] = y_shape[1:4]-1
    x_pad = [[0,0], [0, y_shape[1]],[0, y_shape[2]], [0, y_shape[3]]]

    x = tf.pad(x, x_pad)
    y = tf.pad(y, y_pad)

    y = tf.cast(y, tf.complex64, name='complex_Y')
    x = tf.cast(x, tf.complex64, name='complex_X')

    convftt = tf.real(tf.ifft3d(tf.multiply(tf.fft3d(x), tf.fft3d(y), name='fft_mult')))

    print(convftt.get_shape())
    #Slice correctly based on requirements
    if padding == 'VALID':
        begin = [0, y_shape[1], y_shape[2],  y_shape[3]]
        size  = [x_shape[0], x_shape[1]-y_shape[1], x_shape[2]-y_shape[1], 1]

    if padding == 'SAME':
        begin = [0, y_shape[1]/2-1, y_shape[2]/2-1, y_shape[3]-1]
        size  = x_shape #[-1, x_shape[0], x_shape[1]]

    z = tf.slice(convftt, begin, size)
    z = tf.squeeze(z)
    return z

def normxcorr2FFT(img, template, strides=[1,1,1,1], padding='VALID', eps = 0.01):

    # Preprocessing for 2d or 3d normxocrr
    axis = [1,2]
    t_shape = template.get_shape().as_list()
    shape = t_shape[1]*t_shape[2]
    convolve = lambda x, y: fftconvolve2d(x, y, padding = padding)

    if len(t_shape)>3:
        axis = [1,2,3]
        shape *= t_shape[3]
        convolve = lambda x, y: fftconvolve3d(x, y, padding = padding)

    #normalize and get variance'
    dt = template - tf.reduce_mean(template, axis = axis, keep_dims = True)

    templatevariance = tf.reduce_sum(tf.square(dt), axis = axis, keep_dims = True)
    if len(t_shape)>3: templatevariance =  tf.squeeze(templatevariance, [3])

    t1 = tf.ones(tf.shape(dt))
    tr = tf.reverse(dt, axis)

    numerator = convolve(img, tr)
    localsum2 = convolve(tf.square(img), t1)
    localsum = convolve(img, t1)

    localvariance = localsum2-tf.square(localsum)/shape

    denominator = tf.sqrt(localvariance*templatevariance)

    #zero housekeeping
    numerator = tf.where(denominator<=tf.zeros(tf.shape(denominator)),
                            tf.zeros(tf.shape(numerator), tf.float32),
                            numerator)

    denominator = tf.where(denominator<=tf.zeros(tf.shape(denominator))+tf.constant(eps),
                            tf.zeros(tf.shape(denominator),tf.float32)+tf.constant(eps),
                            denominator)

    #Compute Pearson
    p = tf.div(numerator,denominator)
    p = tf.where(tf.is_nan(p, name=None), tf.zeros(tf.shape(p), tf.float32), p, name=None)

    return p

def rotate_image(image, angle):
    image = tf.expand_dims(image, dim=2)
    image = rotate_image_tensor(image, angle)
    image = tf.squeeze(image)
    return image

def rotate_image_tensor(image, angle, mode='black'):
    """
    BY zimmermc from stackoverflow
    Rotates a 3D tensor (HWD), which represents an image by given radian angle.

    New image has the same size as the input image.

    mode controls what happens to border pixels.
    mode = 'black' results in black bars (value 0 in unknown areas)
    mode = 'white' results in value 255 in unknown areas
    mode = 'ones' results in value 1 in unknown areas
    mode = 'repeat' keeps repeating the closest pixel known
    """
    s = image.get_shape().as_list()
    assert len(s) == 3, "Input needs to be 3D."
    assert (mode == 'repeat') or (mode == 'black') or (mode == 'white') or (mode == 'ones'), "Unknown boundary mode."
    image_center = [np.floor(x/2) for x in s]

    # Coordinates of new image
    coord1 = tf.range(s[0])
    coord2 = tf.range(s[1])

    # Create vectors of those coordinates in order to vectorize the image
    coord1_vec = tf.tile(coord1, [s[1]])

    coord2_vec_unordered = tf.tile(coord2, [s[0]])
    coord2_vec_unordered = tf.reshape(coord2_vec_unordered, [s[0], s[1]])
    coord2_vec = tf.reshape(tf.transpose(coord2_vec_unordered, [1, 0]), [-1])

    # center coordinates since rotation center is supposed to be in the image center
    coord1_vec_centered = coord1_vec - image_center[0]
    coord2_vec_centered = coord2_vec - image_center[1]

    coord_new_centered = tf.cast(tf.stack([coord1_vec_centered, coord2_vec_centered]), tf.float32)

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.dynamic_stitch([tf.constant(0), tf.constant(1), tf.constant(2), tf.constant(3)], [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)])
    rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
    coord_old_centered = tf.matmul(rot_mat_inv, coord_new_centered)

    # Find nearest neighbor in old image
    coord1_old_nn = tf.cast(tf.round(coord_old_centered[0, :] + image_center[0]), tf.int32)
    coord2_old_nn = tf.cast(tf.round(coord_old_centered[1, :] + image_center[1]), tf.int32)

    # Clip values to stay inside image coordinates
    if mode == 'repeat':
        coord_old1_clipped = tf.minimum(tf.maximum(coord1_old_nn, 0), s[0]-1)
        coord_old2_clipped = tf.minimum(tf.maximum(coord2_old_nn, 0), s[1]-1)
    else:
        outside_ind1 = tf.logical_or(tf.greater(coord1_old_nn, s[0]-1), tf.less(coord1_old_nn, 0))
        outside_ind2 = tf.logical_or(tf.greater(coord2_old_nn, s[1]-1), tf.less(coord2_old_nn, 0))
        outside_ind = tf.logical_or(outside_ind1, outside_ind2)

        coord_old1_clipped = tf.boolean_mask(coord1_old_nn, tf.logical_not(outside_ind))
        coord_old2_clipped = tf.boolean_mask(coord2_old_nn, tf.logical_not(outside_ind))

        coord1_vec = tf.boolean_mask(coord1_vec, tf.logical_not(outside_ind))
        coord2_vec = tf.boolean_mask(coord2_vec, tf.logical_not(outside_ind))


    coord_old_clipped = tf.cast(tf.transpose(tf.stack([coord_old1_clipped, coord_old2_clipped]), [1, 0]), tf.int32)

    # Coordinates of the new image
    coord_new = tf.transpose(tf.cast(tf.stack([coord1_vec, coord2_vec]), tf.int32), [1, 0])

    image_channel_list = tf.split(image, s[2], 1)

    image_rotated_channel_list = list()
    for image_channel in image_channel_list:
        image_chan_new_values = tf.gather_nd(tf.squeeze(image_channel), coord_old_clipped)

        if (mode == 'black') or (mode == 'repeat'):
            background_color = 0
        elif mode == 'ones':
            background_color = 1
        elif mode == 'white':
            background_color = 255

        image_rotated_channel_list.append(tf.sparse_to_dense(coord_new, [s[0], s[1]], image_chan_new_values,
                                                             background_color, validate_indices=False))

    image_rotated = tf.transpose(tf.stack(image_rotated_channel_list), [1, 2, 0])

    return image_rotated
