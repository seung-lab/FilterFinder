import tensorflow as tf
import sys
import helpers
import loss
import metrics
from datetime import datetime

def premodel_mnist(g, hparams): #Mnist

    g.x = tf.placeholder(tf.float32, [50, 784])
    g.y = tf.placeholder(tf.float32, [50, 10])

    # First Layer
    W_conv1 = g.kernel_conv[0]
    b_conv1 = g.bias[0]
    x_image = tf.reshape(g.x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(helpers.convolve2d(x_image, W_conv1) + b_conv1)
    h_pool1 = helpers.max_pool_2x2(h_conv1)

    # Second Layer
    W_conv2 = g.kernel_conv[1]
    b_conv2 = g.bias[1]
    h_conv2 = tf.nn.relu(helpers.convolve2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = helpers.max_pool_2x2(h_conv2)

    W_fc1 = helpers.weight_variable([4*4*g.bias[1].get_shape().as_list()[0], 1024], summary=False)
    b_fc1 = helpers.bias_variable(shape=[1024])
    print(h_pool2.get_shape())
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*g.bias[1].get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, g.dropout)

    W_fc2 = helpers.weight_variable([1024, 10],  summary=False)
    b_fc2 = helpers.bias_variable([10])

    g.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    #Training steps
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=g.y_conv, labels=g.y))
    g.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(g.y_conv,1), tf.argmax(g.y,1))
    g.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    g.sess.run(init_op)

    return g

# import normxcorr2FFT, fftconvolve, bias, weights, loss
def model(g, hparams):
    return Unet(g,hparams)
    # Init Convolution Weights
    g.source_alpha = [g.image]
    g.template_alpha = [g.template]

    # Multilayer convnet
    g.kernel_conv = []
    g.bias = []

    n = hparams.kernel_shape.shape[0]

    # Setup Weights
    with tf.variable_scope('Filters'):
        for i in range(n):
            # Set variables
            g.bias.append(helpers.bias_variable(hparams.identity_init, shape=[hparams.kernel_shape[i,3]], name='bias_layer_'+str(i)))
            g.kernel_conv.append(helpers.weight_variable(hparams.kernel_shape[i], hparams.identity_init, name='layer_'+str(i)))

    with tf.variable_scope('Passes'):
        for i in range(n):
            g.source_alpha.append(helpers.convolve2d(g.source_alpha[i], g.kernel_conv[i], 'VALID', rate = hparams.dialation_rate))
            if not hparams.linear: g.source_alpha[i+1] = tf.tanh(g.source_alpha[i+1]+g.bias[i])

            g.template_alpha.append(helpers.convolve2d(g.template_alpha[i], g.kernel_conv[i], 'VALID', rate = hparams.dialation_rate))
            if not hparams.linear: g.template_alpha[i+1] = tf.tanh(g.template_alpha[i+1]+g.bias[i])

            # Max_pooling
            if i == 1 and False: #or i == 2:
                g.source_alpha[i+1] = helpers.max_pool_2x2(g.source_alpha[i+1])
                g.template_alpha[i+1] = helpers.max_pool_2x2(g.template_alpha[i+1])

            # Dropout Layer
            if i == 1 or i == 3:
                if hparams.dropout<1: g.source_alpha[i+1] = tf.nn.dropout(g.source_alpha[i+1], g.dropout)
                if hparams.dropout<1: g.template_alpha[i+1] = tf.nn.dropout(g.template_alpha[i+1], g.dropout)

        slice_source = tf.squeeze(tf.slice(g.source_alpha[-1], [0, 0, 0, 0], [-1, -1, -1, 1]))
        slice_template = tf.squeeze(tf.slice(g.template_alpha[-1], [0, 0, 0, 0], [-1, -1, -1, 1]))

        metrics.image_summary(slice_source, 'search_space')
        metrics.image_summary(slice_template, 'template')

    # Final Layer
    #g.bias.append(helpers.bias_variable(hparams.identity_init))
    #g.kernel_conv.append(helpers.weight_variable(hparams.kernel_shape[n-1], hparams.identity_init))

    #g.source_alpha.append(helpers.fftconvolve2d(g.source_alpha[n-1], g.kernel_conv[n-1], 'VALID'))
    #g.source_alpha[n] = tf.nn.sigmoid(g.source_alpha[n])+g.bias[n-1]

    #g.template_alpha.append(helpers.fftconvolve2d(g.template_alpha[n-1], g.kernel_conv[n-1], 'VALID'))
    #g.template_alpha[n] = tf.nn.sigmoid(g.template_alpha[n])+g.bias[n-1]

    return g

def Unet(g, hparams):
    # Init Convolution Weights
    g.source_alpha = [g.image]
    g.template_alpha = [g.template]

    # Multilayer convnet
    g.kernel_conv = []
    g.kernel_conv_right = []
    g.kernel_conv_up = []

    g.bias = []
    g.bias_right = []
    g.bias_up = []

    n = hparams.kernel_shape.shape[0]

    # Setup Weights
    with tf.variable_scope('Filters'):
        # Unet weights
        for i in range(n):

            #First layer
            shape = hparams.kernel_shape[i]
            g.kernel_conv = helpers.add_conv_weight_layer(g.kernel_conv, shape, g.bias) # Left
            if not i==(n-1): # Right
                shape[2] = 2*shape[3]
                g.kernel_conv_right = helpers.add_conv_weight_layer(g.kernel_conv_right, shape, g.bias_right)

            #Up layer
            if not i==(n-1):
                new_shape = [2,2, shape[3], 2*shape[3]]
                g.kernel_conv_up = helpers.add_conv_weight_layer(g.kernel_conv_up, new_shape, g.bias_up) # Up

            #Second layer
            shape[2] = shape[3]
            g.kernel_conv = helpers.add_conv_weight_layer(g.kernel_conv, shape, g.bias) # Left
            if not i==(n-1): g.kernel_conv_right = helpers.add_conv_weight_layer(g.kernel_conv_right, shape, g.bias_right) # Right

    count = len(g.kernel_conv)

    with tf.variable_scope('Passes'):
        # Down
        for i in range(count):
            g.source_alpha.append(helpers.convolve2d(g.source_alpha[-1], g.kernel_conv[i]))
            if not hparams.linear: g.source_alpha[-1] = tf.tanh(g.source_alpha[-1]+g.bias[i])

            g.template_alpha.append(helpers.convolve2d(g.template_alpha[-1], g.kernel_conv[i]))
            if not hparams.linear: g.template_alpha[-1] =tf.tanh(g.template_alpha[-1]+g.bias[i])

            # Max_pooling
            if i%2==0 and i!=count and i!=0: #or i == 2:
                g.source_alpha[-1] = helpers.max_pool_2x2(g.source_alpha[-1])
                g.template_alpha[-1] = helpers.max_pool_2x2(g.template_alpha[-1])

        print('left done')

        # Up
        for i in range(len(g.kernel_conv_up)):
            #Deconvolution
            g.source_alpha.append(helpers.deconv2d(g.source_alpha[-1], g.kernel_conv_up[-i-1], output_shape= g.source_alpha[count-2*(i+1)].get_shape().as_list()))
            g.template_alpha.append(helpers.deconv2d(g.template_alpha[-1], g.kernel_conv_up[-i-1], output_shape= g.template_alpha[count-2*(i+1)].get_shape().as_list()))

            #Concatination
            g.source_alpha[-1] = helpers.concat(g.source_alpha[count-2*(i+1)], g.source_alpha[-1])
            g.template_alpha[-1] = helpers.concat(g.template_alpha[count-2*(i+1)], g.template_alpha[-1])


            print('convolutions')
            print(g.source_alpha[-1].get_shape())
            #2 Convolutions
            for j in range(2):
                g.source_alpha.append(helpers.convolve2d(g.source_alpha[-1], g.kernel_conv_right[-2*i-2+j]))
                if not hparams.linear: g.source_alpha[-1] = tf.tanh(g.source_alpha[-1]+g.bias_right[-2*i-2+j])

                g.template_alpha.append(helpers.convolve2d(g.template_alpha[-1], g.kernel_conv_right[-2*i-2+j]))
                if not hparams.linear: g.template_alpha[-1] = tf.tanh(g.template_alpha[-1]+g.bias_right[-2*i-2+j])

        #Output convolution
        g.kernel_conv = helpers.add_conv_weight_layer(g.kernel_conv, [1,1,hparams.kernel_shape[0,3],1], g.bias)
        g.source_alpha.append(helpers.convolve2d(g.source_alpha[-1], g.kernel_conv[-1]))
        g.template_alpha.append(helpers.convolve2d(g.template_alpha[-1], g.kernel_conv[-1]))

        slice_source = tf.squeeze(tf.slice(g.source_alpha[-1], [0, 0, 0, 0], [-1, -1, -1, 1]))
        slice_template = tf.squeeze(tf.slice(g.template_alpha[-1], [0, 0, 0, 0], [-1, -1, -1, 1]))

        metrics.image_summary(slice_source, 'search_space')
        metrics.image_summary(slice_template, 'template')
    return g


class Graph(object):
    pass


def normxcorr(g, hparams):

    source = g.source_alpha[-1]
    template = g.template_alpha[-1]
    s_shape = source.get_shape().as_list()
    t_shape = template.get_shape().as_list()


    with tf.variable_scope('normxcor'):
        source = tf.transpose(source, [0,3,1,2])
        template = tf.transpose(template, [0,3,1,2])

        source = tf.reshape(source, [s_shape[0]*s_shape[3],s_shape[1], s_shape[2]])
        template = tf.reshape(template, [t_shape[0]*t_shape[3], t_shape[1], t_shape[2]])

        g.p = helpers.normxcorr2FFT(source, template) #get last convolved images

        p_shape = g.p.get_shape().as_list()

        g.p = tf.reshape(g.p, [s_shape[0], s_shape[3], p_shape[1], p_shape[2]])
        g.p = tf.reduce_sum(g.p, axis=[1])
        #g.p = tf.sqrt(tf.reduce_sum(tf.square(g.p), axis=[1])) # Take the norm
        metrics.image_summary(g.p, 'template_space')
    return g

def create_model(hparams, data, train = True):
    g = Graph()
    config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    if train:
        g.sess = tf.Session(config=config)
    else:
        g.sess = tf.InteractiveSession()

    # Write normalised cross-correlation and loss function
    with tf.variable_scope('input'):
        #if train:
        #    g.image, g.template = data.inputs(train, hparams)
        #else:
        g.image = tf.placeholder(tf.float32, shape=[hparams.batch_size, hparams.source_width, hparams.source_width])
        g.template = tf.placeholder(tf.float32, shape=[hparams.batch_size, hparams.template_width, hparams.template_width])
        g.dropout = tf.placeholder(tf.float32)

        # Add to metrics
        metrics.image_summary(g.image, 'search_space')
        metrics.image_summary(g.template, 'template_space')

    # Build the model
    g = model(g, hparams)
    g = normxcorr(g, hparams)
    g = loss.loss(g, hparams)

    # Decaying step
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay( hparams.learning_rate, global_step,
                                                hparams.decay_steps, hparams.decay, staircase=False)

    #g.train_step = tf.cond(g.to_update,
    #                            lambda: tf.train.AdamOptimizer(learning_rate).minimize(g.l, global_step=global_step),
    #                            lambda: tf.no_op(), name=None)
    g.train_step = tf.train.AdamOptimizer(learning_rate).minimize(g.l, global_step=global_step)  #  tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum = hparams.momentum,).minimize(g.l,  global_step=global_step) #

    g.merged = tf.summary.merge_all()
    g.saver = tf.train.Saver()

    if hparams.exp_name is None:
        now = datetime.now()
        g.id = now.strftime("%Y%m%d-%H%M%S")
    else:
        g.id = hparams.exp_name

    logdir = hparams.loging_dir + g.id + "/"
    g.train_writer = tf.summary.FileWriter(logdir + 'train', g.sess.graph)
    g.test_writer = tf.summary.FileWriter(logdir + 'test')

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    g.sess.run(init_op)

    g.coord = tf.train.Coordinator()
    g.threads = tf.train.start_queue_runners(sess=g.sess, coord=g.coord)

    if train==False:
        ckpt = tf.train.get_checkpoint_state(hparams.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            g.saver.restore(g.sess, ckpt.model_checkpoint_path)

    return g
