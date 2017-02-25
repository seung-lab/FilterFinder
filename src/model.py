import tensorflow as tf
import sys
import helpers
import loss
import metrics
from datetime import datetime

# import normxcorr2FFT, fftconvolve, bias, weights, loss
def model(g, hparams):

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
            g.bias.append(helpers.bias_variable(hparams.identity_init, name='bias_layer_'+str(i)))
            g.kernel_conv.append(helpers.weight_variable(hparams.kernel_shape[i], hparams.identity_init, name='layer_'+str(i)))

    with tf.variable_scope('Passes'):
        for i in range(n):
            g.source_alpha.append(helpers.convolve2d(g.source_alpha[i], g.kernel_conv[i], 'VALID', rate = hparams.dialation_rate))
            g.source_alpha[i+1] = tf.tanh(g.source_alpha[i+1])#-g.bias[i]

            g.template_alpha.append(helpers.convolve2d(g.template_alpha[i], g.kernel_conv[i], 'VALID', rate = hparams.dialation_rate))
            g.template_alpha[i+1] = tf.tanh(g.template_alpha[i+1])#-g.bias[i]

            # Dropout Layer
            #g.source_alpha[i+1] = tf.nn.dropout(g.source_alpha[i+1], g.dropout)
            #g.template_alpha[i+1] = tf.nn.dropout(g.template_alpha[i+1], g.dropout)

        metrics.image_summary(g.source_alpha[-1], 'search_space')
        metrics.image_summary(g.template_alpha[-1], 'template')

    # Final Layer
    #g.bias.append(helpers.bias_variable(hparams.identity_init))
    #g.kernel_conv.append(helpers.weight_variable(hparams.kernel_shape[n-1], hparams.identity_init))

    #g.source_alpha.append(helpers.fftconvolve2d(g.source_alpha[n-1], g.kernel_conv[n-1], 'VALID'))
    #g.source_alpha[n] = tf.nn.sigmoid(g.source_alpha[n])+g.bias[n-1]

    #g.template_alpha.append(helpers.fftconvolve2d(g.template_alpha[n-1], g.kernel_conv[n-1], 'VALID'))
    #g.template_alpha[n] = tf.nn.sigmoid(g.template_alpha[n])+g.bias[n-1]

    return g

class Graph(object):
    pass

def create_model(hparams, data, train = True):
    g = Graph()
    config = tf.ConfigProto(log_device_placement = False)
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

    # Setup the loss
    with tf.variable_scope('normxcor'):
        g.p = helpers.normxcorr2FFT(g.source_alpha[-1], g.template_alpha[-1]) #get last convolved images
        metrics.image_summary(g.p, 'template_space')

    g = loss.loss(g, hparams)
    g.train_step = tf.train.MomentumOptimizer(learning_rate=hparams.learning_rate, momentum = hparams.momentum).minimize(g.l)

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
