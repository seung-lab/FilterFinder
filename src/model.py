import tensorflow as tf
import sys
import helpers
import loss
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
    for i in range(n):
        # Set variables
        g.bias.append(helpers.bias_variable(hparams.identity_init, name='bias_layer'+str(i)))
        g.kernel_conv.append(helpers.weight_variable(hparams.kernel_shape[i], hparams.identity_init, name='conv_layer'+str(i)))

        g.source_alpha.append(helpers.convolve2d(g.source_alpha[i], g.kernel_conv[i], 'VALID', rate = hparams.dialation_rate))
        g.source_alpha[i+1] = tf.nn.sigmoid(g.source_alpha[i+1])+g.bias[i]

        g.template_alpha.append(helpers.convolve2d(g.template_alpha[i], g.kernel_conv[i], 'VALID', rate = hparams.dialation_rate))
        g.template_alpha[i+1] = tf.nn.sigmoid(g.template_alpha[i+1])+g.bias[i]

        # Dropout Layer
        g.source_alpha[i+1] = tf.nn.dropout(g.source_alpha[i+1], g.dropout)
        g.template_alpha[i+1] = tf.nn.dropout(g.template_alpha[i+1], g.dropout)

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

def create_model(hparams):
    g = Graph()
    g.sess = tf.Session()

    # Write normalised cross-correlation and loss function
    g.image = tf.placeholder(tf.float32, shape=[hparams.source_width, hparams.source_width])
    g.template = tf.placeholder(tf.float32, shape=[hparams.template_width, hparams.template_width])
    g.dropout = tf.placeholder(tf.float32)

    g.source_alpha = tf.placeholder(tf.float32, shape=[hparams.source_width, hparams.source_width])
    g.template_alpha = tf.placeholder(tf.float32, shape=[hparams.template_width, hparams.template_width])

    # Build the model
    g = model(g, hparams)

    # Setup the loss
    g.p = helpers.normxcorr2FFT(g.source_alpha[-1], g.template_alpha[-1]) #get last convolved images
    g = loss.loss(g, hparams)
    g.train_step = tf.train.MomentumOptimizer(learning_rate=hparams.learning_rate, momentum = hparams.momentum).minimize(g.l)

    g.merged = tf.summary.merge_all()
    g.saver = tf.train.Saver()

    now = datetime.now()
    g.id = now.strftime("%Y%m%d-%H%M%S")
    logdir = hparams.loging_dir + g.id + "/"
    g.train_writer = tf.summary.FileWriter(logdir + 'train', g.sess.graph)
    g.test_writer = tf.summary.FileWriter(logdir + 'test')
    g.sess.run(tf.global_variables_initializer())

    return g
