import tensorflow as tf
import numpy as np
def loss(g, hparams, eps = 0.001):

    #Get maximum p and mask the point
    g.p_max = tf.reduce_max(g.p, axis=[1,2], keep_dims=True)
    g.mask_p = tf.cast(g.p>g.p_max-tf.constant(hparams.eps), tf.float32)

    #Design the shape of the mask
    gp_shape = np.array(tuple(g.p.get_shape().as_list()), dtype=np.int32)
    g.mask = tf.ones([hparams.radius*2, hparams.radius*2, gp_shape[3]], tf.float32)
    #g.mask = tf.expand_dims(g.mask, 2)
    #g.mask_p = tf.expand_dims(tf.cast(g.mask_p, tf.float32), 3)

    #Dilate to have square with the center of maximum point and then flip
    print(g.mask)
    print(g.mask_p)

    g.mask_p = tf.nn.dilation2d(g.mask_p, g.mask, [1,1,1,1], [1,1,1,1], 'SAME')
    g.mask_p = tf.to_float(g.mask_p)<=tf.constant(1, dtype='float32')
    g.mask_p = tf.squeeze(tf.cast(g.mask_p, dtype='float32'))

    #print(g.mask_p.get_shape())
    # Care about second distance
    g.p_2 = tf.multiply(g.mask_p,g.p)

    if hparams.softmax:
        # Get the weighted average p_2
        g.soft_mask_p = softmax2d(g.p_2)
        g.p_max_2 = tf.reduce_sum(tf.multiply(g.soft_mask_p,g.p_2), axis=[1,2])
    else:
        #Get the second peak and return
        g.p_max_2 = tf.reduce_max(g.p_2, axis=[1,2], keep_dims=True)

    g.mask_p = tf.multiply(g.mask_p, g.p)

    if hparams.loss_type == 'dist':
        g.l =  (g.p_max-g.p_max_2)

    elif hparams.loss_type == 'ratio':
        g.l = g.p_max/(g.p_max_2+eps)

    if hparams.loss_form == 'minus':
        g.l = -g.l
    elif hparams.loss_form == 'inverse':
        g.l = 1/(g.l+eps)
    elif hparams.loss_form == 'log':
        g.l = -tf.log(g.l)

    #Add more sanity checks
    g.to_update = tf.logical_and(tf.reduce_all(tf.is_finite(g.l)), tf.greater(tf.reduce_max(g.l), -0.1))
    #g.l = tf.where(tf.is_finite(g.l), g.l, tf.zeros(tf.shape(g.l), tf.float32), name=None)

    print('~~~~~~ Combine Difference per layer with combined layer ~~~~~~~')
    print(g.l)
    g.full_loss = g.l
    g.last_layer = tf.slice(g.l, [0, 0,0, hparams.output_layer-1], [-1,-1,-1, 1])
    print(g.last_layer.get_shape())
    g.l = tf.reduce_mean(g.l, axis=3, keep_dims=True)
    g.l = 0.3*g.l + 0.7*g.last_layer

    print('~~~~~~ Cross_similarity ~~~~~~')
    print(g.cross_similarity.get_shape())
    print(g.l)
    g.l = g.l - 0.1*tf.abs(g.cross_similarity)

    if hparams.mean_over_batch == True:
        g.l = tf.reduce_mean(g.l)
    else:
        g.l = tf.reduce_max(g.l)



    g.l = tf.where(g.similar>0, g.l, tf.reduce_mean(g.p_max), name=None) #tf.abs(g.l)
    #g.l = tf.where(g.similar>0, g.l, tf.abs(g.l), name=None)



    g.p_max = tf.reduce_mean(g.p_max)
    g.p_max_2 = tf.reduce_mean(g.p_max_2)

    with tf.name_scope('loss'):
        tf.summary.scalar('loss', g.l)
        tf.summary.scalar('distance', g.p_max-g.p_max_2)
        tf.summary.scalar('max',  g.p_max)
        tf.summary.scalar('second_max',  g.p_max_2)

    return g
