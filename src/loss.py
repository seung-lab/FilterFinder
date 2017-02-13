import tensorflow as tf

def loss(g, hparams):

    #Get maximum p and mask the point
    g.p_max = tf.reduce_max(g.p)
    g.mask_p = g.p>g.p_max-tf.constant(hparams.eps)

    #Design the shape of the mask
    g.mask = tf.ones([hparams.radius*2,hparams.radius*2], tf.float32)
    g.mask = tf.expand_dims(g.mask, 2)
    g.mask_p = tf.expand_dims(tf.expand_dims(tf.cast(g.mask_p, tf.float32), 0), 3)

    #Dilate to have square with the center of maximum point and then flip
    g.mask_p = tf.nn.dilation2d(g.mask_p, g.mask, [1,1,1,1], [1,1,1,1], 'SAME')
    g.mask_p = tf.to_float(g.mask_p)<=tf.constant(1, dtype='float32')
    g.mask_p = tf.squeeze(tf.cast(g.mask_p, dtype='float32'))

    # Care about second distance
    g.p_2 = tf.multiply(g.mask_p,g.p)
    if hparams.softmax:
        # Get the weighted average p_2
        g.soft_mask_p = softmax2d(g.p_2)
        g.p_max_2 = tf.reduce_sum(tf.multiply(g.soft_mask_p,g.p_2))
    else:
        #Get the second peak and return
        g.p_max_2 = tf.reduce_max(g.p_2)

    g.mask_p = tf.multiply(g.mask_p,g.p)

    if hparams.loss_type == 'dist':
        g.l =  -(g.p_max-g.p_max_2)

    elif hparams.loss_type == 'ratio':
        g.l = -g.p_max/(g.p_max_2+eps)

    elif hparams.loss_type == 'inv_dist':
        g.l = 1/((g.p_max-g.p_max_2)+eps)

    elif hparams.loss_type == 'dist_ratio':
        g.l = hparams.lambd*g.p_max/(g.p_max_2+eps) + (g.p_max_2 - g.p_max)

    with tf.name_scope('loss'):
        tf.summary.scalar('loss',  g.l)
        tf.summary.scalar('max',  g.p_max)
        tf.summary.scalar('second_max',  g.p_max_2)

    return g
