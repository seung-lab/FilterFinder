
# %%

#construct the model, init variables and return the computation
def model(img, tmp, kernel_shape, identity = False):

    # Init Convolution Weights
    kernel = weight_variable(kernel_shape, identity)
    kernel_2 = weight_variable(kernel_shape/2, identity)

    bias_1 = bias_variable(identity)
    bias_2 = bias_variable(identity)

    # First Layer
    source_alpha = fftconvolve2d(image, kernel, 'VALID')
    source_alpha = tf.nn.relu(source_alpha+bias_1)

    template_alpha = fftconvolve2d(tmp, kernel, 'VALID')
    template_alpha = tf.nn.relu(template_alpha+bias_1)

    # Second Layer
    source_alpha = fftconvolve2d(source_alpha, kernel_2, 'VALID')
    source_alpha = tf.nn.relu(source_alpha+bias_2)

    template_alpha = fftconvolve2d(template_alpha, kernel_2, 'VALID')
    template_alpha = tf.nn.relu(template_alpha+bias_2)

    return  kernel, kernel_2, source_alpha, template_alpha


#Compute max{p_!A}-max{p_A}
def loss(p, radius, eps= 0.001, ltype='diff', softmax=False):
    #Get maximum p and mask the point
    p_max = tf.reduce_max(p)
    mask_p = p>p_max-tf.constant(eps)

    #Design the shape of the mask
    mask = tf.ones([radius*2,radius*2], tf.float32)
    mask = tf.expand_dims(mask, 2)
    mask_p = tf.expand_dims(tf.expand_dims(tf.cast(mask_p, tf.float32), 0), 3)

    #Dilate to have square with the center of maximum point and then flip
    mask_p = tf.nn.dilation2d(mask_p, mask, [1,1,1,1], [1,1,1,1], 'SAME')
    mask_p = tf.to_float(mask_p)<=tf.constant(1, dtype='float32')#tf.select(mask_p>1, tf.zeros(tf.shape(mask_p), tf.float32), tf.ones(tf.shape(mask_p), tf.float32))
    mask_p = tf.squeeze(tf.cast(mask_p, dtype='float32'))

    # Care about second distance
    p_2 = tf.multiply(mask_p,p)
    if softmax:
        # Get the weighted average p_2
        soft_mask_p = softmax2d(p_2)
        p_max_2 = tf.reduce_sum(tf.multiply(soft_mask_p,p_2))
    else:
        #Get the second peak and return
        p_max_2 = tf.reduce_max(p_2)

    mask_p = tf.multiply(mask_p,p)

    if ltype == 'diff':
        lossum = p_max_2 - p_max

    elif ltype == 'ratio':
        lossum = -p_max/(p_max_2+eps)

    elif  ltype == 'diffratio':
        lossum = lambd*p_max/(p_max_2+eps) + (p_max_2 - p_max)

    return lossum, p_max,  p_max_2, mask_p #Negative in order to minimize

def train(num_steps, source_shape, template_shape, aligned=True):
    loss = np.zeros(num_steps)
    p_max_c1 = np.zeros(num_steps)
    p_max_c2 = np.zeros(num_steps)

    error = np.zeros(num_steps/epoch_size)
    er_p_max_c1 = np.zeros(num_steps/epoch_size)
    er_p_max_c2 = np.zeros(num_steps/epoch_size)

    a = time.time()
    for i in range(num_steps):

        #Check id data is aligned
        if aligned:
            t,s = getAlignedSample(template_shape, source_shape, train_set)
        else:
            t,s = getSample(template_shape, source_shape, resize, metadata)

        #Train step
        _, ls, p_max_c, p_max_c_2 = sess.run([train_step, l, p_max, p_max_2],
                                             feed_dict={image: s, temp: t})

        loss[i] = np.absolute(ls)
        p_max_c1[i] = p_max_c
        p_max_c2[i] = p_max_c_2
        #print("step: %g, p_1: %g, p_2: %g"%(i, p_max_c, p_max_c_2))
        #Evaluate
        if i%epoch_size==0:
            b = time.time()
            error[i/epoch_size], er_p_max_c1[i/epoch_size], er_p_max_c2[i/epoch_size] = test(num_test_steps, source_shape, template_shape, aligned)
            print ("step: %g, test set average: %g, time %g"%(i,error[i/epoch_size], b-a))
            a = time.time()
        #print("step %d, maximizing %g: max_p %g max_p_2 %g, time %g"%(i, -ls, p_max_c, p_max_c_2, b-a))
    #showLoss(loss, 100)
    #showLoss(error, 1)

    showMultiLoss(loss, p_max_c1, p_max_c2, 100)
    showMultiLoss(error, er_p_max_c1, er_p_max_c2, 20)

    return loss, error, p_max_c1, p_max_c2, er_p_max_c1, er_p_max_c2

def test(num_test_steps, source_shape, template_shape, aligned=True, pos = (12334, 4121)):
    sum_dist = 0
    sum_p1 = 0
    sum_p2 = 0
    steps = num_test_steps
    pathset = [ (120,9900, 11000), (20, 9900, 11000),
                (60, 16000, 17000),(70, 16000, 17000),
                (400, 8500, 27000),(400, 7000, 27000),
                (300, 7000, 21500),(151, 4500, 5000),
                (51, 18000, 9500), (52, 18000, 7500),
                (55, 18000, 7500), (60, 18100, 8400)]
    if ~aligned:
        steps = len(pathset)

    for i in range(steps):
        if aligned:
            t,s = getAlignedSample(template_shape, source_shape, test_set, i)
        else:
            t,s = getSample(template_shape, source_shape, resize, metadata, pathset[i][0], pathset[0][1:3])
        ls, p_1, p_2 = sess.run([l, p_max, p_max_2], feed_dict={image: s, temp: t})
        sum_dist = sum_dist + np.absolute(ls)
        sum_p1 = sum_p1 + p_1
        sum_p2 = sum_p2 + p_2

    return sum_dist/num_test_steps, sum_p1/num_test_steps, sum_p2/num_test_steps

def evaluate(deterministic, source_shape, template_shape, aligned=True, pos=(12334, 4121)):
    sum_dist = 0
    if aligned:
        t,s = getAlignedSample(template_shape, source_shape, test_set, deterministic)
    else:
        t,s = getSample(template_shape, source_shape, resize, metadata, deterministic, pos)
    norm, filt, filt_2, s_f, t_f, P_1, P_2 = sess.run([p, kernel, kernel_2, source_alpha, template_alpha, p_max, p_max_2], feed_dict={image: s, temp: t})

    #print (np.mean(filt))
    #print (np.mean(np.absolute(filt)))
    print(P_1, P_2)
    xcsurface(norm)
    show(norm)
    print(filt_2)
    #print((kernel_shape[0]/2,kernel_shape[1]/2))
    #filt[32,32] = 0
    #filt_2[16,16] = 0
    show(filt)
    show(filt_2)
    show(s)
    show(s_f)
    show(t)
    show(t_f)
