
# %%

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



#Compute max{p_!A}-max{p_A}
def loss(p, radius, eps= 0.001, ltype='diff'):
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

    #Get the second peak and return
    p_max_2 = tf.reduce_max(tf.mul(mask_p,p))
    mask_p = tf.mul(mask_p,p)


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
        _, ls, p_max_c, p_max_c_2 = sess.run([train_step, l, p_max, p_max_2], feed_dict={image: s, temp: t})

        loss[i] = np.absolute(ls)
        p_max_c1[i] = p_max_c
        p_max_c2[i] = p_max_c_2

        #Evaluate
        if i%epoch_size==0:
            b = time.time()
            error[i/epoch_size], er_p_max_c1[i/epoch_size], er_p_max_c2[i/epoch_size] = test(num_test_steps, source_shape, template_shape)
            print ("step: %g, test set average: %g, time %g"%(i,error[i/epoch_size], b-a))
            a = time.time()
        #print("step %d, maximizing %g: max_p %g max_p_2 %g, time %g"%(i, -ls, p_max_c, p_max_c_2, b-a))
    #showLoss(loss, 100)
    #showLoss(error, 1)

    showMultiLoss(loss, p_max_c1, p_max_c2, 100)
    showMultiLoss(error, er_p_max_c1, er_p_max_c2, 20)

    return loss, error, p_max_c1, p_max_c2, er_p_max_c1, er_p_max_c2

def test(num_test_steps, source_shape, template_shape, aligned=True):
    sum_dist = 0
    sum_p1 = 0
    sum_p2 = 0
    for i in range(num_test_steps):
        if aligned:
            t,s = getAlignedSample(template_shape, source_shape, test_set, i)
        ls, p_1, p_2 = sess.run([l, p_max, p_max_2], feed_dict={image: s, temp: t})
        sum_dist = sum_dist + np.absolute(ls)
        sum_p1 = sum_p1 + p_1
        sum_p2 = sum_p2 + p_2

    return sum_dist/num_test_steps, sum_p1/num_test_steps, sum_p2/num_test_steps

def evaluate(deterministic = True, source_shape=source_shape, template_shape=template_shape, aligned=True):
    sum_dist = 0
    if aligned:
        t,s = getAlignedSample(template_shape, source_shape, test_set, deterministic)
    norm, filt, s_f, t_f, P_1, P_2 = sess.run([p, kernel, source_alpha, template_alpha, p_max, p_max_2], feed_dict={image: s, temp: t})

    print (np.mean(filt))
    print (np.mean(np.absolute(filt)))

    xcsurface(norm)
    print(P_1, P_2)
    show(filt)
    show(s)
    show(s_f)
    show(t)
    show(t_f)
