
# %%

def bias_variable(shape, identity = False):
    if identity:
        initial = tf.constant(0.0)
    else:
        initial = tf.constant(0.001)

    return tf.Variable(initial)

def weight_variable(shape, identity = False):
    #Build Convolution layer
    if identity:
        kernel_shape = np.array(shape)
        kernel_init = np.zeros(shape)
        kernel_init[kernel_shape[0]/2,shape[1]/2] = 1.0
    else:
        kernel_init = tf.truncated_normal(shape, stddev=0.01)

    return tf.Variable(kernel_init)

def convolve2d(x,y, padding = "VALID", strides=[1,1,1,1]):

    #Dim corrections
    if(len(x.get_shape())<4):
        x = tf.expand_dims(x, dim=0)

    if(len(x.get_shape())<4):
        x = tf.expand_dims(x, dim=3)

    if (len(y.get_shape())==2):
        y = tf.expand_dims(tf.expand_dims(y,  dim=2), dim=3)

    y = tf.to_float(y, name='ToFloat')

    o = tf.nn.conv2d(x, y, strides=strides, padding=padding)
    return tf.squeeze(o)

def softmax2d(image):
    # ASSERT:  if 0 is softmax 0 under all conditions
    shape = tuple(image.get_shape().as_list())
    image = tf.reshape(image, [shape[0]*shape[1]], name=None)
    soft_1D = tf.nn.softmax(image)
    soft_image = tf.reshape(soft_1D, shape, name=None)
    return soft_image

# Compute
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

def select(condition, x, y):
    return tf.cond(condition, lambda:x, lambda: y)

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

#Data Management
def getMetadata():
    import csv
    f = open("/FilterFinder/data/prealigned/registry.txt", 'rt')
    metadata = list(csv.reader(f, delimiter='\t'))
    return np.array(metadata)

def getDataSample(i, x, y, shape, delta,  metadata):
    #Check if not out of borders
    start = np.array([x+delta[0]-shape[0]/2, y+delta[1]-shape[0]/2])
    if(start[0]<0):
        start[0]=0
    if(start[1]<0):
        start[1]=0

    #Check if not exceeding the borders
    with h5py.File('/FilterFinder/data/prealigned/'+metadata[i,0]+'.h5', 'r') as hf:
        data = hf.get('img')
        sample = np.array(data[start[0]:start[0]+shape[0], start[1]:start[1]+shape[1]])
    return sample

def getSample(template_shape, source_shape, resize, metadata, j = 0, pos = (12334, 4121)):
    i = np.random.randint(metadata.shape[0]-1)
    x = np.random.randint(15000)+5000 #Pick a x coordinate and margin by 2000
    y = np.random.randint(10000)+10000 #Pick a y coordinate and margin by 2000
    delta = np.array([0,0])
    #print (i,x,y)
    if j>0:
        (x,y) = pos
        i = j

    template = np.transpose(getDataSample(i,x,y, resize*np.array(template_shape), delta, metadata))
    template = imresize(template, template_shape)
    delta = [-int(metadata[i+1,3])+int(metadata[i,3]), -int(metadata[i+1,2])+int(metadata[i,2])]
    source = np.transpose(getDataSample(i+1,x,y, resize*np.array(source_shape), delta, metadata))
    source = imresize(source, source_shape)
    #source = np.expand_dims(source, 0)

    #HACK: Check the quality
    n_temp = np.prod(template_shape)
    n_sour = np.prod(source_shape)
    if np.sum(template==0)>n_temp/4 or np.sum(template>250)>n_temp/4 or np.sum(source==0)>n_sour/4 or np.sum(template>250)>n_sour/4:
        return getSample(template_shape, source_shape, resize, metadata, j, pos)
    return (template-template.mean())/template.std(), (source-source.mean())/source.std()


def getAlignedData(train=True, test_size=60):
    with h5py.File('/FilterFinder/data/aligned/pinky_aligned_11184-11695_25018-25529_1-260.h5', 'r') as hf:
        #print('List of arrays in this file: \n', hf.keys())
        data = hf.get('img')
        np_data = np.array(data)
        shape = np_data.shape
        if train:
            np_data = np_data[0:shape[0]-test_size,:,:]
        else:
            np_data = np_data[shape[0]-test_size:shape[0],:,:]

        #print('Shape of the array dataset_1: \n', np_data.shape)
    return (np_data-np_data.mean())/np_data.std()

def getAlignedSample(template_shape, source_shape, data, j = 0):
    i = np.random.randint(data.shape[0]-1)
    x = np.random.randint(data.shape[1]-template_shape[0]) #Pick a x coordinate and margin by 2000
    y = np.random.randint(data.shape[2]-template_shape[1])
     #Pick a y coordinate and margin by 2000
    if j>0:
        (i,x,y) = (j-1,50,50)
    source = np.transpose(data[i,:,:])
    template = np.transpose(data[i+1,x:x+template_shape[0],y:y+template_shape[0]])
    return template, source

#Graphical
def show(img):
    fig = plt.figure()
    plt.imshow(img, cmap='Greys_r')
    plt.show()

def showLoss(loss_data, smoothing = 100):
    fig = plt.figure()
    hamming = smooth(loss_data, smoothing, 'hamming')
    iters = loss_data.shape[0]
    plt.plot(xrange(iters), loss_data, c='grey')
    plt.plot(xrange(hamming.shape[0]), hamming, c='r')

def showMultiLoss(loss_data, p1, p2, smoothing = 100):
    fig = plt.figure()
    hamming = smooth(loss_data, smoothing, 'hamming')
    iters = loss_data.shape[0]

    p1_max = smooth(p1, smoothing, 'hamming')
    p2_max = smooth(p2, smoothing, 'hamming')

    plt.plot(xrange(p1_max.shape[0]), p1_max, color = 'orange')
    plt.plot(xrange(p2_max.shape[0]), p2_max, color = 'wheat')
    plt.plot(xrange(iters), loss_data, c='grey')
    plt.plot(xrange(hamming.shape[0]), hamming, c='r')


def xcsurface(xc):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    N=xc.shape[0]
    M=xc.shape[1]
    X = np.arange(0, N, 1)
    Y = np.arange(0, M, 1)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure("xc") #,figsize=(10,10))
    plt.clf()

    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, xc, rstride=10, edgecolors="k",
                    cstride=10, cmap=cm.copper, alpha=1, linewidth=0,
                    antialiased=False)
    ax.set_zlim(-0.5, 2)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=10)

#Deprecated
def part(source, template):
    source = source.reshape(625,1)
    template = source.reshape(625,1)
    u_v = np.inner(source, template)
    u_2 = np.inner(source, source)
    v_2 =  np.inner(template, template)

    u_sigma = - np.square(np.inner(source, source)) + np.inner(np.square(source), np.square(source))
    v_sigma = - np.square(np.inner(template, template)) + np.inner(np.square(template), np.square(template))
    p = (u_v - u_2*v_2)/(np.sqrt(u_sigma*v_sigma+0.001))
    return p

def smooth(x,window_len=11,window='flat'):

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s= np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[window_len/2:y.shape[0]-window_len/2]
