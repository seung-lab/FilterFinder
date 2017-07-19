import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import gc
import h5py
import numpy as np
import src.visual as vis
from scipy.misc import imresize
from scipy import ndimage
path = '/FilterFinder/data/Dodam/'
path_s = '/FilterFinder/data/prealigned/'
from PIL import Image

import tensorflow as tf
import src.model as models
import src.data as d
import src.metrics as metrics
import src.training as training
import src.loss as loss
import src.visual as vis
import hyperparams
import numpy as np
import src.helpers as helpers
import time
import util

from pathos.multiprocessing import ProcessPool, ThreadPool

print('Loading the model...')
pathset = [ (120,9900, 11000), (20, 9900, 11000),
            (60, 16000, 17000),(70, 16000, 17000),
            (400, 8500, 27000),(400, 7000, 27000),
            (300, 7000, 21500),(151, 4500, 5000),
            (51, 18000, 9500), (52, 18000, 7500),
            (55, 18000, 7500), (60, 18100, 8400)]

hparams = hyperparams.create_hparams()
data = d.Data(hparams)
model = models.create_model(hparams, data,train = False)


#Global Parameters
scale = 3
pool = ThreadPool()
#Setup blend_map
blend_map = util.get_blend_map(256,512)
blend_map = np.stack([blend_map, blend_map, blend_map, blend_map], axis=2).astype(np.float16)


# Process by batches
def process(name, (x,y), mset, blend_map,  offset = 4000):
    model_run =[model.source_alpha]
    t1 = time.time()

    inputs = pool.map(util.read, [(name, (x+i*scale*256,y),scale*512) for i in range(8)])
    #print(inputs)
    feed_dict ={model.image: inputs,
                model.template: np.zeros((8, 160,160)),
                model.dropout: 1}

    args = model.sess.run(model_run,feed_dict=feed_dict)

    images = args[0][-1][:]
    images = [images[i][:,:,:] for i in range(8)]

    images_new = pool.map(post_process, images)
    #print(images_new[0].dtype)
    i = 0
    for image in images_new:
        x_temp = x+i*scale*256
        mset[x_temp-offset:x_temp-offset+image.shape[0], y-offset:y-offset+image.shape[1], :] += image[:,:,:]
        i += 1
    return mset

def post_process(image):
    image = util.normalize(image)
    image = np.multiply(image, blend_map).astype(np.uint8)
    image = ndimage.interpolation.zoom(image, (3,3,1), order=0)
    return image

def process_slice(name):
    step_x = 8*3*256
    step_y = 3*256

    shape = (18432+768, 18432, 4)
    shape_origin = (util.get_shape(name)[0],util.get_shape(name)[1], 4)
    mset = np.zeros(shape, np.uint8)

    print(mset.shape)

    for x in range(shape[0]//(step_x)):
        for y in range(shape[1]//(step_y)-1):
            t1 = time.time()
            mset = process(name, (x*step_x+4000, y*step_y+4000), mset, blend_map)
            t2 = time.time()
            #print((x,y), t2-t1)
    t1 = time.time()

    dset = util.create_dataset(name[0:2]+'50'+name[2:], shape=(4, 15001, 15001))
    dset[:,:,:] = np.transpose(mset[1000:16001,1000:16001, :], (2,0,1))
    t2 = time.time()
    #print('writing',t2-t1)


#process_slice("1,70_prealigned")
done = []
def process_all():
    for i in range(1):
        for x in range(1,97):
            if x+2 in done:
                print('pass',i+1,x+2)
                continue
            try:
                #gc.collect()
                t1 = time.time()
                process_slice(str(i+1)+','+str(x+2)+'_'+'prealigned')
                done.extend((i+1, x+2))
                t2 = time.time()
                print((i+1,x+2), t2-t1)
            except:
                print('err',i+1,x+2)

process_all()
