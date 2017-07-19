import gc
import h5py
import numpy as np
import src.visual as vis
from scipy.misc import imresize
from scipy import ndimage
path = '/FilterFinder/data/Dodam/'
path_s = '/FilterFinder/data/prealigned/'
from PIL import Image


def read((name, (x,y), size)):
    with h5py.File(path_s+name+'.h5', 'r') as hf:
        data = hf.get('img')
        sample = np.array(data[x:x+size, y:y+size])
    return resize(sample, (512, 512))

def resize(image, shape):
    return ndimage.interpolation.zoom(image, (1/3.0, 1/3.0), order=0)/255.0

def get_shape(name):
    hf = h5py.File(path_s+name+'.h5', 'r')
    return hf.get('img').shape

def create_dataset(name, shape=(30159, 40229)):
    hf = h5py.File(path+name+'.h5')
    dset = hf.create_dataset("img", shape, dtype='uint8')
    return dset

def write(array, (x, y), dset):
    dset[x:x+array.shape[0], y:y+array.shape[1], :] += array

def normalize(image):
    if len(image.shape)>2:
        for i in range(image.shape[2]):
            image[:,:,i] += abs(image[:,:,i].min())
            image[:,:,i] /= abs(image[:,:,i].max())
            image[:,:,i] *= 255.0
    else:
        image += abs(image.min())
        image /= abs(image.max())
        image *= 255

    return image

def save(image, name='out'):
    im = image+np.abs(image.min())
    im = 255*(im/im.max())
    im = np.squeeze(im)
    #print(im.shape)
    result = Image.fromarray(im.astype(np.uint8))
    result.save(name+'.jpg')


def f(x, y, pad, length):
    if (x>=pad and y>=pad) and (x<length-pad and y<length-pad):
        return 1

    #corners
    scale_down = 4*float(pad)
    if (x<pad and y<pad):
        return (x+y+1)/(scale_down)

    if (x>length-pad-1 and y>length-pad-1):
        return (2*length-x-y-1)/(scale_down)  #(length-x-1)/(2*float(pad)**2)

    if (x<pad and y>length-pad-1):
        return (length+x-y)/(scale_down) #y/(2*float(pad)**2)

    if (x>length-pad-1 and y<pad):
        return (length-x+y)/(scale_down)


    #edges
    if (x<pad) and (y>=pad and y<=length-pad-1):
        return (x+0.5)/float(pad)

    if (y<pad) and (x>=pad and x<=length-pad-1):
        return (y+0.5)/float(pad)

    if (x>pad) and (y>=pad and y<=length-pad-1):
        return (length-x-0.5)/float(pad)

    if (y>pad) and (x>=pad and x<=length-pad-1):
        return (length-y-0.5)/float(pad)

    return 0

def get_blend_map(pad, size):
    blend_map = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            blend_map[x,y] = f(x,y, pad, size)
    return blend_map
