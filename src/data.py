import numpy as np
import h5py
from scipy.misc import imresize

#Data Management
class Data(object):
    def __init__(self, hparams):
        self.metadata = self.getMetadata(hparams)

    def getMetadata(self, hparams):
        import csv
        f = open(hparams.metadata_dir, 'rt')
        metadata = list(csv.reader(f, delimiter='\t'))
        return np.array(metadata)

    def getDataSample(self, i, x, y, shape, delta,  metadata):
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

    def getSample(self, template_shape, source_shape, resize, metadata, j = 0, pos = (12334, 4121)):
        template_shape, source_shape, resize,
        i = np.random.randint(metadata.shape[0]-1)
        x = np.random.randint(15000)+5000 #Pick a x coordinate and margin by 2000
        y = np.random.randint(10000)+10000 #Pick a y coordinate and margin by 2000
        delta = np.array([0,0])
        #print (i,x,y)
        if j>0:
            (x,y) = pos
            i = j

        template = np.transpose(self.getDataSample(i,x,y, resize*np.array(template_shape), delta, metadata))
        template = imresize(template, template_shape)
        delta = [-int(metadata[i+1,3])+int(metadata[i,3]), -int(metadata[i+1,2])+int(metadata[i,2])]
        source = np.transpose(self.getDataSample(i+1,x,y, resize*np.array(source_shape), delta, metadata))
        source = imresize(source, source_shape)
        #source = np.expand_dims(source, 0)

        #HACK: Check the quality of the images
        n_temp = np.prod(template_shape)
        n_sour = np.prod(source_shape)
        if np.sum(template==0)>n_temp/4 or np.sum(template>250)>n_temp/4 or np.sum(source==0)>n_sour/4 or np.sum(template>250)>n_sour/4:
            return self.getSample(template_shape, source_shape, resize, metadata, j, pos)

        return template/256.0, source/256.0


    def getAlignedData(self, train=True, test_size=60):
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

    def getAlignedSample(self, template_shape, source_shape, data, j = 0):
        i = np.random.randint(data.shape[0]-1)
        x = np.random.randint(data.shape[1]-template_shape[0]) #Pick a x coordinate and margin by 2000
        y = np.random.randint(data.shape[2]-template_shape[1])
         #Pick a y coordinate and margin by 2000
        if j>0:
            (i,x,y) = (j-1,50,50)
        source = np.transpose(data[i,:,:])
        template = np.transpose(data[i+1,x:x+template_shape[0],y:y+template_shape[0]])
        return template, source
