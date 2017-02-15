import numpy as np
import h5py
from scipy.misc import imresize
import os.path
import tensorflow as tf

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

#Data Management
class Data(object):
    def __init__(self, hparams, prepare= False):
        if prepare:
            self.metadata = self.getMetadata(hparams)
        else:
            self.s_train, self.t_train = self.inputs(True, hparams)


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

        # Pick a y coordinate and margin by 2000
        if j>0:
            (i,x,y) = (j-1,50,50)
        source = np.transpose(data[i,:,:])
        template = np.transpose(data[i+1,x:x+template_shape[0],y:y+template_shape[0]])
        return template, source

    def getBatch(self, hparams, pathset = []):
        # Set shapes
        search_shape = [hparams.batch_size, hparams.source_width, hparams.source_width]
        template_shape = [hparams.batch_size, hparams.template_width, hparams.template_width]

        search_space = np.zeros(search_shape)
        template = np.zeros(template_shape)

        for i in range(hparams.batch_size):
            if pathset==[]:
                template[i], search_space[i] = self.getSample(template_shape[1:3], search_shape[1:3], hparams.resize, self.metadata)
            else:
                template[i], search_space[i] = self.getSample(template_shape[1:3], search_shape[1:3], hparams.resize, self.metadata,  pathset[i][0], pathset[i][1:3])
        return template, search_space

    def getBatch_v2(self, hparams):

        with tf.Session() as sess:

            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            #for i in range(1000):
            example, l = sess.run([image, label])
            #    print (example,l)
            coord.request_stop()
            coord.join(threads)

        return sess.run(self.s_train, self.t_train)

    # Functions below modified from here https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
    def read_and_decode(self, filename_queue, hparams):
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read(filename_queue)
      features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
              'search_raw': tf.FixedLenFeature([], tf.string),
              'template_raw': tf.FixedLenFeature([], tf.string),
          })

      # Convert from a scalar string tensor (whose single string has
      # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
      # [mnist.IMAGE_PIXELS].
      search = tf.decode_raw(features['search_raw'], tf.float64)
      search.set_shape([hparams.source_width*hparams.source_width])
      search = tf.reshape(search, [hparams.source_width, hparams.source_width])

      template = tf.decode_raw(features['template_raw'], tf.float64)
      template.set_shape([hparams.template_width*hparams.template_width])
      template = tf.reshape(template, [hparams.template_width, hparams.template_width])
      # OPTIONAL: Could reshape into a 28x28 image and apply distortions
      # here.  Since we are not applying any distortions in this
      # example, and the next step expects the image to be flattened
      # into a vector, we don't bother.

      # Convert from [0, 255] -> [-0.5, 0.5] floats.
      search = tf.cast(search, tf.float32) #* (1. / 255)
      template = tf.cast(template, tf.float32)# * (1. / 255)
      return search, template


    def inputs(self, train, hparams):
      """Reads input data num_epochs times.
      Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
           train forever.
      Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
          in the range [-0.5, 0.5].
        * labels is an int32 tensor with shape [batch_size] with the true label,
          a number in the range [0, mnist.NUM_CLASSES).
        Note that an tf.train.QueueRunner is added to the graph, which
        must be run using e.g. tf.train.start_queue_runners().
      """
      filename = os.path.join(hparams.data_dir,
                              TRAIN_FILE if train else VALIDATION_FILE)

      with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=1)

        # Even when reading in multiple threads, share the filename
        # queue.
        search, template = self.read_and_decode(filename_queue, hparams)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        search_images, template_images = tf.train.shuffle_batch(
            [search, template], batch_size=hparams.batch_size, num_threads=2,
            capacity=1000 + 3 * hparams.batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

        return search_images, template_images
