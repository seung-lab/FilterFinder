import numpy as np
import h5py
from scipy.misc import imresize
import os.path
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import scipy.ndimage
import helpers

TRAIN_FILE = 'bad_trainset_24000_612_324.tfrecords' #, train_bad_20, train, bad_across_section_10000
VALIDATION_FILE = 'validation.tfrecords'

#Data Management
class Data(object):
    def __init__(self, hparams, prepare= False):
        #self.mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
        if prepare:
            self.metadata = self.getMetadata(hparams)
        else:
            self.metadata = self.getMetadata(hparams)
            self.s_train, self.t_train = self.inputs(True, hparams)
            self.test = self.getBatch(hparams, hparams.pathset)
            #self.s_test, self.t_test = self.inputs(False, hparams)

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
        i = np.random.randint(metadata.shape[0]/2-2)
        x = np.random.randint(15000)+5000 #Pick a x coordinate and margin by 2000
        y = np.random.randint(10000)+10000 #Pick a y coordinate and margin by 2000
        delta = np.array([0,0])
        #print (i,x,y)
        if j>0:
            (x,y) = pos
            i = j
        template = np.transpose(self.getDataSample(i,x,y, resize*np.array(template_shape), delta, metadata))
        template = imresize(template, template_shape)
        delta = [-int(metadata[i+2,3])+int(metadata[i,3]), -int(metadata[i+2,2])+int(metadata[i,2])]
        source = np.transpose(self.getDataSample(i+2,x,y, resize*np.array(source_shape), delta, metadata))
        source = imresize(source, source_shape)
        #source = np.expand_dims(source, 0)

        #HACK: Check the quality of the images
        n_temp = np.prod(template_shape)
        n_sour = np.prod(source_shape)
        if np.sum(template==0)>n_temp/4 or np.sum(template>250)>n_temp/4 or np.sum(source==0)>n_sour/4 or np.sum(template>250)>n_sour/4:
            return self.getSample(template_shape, source_shape, resize, metadata, j, pos)

        return template/255.0, source/255.0

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

    def getTrainBatch(self):
        return self.test


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
      search = tf.decode_raw(features['search_raw'], tf.uint8) # Change to tf.int8
      search.set_shape([hparams.in_source_width*hparams.in_source_width])
      search = tf.reshape(search, [hparams.in_source_width, hparams.in_source_width])

      template = tf.decode_raw(features['template_raw'],tf.uint8) # Change to tf.int8
      template.set_shape([hparams.in_template_width*hparams.in_template_width])
      template = tf.reshape(template, [hparams.in_template_width, hparams.in_template_width])

      # OPTIONAL: Could reshape into a 28x28 image and apply distortions
      # here.  Since we are not applying any distortions in this
      # example, and the next step expects the image to be flattened
      # into a vector, we don't bother.

      # Rotation - Random Flip left, right, random, up down
      if hparams.flipping:
          distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
          search = self.image_distortions(search, distortions)
          template =  self.image_distortions(template, distortions)

      # Rotation by degree (rotate only single channel)
      if hparams.rotating:
          angle = tf.random_uniform([1], -hparams.degree, hparams.degree, dtype=tf.float32)
          search = helpers.rotate_image(search, angle)

      # Translation - Crop 712 - > 512 and 324 -> 224 ( At least 10 times bigger)
      search = tf.random_crop(search,  [hparams.source_width, hparams.source_width])
      template =  tf.random_crop(template, [hparams.template_width, hparams.template_width])



      # Convert from [0, 255] -> [-0.5, 0.5] floats.
      search = tf.cast(search, tf.float32) / 255
      template = tf.cast(template, tf.float32) / 255
      return search, template

    def image_distortions(self, image, distortions):

        distort_left_right_random = distortions[0]
        mirror = tf.cond(distort_left_right_random > 0.5, lambda: tf.constant([0,1]), lambda: tf.constant([0]))
        image = tf.reverse(image, mirror)

        distort_up_down_random = distortions[1]
        mirror = tf.cond(distort_up_down_random > 0.5, lambda: tf.constant([0,1]), lambda: tf.constant([0]))
        image = tf.reverse(image, mirror)

        return image

    def inputs(self, train, hparams, num_epochs=None):
      """Reads input data num_epochs times.
      Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
           train forever.
      Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
          in the range [0, 1].
        * labels is an int32 tensor with shape [batch_size] with the true label,
          a number in the range [0, mnist.NUM_CLASSES).
        Note that an tf.train.QueueRunner is added to the graph, which
        must be run using e.g. tf.train.start_queue_runners().
      """

      filename = os.path.join(hparams.data_dir, TRAIN_FILE)

      with tf.name_scope('input_provider'):
        filename_queue = tf.train.string_input_producer(
            [filename for x in range(100)], num_epochs=1)

        # Even when reading in multiple threads, share the filename
        # queue.
        search, template = self.read_and_decode(filename_queue, hparams)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.

        search_images, template_images = tf.train.shuffle_batch(
            [search, template], batch_size=hparams.batch_size, num_threads=2,
            capacity=1000 * hparams.batch_size,
            allow_smaller_final_batch=True,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

        return search_images, template_images


    def augment(self, search, template, hparams):
        search_shape = 512
        template_shape = 128
        aug_search = np.zeros((hparams.batch_size, search_shape, search_shape))
        aug_template = np.zeros((hparams.batch_size, template_shape, template_shape))
        ratio = 6

        # Translation
        for i in range(hparams.batch_size):
            #(x, y) = (np.random.randint(search.shape[1])/ratio, np.random.randint(search.shape[2])/ratio)
            #aug_search[i, :, :] = search[i, x:x+search_shape, y:y+search_shape]

            (x, y) = (0,0) #(np.random.randint(template.shape[1])/ratio, np.random.randint(template.shape[2])/ratio)
            aug_template[i, :, :] = template[i, x:x+template_shape, y:y+template_shape]

        return aug_search, aug_template

    def addNoise(self, image, template):
        width = 28*7
        for i in range(image.shape[0]):
            (x,y) = (np.random.randint(image.shape[1])/8, np.random.randint(image.shape[2])/8)
            (x,y) = (x+180,y+180)
            #(x,y) = (200,200)
            c = np.random.choice(9)
            blob =  np.multiply(image[i, x:x+width,y:y+width],self.getDigit(c, random=False))
            image[i, x:x+width,y:y+width] = blob
            template[i, 0:width,0:width] =  blob

        return image

    def dissimilar(self, images, templates):
        length = templates.shape[0]-1
        temp = np.array(templates[0])
        templates[0:length] = templates[1:length+1]
        templates[length] = temp
        return images, templates

    def getDigit(self, d, random = False):

        label = np.zeros((10))
        label[d] = 1
        length = self.mnist.train.images.shape[0]
        start = 0
        if random:
            start = np.random.choice(3*length/4)

        for i in range(length):
            if self.mnist.train.labels[i+start, d] == label[d]:
                return scipy.ndimage.zoom(1-self.mnist.train.images[i+start].reshape((28,28)), 7, order=0)

    def check_validity(self, search, template, hparams):
        t = np.array(template.shape)
        if np.any(np.sum(search<0.01, axis=(1,2)) >= t[1]*t[2]) or search.shape[0]<hparams.batch_size:
            return False
        return True
