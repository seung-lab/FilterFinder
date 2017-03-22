from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class Data(object):
    def __init__(self, hparams, prepare= False):

        self.mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)

    def fake_data_noisy(self, hparams):
        s_w = hparams.source_width
        t_w = hparams.template_width

        s = np.zeros((hparams.batch_size, s_w, s_w))
        t = np.zeros((hparams.batch_size, t_w, t_w))
        for i in range(hparams.batch_size):
            s[i,:,:] = np.random.choice(np.arange(0, 2), p=[0.4, 0.6], size=[s_w, s_w])

            start = s_w/2 - t_w/2
            end = s_w/2 + t_w/2
            t[i] = np.array(s[i, start:end, start:end])
            t_prime = np.random.choice(np.arange(0, 2), p=[0.7, 0.3], size=[t_w, t_w])
            t_prime_2 = np.random.choice(np.arange(0, 2), p=[0.8, 0.2], size=[t_w, t_w])
            t[i] = np.multiply(t[i],t_prime)
            t[i] += np.multiply((1-t[i]),t_prime_2)

        return s, t

    def fake_data(self, hparams):

        seven = self.getDigit(8)
        one = self.getDigit(2)

        s_w = hparams.source_width
        t_w = hparams.template_width
        d_w = one.shape[0]

        s = np.zeros((hparams.batch_size, s_w, s_w))
        t = np.zeros((hparams.batch_size, t_w, t_w))

        start_t = t_w/2 - d_w/2
        end_t = t_w/2 + d_w/2

        for i in range(hparams.batch_size):
            digits = []
            for j in range(10):
                digits.append(self.getDigit(j, random=False))

            c = np.random.choice(len(digits))
            s[i,:,:] = self.construct_digit_texture(digits[:c] + digits[c+1 :], c, hparams)

            t[i,start_t:end_t,start_t:end_t] = self.getDigit(c, random=True) #np.multiply(one,np.random.choice(np.arange(0, 2), p=[0.3, 0.7], size=[d_w, d_w]))
        s = np.asarray(s>0.5,  dtype=int)
        return s, t

    def construct_digit_texture(self, digits, c, hparams):

        texture = np.zeros((hparams.source_width, hparams.source_width))
        cols = hparams.source_width/digits[0].shape[0]
        width = digits[0].shape[0]

        for x in range(cols):
            for y in range(cols):
                choice = np.random.choice(len(digits))
                texture[x*width:(x+1)*width, y*width:(y+1)*width] = digits[choice][:,:]

        (x,y) = np.random.choice(cols), np.random.choice(cols)
        texture[x*width:(x+1)*width, y*width:(y+1)*width] = self.getDigit(c, random=False)
        return texture

    def getDigit(self, d, random = False):

        label = np.zeros((10))
        label[d] = 1
        length = self.mnist.train.images.shape[0]
        start = 0
        if random:
            start = np.random.choice(3*length/4)

        for i in range(length):
            if self.mnist.train.labels[i+start, d] == label[d]:
                return self.mnist.train.images[i+start].reshape((28,28))
