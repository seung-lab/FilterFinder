import tensorflow as tf
import subprocess
import sys
import itertools
import json
import random

# Parameters for loss types
mean_over_batch = {"True", "False"}
loss_type = {"dist", "ratio"}
loss_form = {"log", "minus", "inverse" }
argnames= ["mean_over_batch", "loss_type", "loss_form"]


# Params for Architecture
losses= [['True', 'ratio', 'minus'], ['False', 'dist', 'log']]
number_of_layers = [2, 3, 4, 5, 6]
kernel_size = [16, 7, 5, 3]
channels = [1, 2, 4]

#Try later
#dilation = [1,2]
#dropout = [0.8,1]
learning_rate = [0.01, 0.001, 0.0001]

argnames= ["mean_over_batch", "loss_type", "loss_form", "kernel_shape", "dialation_rate"]

#archs_perm = [losses, number_of_layers, kernel_type, channels]


def main(unusedargs):
    architecture_experiment()


def loss_experiment():
    print('Running..')
    params = list(itertools.product(*loss_perm))

    for param in params:
        script = ["python train.py"]
        i = 0
        name = "loss="
        for argname in argnames:
            script.append("--"+argname+"="+param[i])
            name += '_'+param[i]
            i = i + 1
        script.append("--exp_name="+str(name))
        script.append("--steps=1000")
        script = ' '.join(script)
        print(script)
        subprocess.call(script, shell=True)


# Experiment for architectures
def architecture_experiment():
    kernel_shapes = []
    for i in range (30):
        kernel_shapes.append(construct_kernel(random.choice(number_of_layers)))

    archs_perm = [losses, kernel_shapes, learning_rate]
    params = list(itertools.product(*archs_perm))
    argnames= ["mean_over_batch", "loss_type", "loss_form"]
    #print(len(params))
    for param in params:
        script = ["python train.py"]
        i = 0
        name = "arch_l" + str(len(param[1]))+"=["
        for row in param[1]:
            name += str(row[0])+","+str(row[3])+"-"
        name +="]"
        for argname in argnames:
            script.append("--"+argname+"="+param[0][i])
            name += '_'+param[0][i]
            i = i + 1

        script.append("--kernel_shape="+json.dumps(param[1]).replace(" ", ""))
        script.append("--learning_rate="+str(param[2]))
        script.append("--exp_name="+str(name))
        script.append("--steps=800")
        script = ' '.join(script)
        #print(name)
        subprocess.call(script, shell=True)



def construct_kernel(num_layer):

    def calc_channel(k_size, coef):
        return 25*coef/(k_size)

    k_size = random.choice(kernel_size)
    if num_layer == 1:
        return [[k_size, k_size, 1, 1]]

    k_last_size = random.choice(kernel_size)
    coef = random.choice(channels)
    channel = calc_channel(k_last_size, coef)
    kernel_shape = [[k_size, k_size, 1, channel]]

    if num_layer == 2:
        new_layer = [k_last_size, k_last_size, channel, 1]
        kernel_shape.append(new_layer)
        return kernel_shape

    if num_layer>2:
        for i in range(num_layer-2):
            old_channel = channel
            coef = random.choice(channels)
            k_size = random.choice(kernel_size)
            channel = calc_channel(k_size, coef)

            new_layer = [k_size, k_size, old_channel, channel]
            kernel_shape.append(new_layer)

        new_layer = [k_last_size, k_last_size, channel, 1]
        kernel_shape.append(new_layer)
        return kernel_shape


if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
