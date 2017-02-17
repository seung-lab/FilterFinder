import tensorflow as tf
import subprocess
import sys
import itertools

# Move to JSON file or smth similiar
mean_over_batch = {"True", "False"}
loss_type = {"dist", "ratio"}
loss_form = {"log", "minus", "inverse"}
# softmax over p_max
argnames= ["mean_over_batch", "loss_type", "loss_form"]
s=[mean_over_batch, loss_type, loss_form]

def main(unusedargs):
    print('Running..')
    params = list(itertools.product(*s))

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

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
