# FilterFinder
Filter Finder for Normalized Cross Correlation on EM images of neurons

Setup
-----------

Clone the git
```
git clone https://github.com/seung-lab/FilterFinder.git
```

Please customize the pathways before running the docker.
```
sudo nvidia-docker run -it --net=host
    -v /usr/people/__username__/seungmount/research/Alembic/datasets/piriform/3_prealigned/:/FilterFinder/data/prealigned
    -v /usr/people/__username__/seungmount/Omni/TracerTasks/pinky/ground_truth/vol40/:/FilterFinder/data/aligned
    -v ..path_to/FilterFinder:/FilterFinder
    davidbun/workflow:latest bash
```

Prepare
-----------
To prepare the data run the following command which will consume couple hours to read the data and create the dataset for training
```
python prepare_data.py
```

Train
-----------
To train the model
```
python train.py
```

To visualize the training
-----------
```
Tensorboard --logdir=logs
```

Notebook
-----------

To run the model for interactive processing
```
./run_jupyter.sh
```
and go to FilterFinder/notebook/eval.ipynb

Notes
-----------
If there are some questions or something is broken contact me at davit@princeton.edu
