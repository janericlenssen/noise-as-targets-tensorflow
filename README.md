# noise-as-targets-tensorflow
Noise-as-targets representation learning for cifar10. 
Implementation based on the arxiv-paper "Unsupervised Learning by Predicting Noise" by Bojanowski and Joulin: https://arxiv.org/abs/1704.05310

- Trains the encoder to map each example to one predefined target representation from the n-dimensional unit-sphere
- Optimizes the assignment of target vectors every 3 epochs using the hungarian algorithm
- Trains a supervised MLP every x epochs to test the discriminative power of the learned representation

Training:
1. Set model_dir and data_dir parameters in cifar10_natenc_train.py
2. Run cifar10_natenc_train.py

Get neighbors:
1. Set model_dir and out_path parameters in cifar10_natenc_getNeighbors.py
2. Run cifar10_natenc_getNeighbors.py

Current status: Work in progress, best cifar10 test classification accuracy after 50 epochs of unsupervised training: 43,8%, not clear how to chose parameters, discussions, feedback or suggestions are welcome!

Example results of nearest neighbor search on the learned representation (for Cifar 10 test examples):

![Examples for nearest neighbor search](neighbors.png?raw=true "Examples for nearest neighbor search")

First column: queried images, second to sixth columns: nearest neighbors.
