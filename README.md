# Neural Warp

A supporting website for the paper "NeuralWarp: Time-Series Similarity with Warping Networks" https://arxiv.org/abs/1812.08306

* We provide a direct link to one example dataset, located under folder shar
* The code was tested on Nvidia GTX 1080 Ti GPU with the Anaconda installation of Python 3 dependencies

## Training NeuralWarp

* Clone the repository locally "git clone https://github.com/josifgrabocka/neuralwarp.git && cd neuralwarp/"
* Install dependencies, python3, numpy, tensorflow, scikit-learn, etc ... We have only tested the package installation through Anaconda.
* To train the method WarpedSiameseRNN for the dataset shar, call "python3 -u main.py shar/ WarpedSiameseRNN"
* The method option is one of unwarped {SiameseRNN, CNNSim} or warped {WarpedSiameseRNN, CNNWarpedSim} architectures
* A checkpoint of the deep learning model will be saved under the "saved_models" folder every 1000 batches
* The learning algorithm should print to the console the averge similarity loss after every 1000 batches
* Warning: Training Deep Learning models on plain CPUs machines can be prohibitive in terms of runtime.

## Testing NeuralWarp

* To test with the checkpoint created at the X-th step run "python3 -u inference.py test WarpedSiameseRNN saved_models/WarpedSiameseRNN__shar.ckpt-X shar/ 0.0 1.0"
* The last two parameters of the inference.py file are the percentual starting index at the test set and the percentual size of the test instances to be classified, e.g. 0.0 1.0 means starting at 0%-th test series predict the target of 100% of the test set, or 0.05 0.2 means from the 5%-th until 25%-th of test instances
* The logs of the inference.py file will output the test series index and the classification accuracy so far.

* We uploaded our trained checkpoints on the shar dataset from all the four Siamese CNN and RNN models (warped and unwarped) under the trained_models folder. E.g. to test with a trained WarpedSiameseRNN network run "python3 -u inference.py test WarpedSiameseRNN trained_models/WarpedSiameseRNN_shar_0.ckpt-999 shar/ 0.0 1.0"
