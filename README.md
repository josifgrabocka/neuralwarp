# Deep Warp

A supporting website for the paper "Grabocka et al., DeepWarp: Time-Series Similarity with Deep Warping Networks"

* We provide a direct link to an example dataset, located under folder shar
* The code was tested on Nvidia GTX 1080 Ti GPU with the Anaconda installation of Tensorflow

## Running the code

* Clone the repository locally "git clone https://github.com/josifgrabocka/deepwarp.git && cd deepwarp/"
* Install dependencies, tensorflow, scikit-learn, etc ...
* E.g. to run the method WarpedSiameseRNN for the dataset shar, call "python3 -u main.py shar/ WarpedSiameseRNN"
* The method option is one of SiameseRNN, WarpedSiameseRNN, CNNSim, CNNWarpedSim
* A checkpoint of the deep learning model will be saved under the "saved_models" folder every 1000 batches
* To test with the checkpoint created at the X-th step run "python3 -u inference.py test WarpedSiameseRNN saved_models/WarpedSiameseRNN__shar.ckpt-X shar/ 0.0 1.0"
* The last two parameters of the inference.py fila are the percentual starting index at the test set and the percentual size of the test instances to be classified, e.g. 0.0 1.0 means starting at 0%-th test series predict the target of 100% of the test set.
* The logs of the inference.py file will output the test series index and the classification accuracy so far.
