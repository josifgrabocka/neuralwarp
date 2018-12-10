# Deep Warp

A supporting website for the paper "Grabocka et al., DeepWarp: Time-Series Similarity with Deep Warping Networks"

## Running the code

* Clone the repository locally "git clone https://github.com/josifgrabocka/deepwarp.git && cd deepwarp/"
* Install dependencies, tensorflow, scikit-learn, etc ...
* E.g. to run the method WarpedSiameseRNN for the dataset shar, call "python3 -u main.py shar/ WarpedSiameseRNN"
* The method option is one of SiameseRNN, WarpedSiameseRNN, CNNSim, CNNWarpedSim
* We provide a direct link to an example dataset, located under folder shar
* The code was tested on Nvidia GTX 1080 Ti GPU with the Anaconda installation of Tensorflow
