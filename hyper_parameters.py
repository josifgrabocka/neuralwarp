import numpy as np
import json

class HyperParams:

    def __init__(self):
        pass

    def get_uniwarp_config(self, argv):

        config = {}
        config['optimizer:num_epochs'] = 1000000
        config['model:num_batch_pairs'] = 100
        config['uniwarp:length'] = 1024
        config['uniwarp:rnn_encoder_layers'] = [256, 128, 64]
        config['uniwarp:warp_nn_layers'] = [64, 16, 1]
        config['uniwarp:eta'] = 0.0001
        config['uniwarp:max_grad_norm'] = 10.0
        config['uniwarp:lambda'] = 0.0
        config['uniwarp:cnn_encoder_layers'] = [1024, 256, 64]
        config['uniwarp:cnn_kernel_lengths'] = [5, 5, 3]
        config['uniwarp:cnn_strides'] = [2, 1, 1]
        config['uniwarp:dropout_rate'] = 0.05
        config['uniwarp:enable_batch_normalization'] = True
        config['dataset:num_channels'] = 1

        return config

    def restore(file_path):
        return json.loads(file_path)

