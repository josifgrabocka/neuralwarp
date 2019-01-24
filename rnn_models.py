import tensorflow as tf
import numpy as np
from sim_model import AbstractSimModel
import time

class RNNAbstractSimModel(AbstractSimModel):

    def __init__(self, config):
        AbstractSimModel.__init__(self, config)

        self.cells_bwd, self.cells_bwd = None, None
        self.h_fw, self.h_bw = None, None
        self.name = 'RNNSingleAbstractSimModel'

    # the rnn decoder produces the forward and backward activations
    def create_encoder(self):

        with tf.variable_scope("RNNEncoder"):

            print('Creating RNN encoder')
            cells_fwd_list = []
            for num_cells in self.config['uniwarp:rnn_encoder_layers']:
                print("Forward RNN layer with", num_cells, "cells")
                cells_fwd_list.append(tf.nn.rnn_cell.LSTMCell(num_units=num_cells, activation=tf.nn.tanh))
                self.cells_fwd = tf.nn.rnn_cell.MultiRNNCell(cells_fwd_list, state_is_tuple=True)

            cells_bwd_list = []
            for num_cells in self.config['uniwarp:rnn_encoder_layers']:
                print("Backward RNN layer with", num_cells, "cells")
                cells_bwd_list.append(tf.nn.rnn_cell.LSTMCell(num_units=num_cells, activation=tf.nn.tanh))
                self.cells_bwd = tf.nn.rnn_cell.MultiRNNCell(cells_bwd_list, state_is_tuple=True)

            # create a dynamic rnn with a specified sequence length
            # which outputs the activations and states of the last LSTM's layer for each time index
            (outputs, state_fw, state_bw) = tf.nn.static_bidirectional_rnn(
                cell_fw=self.cells_fwd,
                cell_bw=self.cells_bwd,
                inputs=tf.unstack(self.X_batch, axis=1),
                dtype=tf.float32)
            # stack the outputs list into a tensor
            self.h = tf.stack(outputs)

            self.h = tf.layers.batch_normalization(self.h, training=self.is_training,
                                                   name='RNNActivationBatchNorm')

            self.h = tf.layers.dropout(self.h, rate=self.config['uniwarp:dropout_rate'],
                                              training=self.is_training, name='RNNActivationDropOut')


# the siamese rnn model
class SiameseRNN(RNNAbstractSimModel):
    # constructor
    def __init__(self, config):
        RNNAbstractSimModel.__init__(self, config)
        self.name = 'SiameseRNN'

    def create_encoder(self):
        RNNAbstractSimModel.create_encoder(self)

        with tf.variable_scope("SiameseRNNEncoder"):
            self.pair_dists = tf.map_fn(lambda pair_idx: self.dist_pair(pair_idx),
                                        tf.range(self.config['model:num_batch_pairs'], dtype=tf.int32),
                                        back_prop=True,
                                        name='PairWiseDistMap',
                                        dtype=tf.float32)

    def dist_pair(self, pair_ixd):
            return tf.losses.absolute_difference(self.h[:, 2*pair_ixd, :], self.h[:, 2*pair_ixd+1, :])

    def create_similarity(self):
        with tf.variable_scope("SiameseRNNSimilarity"):
            # the rbf similarity
            self.pred_similarities = tf.exp(-self.pair_dists, name='SiameseRNNSim')

# the warped siamese rnn model
class WarpedSiameseRNN(SiameseRNN):
    # constructor
    def __init__(self, config):
        SiameseRNN.__init__(self, config)
        self.name = 'WarpedSiameseRNN'
        self.is_first_dist_pair_call = True

    # redefine the distance between a pair of instances
    def dist_pair(self, pair_ixd):

        with tf.variable_scope("WarpedSiameseRNNDistPair") as scope:

            # unless it is the first call, then reuse the variables of the scope
            if self.is_first_dist_pair_call:
                self.is_first_dist_pair_call = False
            else:
                scope.reuse_variables()

            # T x K tensors, for T the latent time indices and K the number of RNN cells (encoder length)
            A = self.h[:, 2*pair_ixd, :]
            B = self.h[:, 2*pair_ixd+1, :]
            # the first indices [0,0,0,...,0,1,1,1,...,1,2,2,2,...]
            idx_A = tf.range(A.shape[0])
            idx_A = tf.tile(idx_A, [A.shape[0]])
            # the second indices [0,1,2,...,B.shape[0],0,1,2,...,B.shape[0],0,1,2,...]
            idx_B = tf.range(B.shape[0])
            idx_B = tf.reshape(idx_B, [-1, 1])
            idx_B = tf.tile(idx_B, [1, B.shape[0]])
            idx_B = tf.reshape(idx_B, [-1])
            # gather the features for the indices
            A_expanded = tf.gather(A, idx_A)
            B_expanded = tf.gather(B, idx_B)

            # concatenate the two feature tensor to serve as the input for the warping weight neural network
            AB_concat = tf.concat([A_expanded, B_expanded], axis=1, name='ConcatenatedPairwiseIndices')
            # define the warping neural network
            warp_weights = AB_concat
            for num_units in self.config['uniwarp:warp_nn_layers']:
                print('Adding Warping NN layer with ', num_units, 'neurons')
                warp_weights = tf.layers.dense(inputs=warp_weights, activation=tf.nn.relu, units=num_units)

            # a final linear layer for the warping weights output in [0, 1]
            warp_weights = tf.layers.dense(inputs=warp_weights, activation=tf.nn.sigmoid, units=1)

            # the squared euclidean distance of all pairs
            A_minus_B_square = tf.abs(tf.subtract(A_expanded, B_expanded))
            pairs_dists = tf.expand_dims(tf.reduce_mean(A_minus_B_square, axis=1), axis=-1, name='PairsDists')

            # the warped distances
            warped_dists = tf.multiply(pairs_dists, warp_weights, name="WarpedSiameseRNN")

            return tf.reduce_mean(warped_dists)
