import tensorflow as tf
from sim_model import AbstractSimModel
import time

class CNNAbstractSimModel(AbstractSimModel):

    def __init__(self, config):
        AbstractSimModel.__init__(self, config)
        self.last_feature_map = None
        self.name = 'CNNAbstractSimModel'

    # the rnn decoder produces the forward and backward activations
    def create_encoder(self):

        with tf.name_scope("CNNEncoder"):
            feature_map = self.X_batch

            for num_filters, kernel_length, stride in zip(self.config['uniwarp:cnn_encoder_layers'],
                                                            self.config['uniwarp:cnn_kernel_lengths'],
                                                            self.config['uniwarp:cnn_strides']):

                feature_map = tf.layers.conv1d(inputs=feature_map, filters=num_filters, padding='VALID',
                                               kernel_size=kernel_length,
                                               strides=stride)

                feature_map = tf.layers.batch_normalization(feature_map, training=self.is_training)

                feature_map = tf.nn.relu(feature_map)

                print('Add CNN layer', feature_map, ' kernel_length', kernel_length)

            # pass the last feaure map through drop out
            self.h = feature_map

            # add drop out
            self.h = tf.layers.dropout(self.h, rate=self.config['uniwarp:dropout_rate'],
                                       training=self.is_training, name='CNNActivation')



class CNNSim(CNNAbstractSimModel):
    def __init__(self, config):
        CNNAbstractSimModel.__init__(self, config)
        self.name = 'CNNSim'

    def create_encoder(self):
        CNNAbstractSimModel.create_encoder(self)

        with tf.variable_scope("CNNEncoderDists"):
            self.pair_dists = tf.map_fn(lambda pair_idx: self.dist_pair(pair_idx),
                                        tf.range(self.config['model:num_batch_pairs'], dtype=tf.int32),
                                        back_prop=True,
                                        name='PairWiseDistMap',
                                        dtype=tf.float32)

    def dist_pair(self, pair_ixd):
            return tf.losses.absolute_difference(self.h[2*pair_ixd, :, :], self.h[2*pair_ixd+1, :, :])

    def create_similarity(self):
        with tf.variable_scope("CNNSimilarity"):
            # the rbf similarity
            self.pred_similarities = tf.exp(-self.pair_dists, name='CNNSim')

# the warped version based on CNN features
class CNNWarpedSim(CNNSim):

    def __init__(self, config):
        CNNSim.__init__(self, config)
        self.name = 'CNNWarpedSim'
        self.is_first_dist_pair_call = True

    # redefine the distance between a pair of instances
    def dist_pair(self, pair_ixd):

        with tf.variable_scope("CNNWarpedSimDistPair") as scope:

            # unless it is the first call, then reuse the variables of the scope
            if self.is_first_dist_pair_call:
                self.is_first_dist_pair_call = False
            else:
                scope.reuse_variables()

            # T x K tensors, for T the latent time indices and K the number of RNN cells (encoder length)
            self.A = self.h[2*pair_ixd, :, :]
            self.B = self.h[2*pair_ixd + 1, :, :]
            # the first indices [0,0,0,...,0,1,1,1,...,1,2,2,2,...]
            idx_A = tf.range(self.A.shape[0])
            idx_A = tf.tile(idx_A, [self.A.shape[0]])
            # the second indices [0,1,2,...,B.shape[0],0,1,2,...,B.shape[0],0,1,2,...]
            idx_B = tf.range(self.B.shape[0])
            idx_B = tf.reshape(idx_B, [-1, 1])
            idx_B = tf.tile(idx_B, [1, self.B.shape[0]])
            idx_B = tf.reshape(idx_B, [-1])
            # gather the features for the indices
            self.A_expanded = tf.gather(self.A, idx_A)
            self.B_expanded = tf.gather(self.B, idx_B)

            # concatenate the two feature tensor to serve as the input for the warping weight neural network
            self.AB_concat = tf.concat([self.A_expanded, self.B_expanded], axis=1, name='ConcatenatedPairwiseIndices')
            # define the warping neural network
            self.warp_weights = self.AB_concat
            for num_units in self.config['uniwarp:warp_nn_layers']:
                print('Adding Warping NN layer with ', num_units, 'neurons')
                self.warp_weights = tf.layers.dense(inputs=self.warp_weights, activation=tf.nn.relu, units=num_units)

            # a final linear layer for the warping weights output in [0, 1]
            self.warp_weights = tf.layers.dense(inputs=self.warp_weights, activation=tf.nn.sigmoid, units=1,
                                                name='WarpingWeights')

            # the squared euclidean distance of all pairs
            A_minus_B_square = tf.abs(tf.subtract(self.A_expanded, self.B_expanded))
            self.pairs_dists = tf.expand_dims(tf.reduce_mean(A_minus_B_square, axis=1), axis=-1, name='PairsDists')

            # the warped distances
            self.warped_dists = tf.multiply(self.pairs_dists, self.warp_weights, name="CNNWarpedSim")

            return tf.reduce_mean(self.warped_dists)
