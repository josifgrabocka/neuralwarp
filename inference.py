import tensorflow as tf
import numpy as np
from hyper_parameters import HyperParams
from dataset import Dataset
import rnn_models
import cnn_models
import sys
import os
import sklearn.metrics

class Inference_Experiments:

    def __init__(self, model_type, model_file, dataset_path):

        self.model_type = model_type
        self.model_file = model_file
        self.dataset_path = dataset_path

        # load the configuration of the hyperparameters
        hp = HyperParams()
        self.config = hp.get_uniwarp_config(None)

        # load the dataset
        self.ds = Dataset()
        self.ds.load_multivariate(dataset_path)

        # set the length and number of channels
        self.config['uniwarp:length'] = self.ds.series_length
        self.config['dataset:num_channels'] = self.ds.num_channels

        # create the model
        self.model = None

        if model_type == 'SiameseRNN':
            self.model = rnn_models.SiameseRNN(config=self.config)
        elif model_type == 'WarpedSiameseRNN':
            self.model = rnn_models.WarpedSiameseRNN(config=self.config)
        elif model_type == 'CNNSim':
            self.model = cnn_models.CNNSim(config=self.config)
        elif model_type == 'CNNWarpedSim':
            self.model = cnn_models.CNNWarpedSim(config=self.config)
        else:
            print("Test - No model of type", model_type)

        self.model.create_model()

        # a tensorflow saver to be used for loading models
        self.saver = tf.train.Saver()

        # a batch tensor
        self.X_batch = np.zeros((2*self.config['model:num_batch_pairs'], self.config['uniwarp:length'],
                                       self.config['dataset:num_channels']))
        # a batch similarity
        self.true_sim_batch = np.zeros((self.config['model:num_batch_pairs'],))

        print('Model has', self.model.num_model_parameters(), 'parameters')

    # infer the target of the test instances of a dataset
    # starting from the {start_pct} percentage of the instances for {chunk_pct} many instances
    # e.g. start_pct=0.1, chunk_pct=0.2 means classifying the segment between [10%, 30%)
    def infer_dataset(self, start_pct, chunk_pct):

        start_range = int(start_pct * self.ds.num_test_instances)
        stop_range = int((start_pct + chunk_pct) * self.ds.num_test_instances)
        if stop_range > self.ds.num_test_instances:
            stop_range = self.ds.num_test_instances

        with tf.Session() as sess:

            # restore the variables for the computational graph created in the constructor
            self.saver.restore(sess, self.model_file)

            correct, num_infers = 0, 0

            time = -1

            for idx_test in range(start_range, stop_range):

                max_similarity = 0
                max_similarity_idx = 0

                for idx in range(0, self.ds.num_train_instances, self.config['model:num_batch_pairs']):

                    # fix the starting index, if the batch exceeds the number of train instances
                    start_idx = idx
                    if idx + self.config['model:num_batch_pairs'] >= self.ds.num_train_instances:
                        start_idx = self.ds.num_train_instances - self.config['model:num_batch_pairs']

                    # create a batch of pair between the test series and the batch train series
                    for i in range(self.config['model:num_batch_pairs']):
                        self.X_batch[2*i] = self.ds.X_test[idx_test]
                        self.X_batch[2*i+1] = self.ds.X_train[start_idx+i]

                    # measure the similarity between the test series and the training batch series
                    sim = sess.run(self.model.pred_similarities,
                                   feed_dict={self.model.X_batch: self.X_batch,
                                              self.model.is_training: False})

                    # check similarities of all pairs and record the closest training series S
                    for i in range(self.config['model:num_batch_pairs']):
                        if sim[i] >= max_similarity:
                            max_similarity = sim[i]
                            max_similarity_idx = start_idx+i

                # check if correctly classified
                if np.array_equal(self.ds.Y_test[idx_test], self.ds.Y_train[max_similarity_idx]):
                    correct += 1
                num_infers += 1

                print(idx_test, correct / num_infers)

            print(num_infers, correct, time, dataset_path)

    # the pairwise similarities of the test series
    def test_pairwise_similarities(self, n, folder_path):

        num_test_series = n
        dists = np.zeros((num_test_series,num_test_series))

        with tf.Session() as sess:

            self.saver.restore(sess, self.model_file)

            # all pairs of the first num_test_series
            pairs_list = []
            for i in np.arange(0, num_test_series, 1):
                for j in np.arange(0, num_test_series, 1):
                    pairs_list.append((i, j))

            num_pairs = len(pairs_list)
            batch_start_pair_idx = 0

            print('Num pairs:', len(pairs_list))

            # compute pair similarities in batches
            while batch_start_pair_idx < num_pairs:

                # create a batch of pair between the test series and the batch train series
                for i in range(self.config['model:num_batch_pairs']):

                    # the index of the pair
                    j = batch_start_pair_idx + i
                    if j >= num_pairs:
                        j = num_pairs - 1

                    self.X_batch[2 * i] = self.ds.X_test[pairs_list[j][0]]
                    self.X_batch[2 * i + 1] = self.ds.X_test[pairs_list[j][1]]

                #print('batch starting at', batch_start_pair_idx)

                # measure the similarity between the test series and the training batch series
                sim = sess.run(self.model.pred_similarities,
                                   feed_dict={self.model.X_batch: self.X_batch,
                                              self.model.is_training: False})
                # set the distances
                for i in range(self.config['model:num_batch_pairs']):
                    # the index of the pair
                    j = batch_start_pair_idx + i
                    if j >= num_pairs:
                        j = num_pairs - 1
                    # set the distance
                    dists[pairs_list[j][0]][pairs_list[j][1]] = 1.0 - sim[i]

                # the batch pair index increases
                batch_start_pair_idx += self.config['model:num_batch_pairs']

            # the distances
            print(dists.shape)

            np.save(os.path.join(folder_path, self.model.name + '_' + self.ds.dataset_name + "_dists.npy"), dists)
            np.save(os.path.join(folder_path, self.model.name + '_' + self.ds.dataset_name + "_labels.npy"), self.ds.Y_test[:num_test_series])

    # the pairwise similarities of the test series
    def pairwise_test_accuracy(self, num_test_batches):

        test_acc = 0

        with tf.Session() as sess:

            self.saver.restore(sess, self.model_file)

            for i in range(num_test_batches):

                # draw the random test batch
                batch_pairs_idxs = []
                batch_true_similarities = []
                for j in range(self.config['model:num_batch_pairs'] // 2):
                    pos_idxs = self.ds.draw_test_pair(True)
                    batch_pairs_idxs.append(pos_idxs[0])
                    batch_pairs_idxs.append(pos_idxs[1])
                    batch_true_similarities.append(1.0)

                    neg_idxs = self.ds.draw_test_pair(False)
                    batch_pairs_idxs.append(neg_idxs[0])
                    batch_pairs_idxs.append(neg_idxs[1])
                    batch_true_similarities.append(0.0)
                # the numpy tensors of the series and ground truth similarities
                X_batch = np.take(a=self.ds.X_test, indices=batch_pairs_idxs, axis=0)
                sim_batch = np.asarray(batch_true_similarities)

                # measure the batch loss of the model
                pred_similarities = sess.run(self.model.pred_similarities, feed_dict={
                            self.model.X_batch: X_batch,
                            self.model.true_similarities: sim_batch,
                            self.model.is_training: False})

                pred_label = np.where(pred_similarities >= 0.5, 1, 0)

                test_acc += sklearn.metrics.accuracy_score(sim_batch, pred_label)

                # print progress
                print(i, test_acc / (i+1))

        # print test batches
        print(test_acc / num_test_batches)


    # the pairwise similarities of the test series
    def transductive_test_loss(self):

        test_loss = 0

        with tf.Session() as sess:

            self.saver.restore(sess, self.model_file)

            for i in range(num_test_batches):

                # draw the random test batch
                batch_pairs_idxs = []
                batch_true_similarities = []
                for j in range(self.config['model:num_batch_pairs'] // 2):
                    pos_idxs = self.ds.draw_test_pair(True)
                    batch_pairs_idxs.append(pos_idxs[0])
                    batch_pairs_idxs.append(pos_idxs[1])
                    batch_true_similarities.append(1.0)

                    neg_idxs = self.ds.draw_test_pair(False)
                    batch_pairs_idxs.append(neg_idxs[0])
                    batch_pairs_idxs.append(neg_idxs[1])
                    batch_true_similarities.append(0.0)
                # the numpy tensors of the series and ground truth similarities
                X_batch = np.take(a=self.ds.X_test, indices=batch_pairs_idxs, axis=0)
                sim_batch = np.asarray(batch_true_similarities)

                # measure the batch loss of the model
                batch_loss = sess.run(self.model.loss, feed_dict={
                            self.model.X_batch: X_batch,
                            self.model.true_similarities: sim_batch,
                            self.model.is_training: False})

                test_loss += batch_loss

                # print progress
                print(i, test_loss / (i+1))

        # print test batches
        print(test_loss / num_test_batches)



# the main file for the experiments
choice = sys.argv[1]
model_type = sys.argv[2]
model_file = sys.argv[3]
dataset_path = sys.argv[4]


#folder_path = sys.argv[4]

if choice == 'test':
    start_pct = float(sys.argv[5])
    chunk_pct = float(sys.argv[6])
elif choice == 'pairwise':
    num_test_batches = int(sys.argv[5])


#model_type = "SiameseRNN"
#model_file = "/home/josif/ownCloud/research/parametricwarp/saved_models/SiameseRNN_satellite_0.ckpt-99"
#config_file = "/home/josif/ownCloud/research/uniwarp/warp/saved_models/params.json"
#dataset_path = "/home/josif/ownCloud/research/tsc/baselines/satellite/0/"

# load the data
ie = Inference_Experiments(model_type=model_type,
                           model_file=model_file,
                           dataset_path=dataset_path)

if choice == 'test':
    ie.infer_dataset(start_pct, chunk_pct)
elif choice == 'pairwise':
    ie.pairwise_test_accuracy(num_test_batches)


