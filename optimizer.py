import numpy as np
import tensorflow as tf


class Optimizer:

    def __init__(self, config, dataset, sim_model):

        self.config = config
        self.dataset = dataset
        self.num_epochs = self.config['optimizer:num_epochs']
        self.sim_model = sim_model
        # create a saver
        self.saver = tf.train.Saver(max_to_keep=100)


    def optimize(self):

        # save the hyper_parameters before the optimization
        #with open("./saved_models/" + self.sim_model.name + "_hyper_params.json", "w") as hyper_params_file:
        #    json.dump(self.config, hyper_params_file)

        with tf.Session() as sess:

            # initialize all variables
            sess.run(tf.global_variables_initializer())
            loss = 0
            freq=100

            # iterate for a number of epochs
            for epoch_idx in range(self.num_epochs):

                batch_true_similarities = []
                batch_pairs_idxs = []

                for i in range(self.config['model:num_batch_pairs']//2):

                    pos_idxs = self.dataset.draw_pair(True)
                    batch_pairs_idxs.append(pos_idxs[0])
                    batch_pairs_idxs.append(pos_idxs[1])
                    batch_true_similarities.append(1.0)

                    neg_idxs = self.dataset.draw_pair(False)
                    batch_pairs_idxs.append(neg_idxs[0])
                    batch_pairs_idxs.append(neg_idxs[1])
                    batch_true_similarities.append(0.0)

                pair_loss = self.update_model(sess, batch_pairs_idxs, batch_true_similarities)
                loss += pair_loss

                if epoch_idx % freq == 0:

                    if epoch_idx > 0:
                        loss /= freq

                    print('DS', epoch_idx, self.dataset.dataset_name, loss)
                    self.saver.save(sess, "./saved_models/" + self.sim_model.name + "_" + self.dataset.dataset_name
                                    + ".ckpt", global_step=epoch_idx//freq)

                    loss = 0

    # update the model for the pairs of similar and dissimilar series
    def update_model(self, sess, batch_pairs_idxs, batch_true_similarities):

        X_batch = np.take(a=self.dataset.X_train, indices=batch_pairs_idxs, axis=0)
        sim_batch = np.asarray(batch_true_similarities)

        # compute similarity and loss
        pair_loss = sess.run(self.sim_model.loss, feed_dict={
            self.sim_model.X_batch: X_batch,
            self.sim_model.true_similarities: sim_batch,
            self.sim_model.is_training: False})

        # update the deep similarity network
        sess.run(self.sim_model.update_rule,
                 feed_dict={self.sim_model.X_batch: X_batch,
                            self.sim_model.true_similarities: sim_batch,
                            self.sim_model.is_training: True})

        return pair_loss
