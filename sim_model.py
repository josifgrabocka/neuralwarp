import tensorflow as tf

# an abstract similarity model
class AbstractSimModel:

    def __init__(self, config):
        # the configuration
        self.config = config
        # define a minus one constant
        self.minus_one_constant = tf.constant(-1.0, dtype=tf.float32)
        # the maximum sequence length
        self.sequence_length = self.config['uniwarp:length']
        # define the placeholder 
        self.X_batch = tf.placeholder(shape=(2 * self.config['model:num_batch_pairs'], self.config['uniwarp:length'],
                                             self.config['dataset:num_channels']), dtype=tf.float32)

        self.true_similarities = tf.placeholder(shape=(self.config['model:num_batch_pairs'],), dtype=tf.float32)
        # the distance between the pairs
        self.pair_dists = None

        # define the activations, similarity and update rule
        self.h = None, None
        self.loss, self.pred_similarities, self.update_rule = None, None, None
        # the gamma constant
        self.reg_penalty = tf.constant(self.config['uniwarp:lambda'], dtype=tf.float32)
        self.name = 'AbstractSingleSimModel'
        self.is_training = tf.placeholder(tf.bool)
        self.additional_loss = None

    # count the number of parameters in the model
    def num_model_parameters(self):

        total_parameters = 0

        for variable in tf.trainable_variables():
            shape = variable.get_shape()

            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value

            total_parameters += variable_parameters

        return total_parameters

    # will be overriden by the child classes
    def create_encoder(self):
        print("ERROR: Encoder left undefined")
        pass

    # will be overriden by the child classes
    def create_similarity(self):
        print("ERROR: Similarity left undefined")
        pass

        # will be overriden by the child classes
    def dist_pair(self, pair_ixd):
        print("ERROR: Distance of pairs left undefined")
        pass

    # the update rule
    def create_optimization_routine(self):
        # create an update rule using the Adam optimizer for
        # maximizing similarity if similarity_sign == 1 and
        # minimizing similarity if similarity_sign == -1

        with tf.variable_scope("OptimizationRoutines"):

            self.loss = tf.losses.log_loss(self.true_similarities, self.pred_similarities)

            # add additional loss terms, e.g. regularization
            if self.additional_loss is not None:
                print("Adding penalty term", self.additional_loss)
                self.loss += self.reg_penalty*self.additional_loss

            # get all the trainable variables
            trainable_vars = tf.trainable_variables()
            # apply the gradients using clipping to avoid their explosion
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars),
                                                          self.config['uniwarp:max_grad_norm'])

                self.update_rule = tf.train.AdamOptimizer(self.config['uniwarp:eta']).\
                    apply_gradients(zip(clipped_grads, trainable_vars))

    # create the model
    def create_model(self):

        # create the encoder
        self.create_encoder()

        # create the similarity
        self.create_similarity()

        # create the optimization nodes
        self.create_optimization_routine()


