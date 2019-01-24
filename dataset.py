import numpy as np
import os
from sklearn import preprocessing


class Dataset:

    def __init__(self):
        self.Y_train = None
        self.X_train = None
        self.Y_train_raw = None
        self.Y_test_raw = None
        self.Y_test = None
        self.X_test = None
        self.dataset_name = None
        self.num_classes = None
        self.series_length = None
        self.num_channels = None
        self.num_train_instances = None
        self.dataset_name = None

    def load_multivariate(self, dataset_prefix):

        X_train = np.load(dataset_prefix+"train_features.npy")
        Y_train = np.load(dataset_prefix+"train_labels.npy")

        X_test = np.load(dataset_prefix+"test_features.npy")
        Y_test = np.load(dataset_prefix+"test_labels.npy")

        Y_train = np.expand_dims(Y_train, axis=-1)
        Y_test = np.expand_dims(Y_test, axis=-1)

        self.num_train_instances = X_train.shape[0]
        self.num_test_instances = X_test.shape[0]
        self.series_length = X_train.shape[1]
        self.num_channels = X_train.shape[2]

        # the num of instances
        self.num_instances = self.num_train_instances + self.num_test_instances

        sorted_label_values = np.unique(Y_train)
        self.num_classes = sorted_label_values.size

        self.Y_train_multiclass = Y_train

        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        onehot_encoder = onehot_encoder.fit(np.concatenate((Y_train, Y_test), axis=0))

        Y_train_ohe = onehot_encoder.transform(Y_train)
        Y_test_ohe = onehot_encoder.transform(Y_test)

        self.X_train, self.Y_train = X_train, Y_train_ohe
        self.X_test, self.Y_test = X_test, Y_test_ohe

        path = os.path.normpath(dataset_prefix)
        ds_path, fold = os.path.split(path)
        root, ds_name = os.path.split(ds_path)
        self.dataset_name = ds_name + "_" + fold


        print('Train shape', self.X_train.shape)
        print('Test shape', self.X_test.shape)

        #print(self.dataset_name, 'num_train_instances', self.num_train_instances, 'series_length', self.series_length,
        #      'num_channels', self.num_channels, 'num_classes', self.num_classes)

        #print('Test set dims', self.X_test.shape)

    def load_ucr_univariate_data(self, dataset_folder=None):

        # read the dataset name as the folder name
        self.dataset_name = os.path.basename(os.path.normpath(dataset_folder))

        # load the train and test data from files
        file_prefix = os.path.join(dataset_folder, self.dataset_name)
        train_data = np.loadtxt(file_prefix + "_TRAIN", delimiter=",")
        test_data = np.loadtxt(file_prefix + "_TEST", delimiter=",")

        # set train data
        self.Y_train = train_data[:, 0]
        self.X_train = train_data[:, 1:]
        self.num_train_instances = self.X_train.shape[0]

        # get the series length
        self.series_length = self.X_train.shape[1]

        # set the test data
        self.Y_test = test_data[:, 0]
        self.X_test = test_data[:, 1:]
        self.num_test_instances = self.X_test.shape[0]

        self.Y_train_raw = train_data[:, 0]
        self.Y_test_raw = test_data[:, 0]

        # the num of instances
        self.num_instances = self.num_train_instances + self.num_test_instances

        # get the label values in a sorted way
        sorted_label_values = np.unique(self.Y_train)
        self.num_classes = sorted_label_values.size

        #print('Series length', self.series_length, ', Num classes', self.num_classes)

        # encode labels to a range between [0, num_classes)
        label_encoder = preprocessing.LabelEncoder()
        label_encoder = label_encoder.fit(self.Y_train)
        Y_train_encoded = label_encoder.transform(self.Y_train)
        Y_test_encoded = label_encoder.transform(self.Y_test)

        # convert the encoded labels to a 2D array of shape (num_instances, 1)
        Y_train_encoded = Y_train_encoded.reshape(len(Y_train_encoded), 1)
        Y_test_encoded = Y_test_encoded.reshape(len(Y_test_encoded), 1)

        # one-hot encode the labels
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        onehot_encoder = onehot_encoder.fit(Y_train_encoded)
        self.Y_train = onehot_encoder.transform(Y_train_encoded)
        self.Y_test = onehot_encoder.transform(Y_test_encoded)

        # normalize the time series
        X_train_norm = preprocessing.normalize(self.X_train, axis=1)
        X_test_norm = preprocessing.normalize(self.X_test, axis=1)

        self.X_train = np.expand_dims(X_train_norm, axis=-1)
        self.X_test = np.expand_dims(X_test_norm, axis=-1)

        self.num_channels = 1

        print('Train shape', self.X_train.shape)
        print('Test shape', self.X_test.shape)

    # draw a random set of instances from the training set
    def draw_batch(self, batch_size):
        # draw an arraw of random numbers from 0 to num rows in X_train
        random_row_indices = np.random.randint(0, self.num_train_instances, size=batch_size)
        X_batch = self.X_train[random_row_indices]
        Y_batch = self.Y_train[random_row_indices]
        # slice the batch from the training set acc. to. the drawn row indices
        return X_batch, Y_batch

    # draw a random set of instances from the training set
    def draw_similar_batch(self, batch_size):

        # draw a random class and then draw randomly

        return None


    def draw_dissimilar_batch(self, batch_size):
        # draw an arraw of random numbers from 0 to num rows in X_train
        random_row_indices = np.random.randint(0, self.num_train_instances, size=batch_size)
        X_batch = self.X_train[random_row_indices]
        Y_batch = self.Y_train[random_row_indices]
        # slice the batch from the training set acc. to. the drawn row indices
        return X_batch, Y_batch


    # draw a random set of instances from the training set
    def draw_pair(self, is_positive):

        while True:

            first_idx = np.random.randint(0, self.num_train_instances, size=1)[0]
            second_idx = np.random.randint(0, self.num_train_instances, size=1)[0]

            if is_positive:
                if np.array_equal(self.Y_train[first_idx], self.Y_train[second_idx]):
                    return first_idx, second_idx
            else:
                if not np.array_equal(self.Y_train[first_idx], self.Y_train[second_idx]):
                    return first_idx, second_idx

    # draw a random set of instances from the training set
    def draw_test_pair(self, is_positive):

        while True:

            first_idx = np.random.randint(0, self.num_test_instances, size=1)[0]
            second_idx = np.random.randint(0, self.num_test_instances, size=1)[0]

            if is_positive:
                if np.array_equal(self.Y_test[first_idx], self.Y_test[second_idx]):
                    return first_idx, second_idx
            else:
                if not np.array_equal(self.Y_test[first_idx], self.Y_test[second_idx]):
                    return first_idx, second_idx

    # retreive the series of the pair idxs
    def retrieve_series_content(self, pair_idxs, max_length):

        # create a pair tensor filled with zeros up to max length
        X = np.zeros(shape=(2, max_length, 1))

        # set the series content from the current dataset
        series_length = self.series_length
        X[0][:series_length] = np.expand_dims(self.X_train[pair_idxs[0]], axis=-1)
        X[1][:series_length] = np.expand_dims(self.X_train[pair_idxs[1]], axis=-1)

        return X

