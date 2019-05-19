import numpy as np
import random as rd

sex_to_decimal_dict = {'M': 2, 'F': 4, 'I': 8}


class DataManipulations:
    @staticmethod
    def train_fp_to_numpy(x_training_fp, y_training_fp, testing=False):
        x_samples = np.genfromtxt(x_training_fp, dtype='str', delimiter=',')

        if not testing:
            X = [DataManipulations.__parse_sample(s) for s in x_samples]
        else:
            X = x_samples

        Y = np.genfromtxt(y_training_fp, dtype=np.int)

        return X, Y

    @staticmethod
    def __parse_sample(sample):
        x_matrix = np.zeros(len(sample))

        # parse the sex column to decimals
        x_matrix[0] = sex_to_decimal_dict[sample[0]]

        # parse the rest of the samples
        for i in range(1, len(sample)):
            x_matrix[i] = np.float(sample[i])

        return x_matrix

    @staticmethod
    def read_test_file(test_file_fp):
        samples = np.genfromtxt(test_file_fp, dtype='str', delimiter=',')

        return [DataManipulations.__parse_sample(s) for s in samples]

    @staticmethod
    def shuffle(X, Y):
        data = list()

        for i in range(len(Y)):
            data.append(np.append(X[i], Y[i]))

        rd.shuffle(data)

        new_x = []
        new_y = []

        for i in range(len(data)):
            new_x.append(data[i][0:-1])
            new_y.append(data[i][-1])

        new_x = np.array(X)
        new_y = np.array(Y).astype(int)

        return new_x, new_y

    @staticmethod
    def split_data_for_testing(numpy_x, numpy_y):
        """
        split_data_for_testing(numpy_x, numpy_y).

        :param numpy_x: X data as numpy array
        :param numpy_y: Y values as numpy array
        :return: train_x, train_y, eval_x, eval_y
        """
        data_len = len(numpy_y)

        train_x_len = 0.8 * data_len
        train_x_len = int(train_x_len)

        train_x = numpy_x[0:train_x_len]
        train_y = numpy_y[0:train_x_len]

        eval_x = numpy_x[train_x_len:]
        eval_y = numpy_y[train_x_len:]

        return train_x, train_y, eval_x, eval_y
