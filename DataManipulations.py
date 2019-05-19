import numpy as np
import random as rd

sex_to_decimal_dict = {'M': 2, 'F': 4, 'I': 8}


class DataManipulations:
    @staticmethod
    def train_fp_to_numpy(x_training_fp, y_training_fp):
        x_samples = np.genfromtxt(x_training_fp, dtype='str', delimiter=',')
        X = [DataManipulations.__parse_sample(s) for s in x_samples]

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
