# Algorithm Developed by Alon Vita
import numpy as np

class Perceptron:

    def __init__(self, training_data_frame):
        self._train_data_frame = training_data_frame

    def xavier_init(self, in_dim, out_dim):
        """
        :param in_dim: self explanatory
        :param out_dim: self explanatory

        :return: matrix in shape of (in_dim, out_dim) with good values
        """
        if in_dim == 1 or out_dim == 1:
            if in_dim == 1:
                shape = (out_dim,)
            else:
                shape = (in_dim,)
        else:
            shape = (in_dim, out_dim)

        dim_sum = in_dim + out_dim

        return np.random.uniform(-np.sqrt(6.0 / dim_sum),
                                 np.sqrt(6.0 / dim_sum),
                                 shape)

    def create_classifier(self, dims):
        """
        returns the parameters for a multi-layer perceptron with an arbitrary number
        of hidden layers.

        dims is a list of length at least 2, where the first item is the input
        dimension, the last item is the output dimension, and the ones in between
        are the hidden layers.

        For example, for:
            dims = [300, 20, 30, 40, 5]

        We will have input of 300 dimension, a hidden layer of 20 dimension, passed
        to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
        an output of 5 dimensions.

        return:
        a list of parameters where the first two elements are the W and b from input
        to first layer, then the second two are the matrix and vector from first to
        second layer, and so on.
        """
        params = []

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            params.append(self.xavier_init(in_dim, out_dim))
            params.append(self.xavier_init(1, out_dim))

        return params

    # Make a prediction with weights
    def predict(self, row_index, weights):
        activation = weights[0]

        for i in range(len(row_index) - 1):
            activation += weights[i + 1] * row_index[i]

        return 1.0 if activation >= 0.0 else 0.0

