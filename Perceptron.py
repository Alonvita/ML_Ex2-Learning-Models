# Algorithm Developed by Alon Vita
import numpy as np
import TrainingSetManipulations


class Perceptron:
    OUT_PUT_DIMS_SIZE = 5
    __FIRST_LAYER_SIZE = 30
    __SECOND_LAYER_SIZE = 20

    _weight_set = None
    _train_data_frame = None

    def __init__(self, training_data_frame):
        self._train_data_frame = training_data_frame

    def run_algorithm(self):
        self._weight_set = self.__create_weights_set()

        # TODO: turn 'Sex' into a numeric type

    @staticmethod
    def __xavier_init(in_dim, out_dim):
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

    def __create_weights_set(self):
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
        # init dims
        dims = [len(self._train_data_frame),
                self.__FIRST_LAYER_SIZE,
                self.__SECOND_LAYER_SIZE,
                self.OUT_PUT_DIMS_SIZE]

        params = []

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            # append params using xavier_init
            params.append(Perceptron.__xavier_init(in_dim, out_dim))
            params.append(Perceptron.__xavier_init(1, out_dim))

        return params

    def initial_prediction_set(self):
        prediction_set = []
        df_values = self._train_data_frame[TrainingSetManipulations.DATA_SET_X_VALUES].values

        # for each index of the DF values
        for index in range(len(df_values)):
            df_row = df_values[index]
            weight_for_row = self._weight_set[index]

            # append the prediction made for row with it's calculated weight
            prediction_set.append(self.__predict(df_row, weight_for_row))

        return

    @staticmethod
    def __predict(row, weights):
        """
        __predict(row, weights).

        :param row: a DF row, assuming all values are integers!
        :param weights: the weight for the row
        :return: the Y prediction for the row, given the weight
        """
        activation = weights[0]

        for i in range(len(row) - 1):
            activation += weights[i + 1] * row[i]

        return 1.0 if activation >= 0.0 else 0.0

