# Algorithm Developed by Or Borreda
import numpy as np
from ModelInterface import Model


class SVM(Model):
    """
    SVM(ModelInterface).
    """
    _weight_matrix = None

    def __init__(self, input_dim, output_dim, coef_lambda):
        # create the dimensions matrix
        self._lambda = coef_lambda
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._weight_matrix = np.zeros((output_dim, input_dim))

    def predict(self, row):
        """
        predict(self, row).

        :param row: a row from the data set

        :return: the argmax of the dot matrix of the _weight_matrix with the given row.
        """
        return np.argmax(np.dot(self._weight_matrix, row))

    def update(self, row, row_y_value, learning_rate):
        """
        update(self, row, row_y_value, learning_rate).

        The assumption is that the dimension of the input row will be LARGER than the
        amount of possible y_values.

        :param row: a row from the data set
        :param row_y_value:  row's Y value
        :param learning_rate: the learning rate.
        """
        # predict
        y_hat = self.predict(row)

        # calc coefficient
        coefficient = self.__calc_coef(learning_rate)

        # for each dim from the output_dim
        for dim in range(self._output_dim):
            # take all the dimensions for a given Weight.
            weight_dim_vec = self._weight_matrix[dim, :] * coefficient

            # encourage the weight dimension if it is equal to the y_value
            if dim == row_y_value:
                weight_dim_vec += (1 - learning_rate * coefficient) * row

            # discourage the weight dimension if it is equal to the prediction
            elif dim == y_hat:
                weight_dim_vec -= (1 - learning_rate * coefficient) * row

            # update matrix
            self._weight_matrix[dim, :] = weight_dim_vec

    def __calc_coef(self, learning_rate):
        return 1 - (learning_rate * self._lambda)
