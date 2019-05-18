# Algorithm Developed by Or Borreda
import numpy as np
from ModelInterface import Model


class SVM(Model):
    """
    SVM(ModelInterface).
    """
    _weight_matrix = None

    def __init__(self, input_dim, output_dim, l):
        # create the dimensions matrix
        self._weight_matrix = np.zeros((output_dim, input_dim))
        self._lambda = l

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


        The update function will predict value.
        update the weight matrix according to the given LR.

        :param row: a row from the data set
        :param row_y_value:  row's Y value
        :param learning_rate: the learning rate.
        """
        y_hat = self.predict(row)
        self._weight_matrix

