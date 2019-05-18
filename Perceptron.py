# Algorithm Developed by Alon Vita
import numpy as np
import ModelInterface


class Perceptron(ModelInterface):
    """
    Perceptron(ModelInterface).
    """
    _weight_matrix = None

    def __init__(self, input_dim, output_dim):
        # create the dimensions matrix
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


        The update function will predict the y_hat value. Should the y_hat != Y,
        update the weight matrix according to the given LR.

        :param row: a row from the data set
        :param row_y_value:  row's Y value
        :param learning_rate: the learning rate.
        """
        y_hat = self.predict(row)

        if row_y_value != y_hat:
            # update the right weight in the _weight_matrix:
            # encourage y_value
            self._weight_matrix[row_y_value, :] += learning_rate * row

            # discourage y_hat values
            self._weight_matrix[y_hat, :] -= learning_rate * row
