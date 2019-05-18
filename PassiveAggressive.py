import numpy as np
import ModelInterface


class PassiveAggressive(ModelInterface):
    """
    PassiveAggressive(ModelInterface)
    """
    def __init__(self, n, k):
        self._weight_matrix = np.zeros((k, n))

    def predict(self, row):
        """
        predict(self, row).

        :param row: a row from the data set
        :return: the argmax of the dot matrix of the _weight_matrix with the given row.
        """
        return np.argmax(np.dot(self._weight_matrix, row))

    def update(self, row, row_y_values, learning_rate=None):
        """
        update(self, x, y, learning_rate).
        
        :param row: a row from the data set
        :param row_y_values: the Y value of the row
        :param learning_rate: learning rate is ignored in the PassiveAggressive model.
        """
        # predict
        y_hat = self.predict(row)

        # if prediction does not meet the row_y_values...
        if row_y_values != y_hat:
            # get the w_y and w_y_hat from the weight matrix
            weight_y = self._weight_matrix[row_y_values, :]
            weight_y_hat = self._weight_matrix[y_hat, :]

            # calculate loss function
            loss = self.__calc_loss(weight_y, weight_y_hat, row)

            # calculate tao as provided in ex2.pdf
            tau = loss / (np.linalg.norm(row) ** 2)

            # encourage the row_y_values
            self._weight_matrix[row_y_values, :] += tau * row
            # discourage the y_hat values
            self._weight_matrix[y_hat] -= tau * row

    @staticmethod
    def __calc_loss(w_y, w_y_hat, row):
        """
        __calc_loss(w_y, w_y_hat, row).

        The loss function was defined in the ex2.pdf file provided with this project.

        :param w_y: weight of Y
        :param w_y_hat: weight of Y_hat
        :param row: a row from the data set
        :return: the result of the loss function calculation
        """
        return max(0, 1 - np.dot(w_y, row) + np.dot(w_y_hat, row))
