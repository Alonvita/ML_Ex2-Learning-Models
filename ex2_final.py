import sys
import time
from SVM import SVM
from FitModel import FitModel
from Perceptron import Perceptron
from PassiveAggressive import PassiveAggressive
from DataManipulations import DataManipulations as DM
import numpy as np
from abc import ABC, abstractmethod
import copy
import random as rd


PERCEPTRON_PREDICTION_OFFSET = 0
SVM_PREDICTION_OFFSET = 1
PA_PREDICTION_OFFSET = 2

PERCEPTRON_EPOCHS = 15
SVM_EPOCHS = 12
PA_EPOCHS = 10

LEARNING_RATE = 0.001
COEF_LAMBDA = 0.12

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


class FitModel:
    def __init__(self, train_x_data, train_y_data):
        self._X = train_x_data
        self._Y = train_y_data

    def fit_model(self, model, epochs, learning_rate, print_results=False):
        """
        train(model, train_x_data, train_y_values, epochs, learning_rate, verbose=False).

        :param model: a given ModelInterface
        :param epochs: given epochs list
        :param learning_rate: learning rate for the networks
        :param print_results: true in order to print the results, or false otherwise. False default.

        :return: the best model found
        """
        if print_results:
            print("--- Start: Printing model results ---")

        indices = list(range(len(self._X)))
        best_model = model  # init as the given model
        best_acc_found = self.__measure_accuracy(model)  # measure accuracy of the given model

        for ep in range(epochs):
            # shuffle indices
            rd.shuffle(indices)

            # update the model, given the X and Y with the given learning rate
            for i in indices:
                x = self._X[i]
                y = self._Y[i]

                model.update(x, y, learning_rate)

            accuracy_percent = self.__measure_accuracy(model) * 100

            if best_acc_found < accuracy_percent:
                best_model = copy.deepcopy(model)
                best_acc_found = accuracy_percent

            if print_results:
                print('Epoch: {0}, Accuracy: {1}%'.format(ep + 1, int(accuracy_percent)))

        if print_results:
            print('Best accuracy found is: {0}%:'.format(int(best_acc_found)))

        if print_results:
            print("--- End: Printing model results ---")

        return best_model

    def __measure_accuracy(self, model):
        """
        measure_accuracy(model).

        :param model: a given model

        :return: #hit / (#hit + #miss)
        """
        # Local Variables
        hit = miss = 0

        for (x, y) in zip(self._X, self._Y):
            # predict
            y_hat = model.predict(x)

            # check if prediction == y
            if y != y_hat:
                miss += 1
            else:
                hit += 1

        return float(hit / (hit + miss))


class Model(ABC):
    @abstractmethod
    def predict(self, row):
        raise NotImplementedError

    @abstractmethod
    def update(self, row, row_y_values, learn_rate):
        raise NotImplementedError


class Perceptron(Model):
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


class PassiveAggressive(Model):
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
            tau = 2 * (np.linalg.norm(row) ** 2)

            # --- calculation overflow reached ---
            # this split is essential as calculating in a single row as caused an overflow...
            tau = loss / tau

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


def main(training_x_fp, training_y_fp, x_values_test_fp, testing_run=False):
    """
    main(training_x_fp, training_y_fp, test_x_fp).

    :param training_x_fp: file path to training train_x_data
    :param training_y_fp: file path to training train_y_data to the train_x_data data set
    :param x_values_test_fp: testing train_x_data fp.
    """
    train_x_data, train_y_data = DM.train_fp_to_numpy(training_x_fp, training_y_fp)

    output_dim = len(set(train_y_data))
    input_dim = len(train_x_data[0])

    perceptron = Perceptron(input_dim, output_dim)
    svm = SVM(input_dim, output_dim, COEF_LAMBDA)
    passive_aggressive = PassiveAggressive(input_dim, output_dim)

    # instantiate a model trainer with the data
    model_trainer = FitModel(train_x_data, train_y_data)

    ideal_models_array = list()

    ideal_models_array.append(model_trainer.fit_model(perceptron, PERCEPTRON_EPOCHS, LEARNING_RATE, print_results=True))
    ideal_models_array.append(model_trainer.fit_model(svm, SVM_EPOCHS, LEARNING_RATE, print_results=True))
    ideal_models_array.append(model_trainer.fit_model(passive_aggressive, PA_EPOCHS, None, print_results=True))

    test_x_values = DM.read_test_file(x_values_test_fp)
    test_y_data = []
    i = 0
    if testing_run:
        test_y_data = np.genfromtxt('testing_run_train_y.txt', dtype=np.int)
    test_y_val = [0, 0, 0]
    for test_value in test_x_values:
        predictions = list()

        for ideal_model in ideal_models_array:
            predictions.append(ideal_model.predict(test_value))

        print('perceptron: {0}, svm: {1}, pa: {2}'.format(
            predictions[PERCEPTRON_PREDICTION_OFFSET],
            predictions[SVM_PREDICTION_OFFSET],
            predictions[PA_PREDICTION_OFFSET]))
        if testing_run:
            i += 1
            if predictions[PERCEPTRON_PREDICTION_OFFSET] == test_y_data[i]:
                test_y_val[PERCEPTRON_PREDICTION_OFFSET] += 1
            if predictions[SVM_PREDICTION_OFFSET] == test_y_data[i]:
                test_y_val[SVM_PREDICTION_OFFSET] += 1
            if predictions[PA_PREDICTION_OFFSET] == test_y_data[i]:
                test_y_val[PA_PREDICTION_OFFSET] += 1
    if testing_run:
        print('validation: perceptron: {0}, svm: {1}, pa: {2}'.format(
            test_y_val[PERCEPTRON_PREDICTION_OFFSET]/len(test_y_data),
            test_y_val[SVM_PREDICTION_OFFSET]/len(test_y_data),
            test_y_val[PA_PREDICTION_OFFSET]/len(test_y_data)))


if __name__ == "__main__":
    if sys.argv[3] == "testing_run":
        numpy_x, numpy_y = DM.train_fp_to_numpy(sys.argv[1], sys.argv[2], testing=True)
        train_x, train_y, eval_x, eval_y = DM.split_data_for_testing(numpy_x, numpy_y)

        # write everything to files
        # write train_x
        with open('testing_run_train_x.txt', 'w') as f:
            # Achievement Unlocked: STRING MASTER MANIPULATOR
            f.write('\n'.join(str(','.join(str(item) for item in row)).strip('[]') for row in train_x))

        with open('testing_run.txt', 'w') as f:
            f.write('\n'.join(str(','.join(str(item) for item in row)).strip('[]') for row in eval_x))

        # write train_y to file
        with open('testing_run_train_y.txt', 'w') as f:
            f.write('\n'.join(str(row) for row in train_y))

        print(time.time())

        main("testing_run_train_x.txt", "testing_run_train_y.txt", "testing_run.txt", testing_run=True)

        print(time.time())

    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
