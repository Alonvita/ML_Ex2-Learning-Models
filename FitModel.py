import copy
import random as rd


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
