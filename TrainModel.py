import copy
import random as rd


class TrainModel:
    @staticmethod
    def train(model, X, Y, epochs, learning_rate, print_results=False):
        """
        train(model, X, Y, epochs, learning_rate, verbose=False).

        :param model: a given ModelInterface
        :param X: X data set
        :param Y: Y values for the X data set
        :param epochs: given epochs list
        :param learning_rate: learning rate for the networks
        :param print_results: true in order to print the results, or false otherwise. False default.

        :return: the best model found
        """
        indices = list(range(len(X)))
        best_model = model  # set as the given model
        best_acc = TrainModel.__measure_accuracy(model, X, Y)  # measure accuracy of the given model

        for ep in range(epochs):
            # shuffle indices
            rd.shuffle(indices)

            # update the model, given the X and Y with the given learning rate
            for i in indices:
                x, y = X[i], Y[i]
                model.update(x, y, learning_rate)

            accuracy = TrainModel.__measure_accuracy(model, X, Y)

            if best_acc < accuracy:
                best_model = copy.deepcopy(model)
                best_acc = accuracy

            if print_results:
                print('Epoch: {0}, Accuracy: {1}%'.format(ep + 1, int(100 * accuracy)))

        return best_model

    @staticmethod
    def __measure_accuracy(model, X, Y):
        """
        measure_accuracy(model, X, Y).

        :param model: a given model
        :param X: X data set
        :param Y: Y values for the given data set

        :return: #hit / (#hit + #miss)
        """
        # Local Variables
        hit = miss = 0.0

        for (x, y) in zip(X, Y):
            # predict
            y_hat = model.predict(x)

            # check if prediction == y
            if y != y_hat:
                miss += 1
            else:
                hit += 1

        return float(hit / (hit + miss))
