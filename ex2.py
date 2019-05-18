import sys
from FitModel import FitModel
from Perceptron import Perceptron
from PassiveAggressive import PassiveAggressive
from DataManipulations import DataManipulations as DM

OUTPUT_SIZE = 50


def main(training_x_fp, training_y_fp, x_values_test_fp):
    """
    main(training_x_fp, training_y_fp, test_x_fp).

    :param training_x_fp: file path to training X
    :param training_y_fp: file path to training Y to the X data set
    :param x_values_test_fp: testing X fp.
    """

    X, Y = DM.train_fp_to_numpy(training_x_fp, training_y_fp)

    output_dim = len(set(Y))
    input_dim = len(X[0])

    perceptron = Perceptron(input_dim, output_dim)
    # svm = SVM(input_dim, output_dim, 0.1) TODO: boredda take this off...
    passive_aggressive = PassiveAggressive(input_dim, output_dim)

    best_perceptron = FitModel.fit_model(perceptron, X, Y, OUTPUT_SIZE, 0.001)
    # best_svm = FitModel.train(svm, X, Y, 50, 0.001) TODO: this too
    best_pa = FitModel.fit_model(passive_aggressive, X, Y, OUTPUT_SIZE, None)

    manipulated_test_X_values = DM.read_test_file(x_values_test_fp)

    for test_value in manipulated_test_X_values:
        perceptron_prediction = best_perceptron.predict(test_value)
        # svm_prediction = best_svm.predict(test_value) TODO: this too :)
        passive_aggressive_prediction = best_pa.predict(test_value)

        print('perceptron: {0},  pa: {1}'.format(perceptron_prediction, passive_aggressive_prediction))

        # TODO: then remove the comment from this part
        # print('perceptron: {0}, svm: {1}, pa: {2}'.format(
        #     perceptron_prediction,
        #     svm_prediction,
        #     passive_aggressive_predict))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
