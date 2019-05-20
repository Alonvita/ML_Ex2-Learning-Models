import sys
import time
from SVM import SVM
from FitModel import FitModel
from Perceptron import Perceptron
from PassiveAggressive import PassiveAggressive
from DataManipulations import DataManipulations as DM

PERCEPTRON_PREDICTION_OFFSET = 0
SVM_PREDICTION_OFFSET = 1
PA_PREDICTION_OFFSET = 2

PERCEPTRON_EPOCHS = 15
SVM_EPOCHS = 12
PA_EPOCHS = 10

LEARNING_RATE = 0.001
COEF_LAMBDA = 0.12


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

    for test_value in test_x_values:
        predictions = list()

        for ideal_model in ideal_models_array:
            predictions.append(ideal_model.predict(test_value))

        print('perceptron: {0}, svm: {1}, pa: {2}'.format(
            predictions[PERCEPTRON_PREDICTION_OFFSET],
            predictions[SVM_PREDICTION_OFFSET],
            predictions[PA_PREDICTION_OFFSET]))


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

        main("testing_run_train_x.txt", "testing_run_train_y.txt", "testing_run.txt")

        print(time.time())

    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
