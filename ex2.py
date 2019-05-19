import sys
from SVM import SVM
from FitModel import FitModel
from Perceptron import Perceptron
from PassiveAggressive import PassiveAggressive
from DataManipulations import DataManipulations as DM

PERCEPTRON_PREDICTION_OFFSET = 0
SVM_PREDICTION_OFFSET = 1
PA_PREDICTION_OFFSET = 2

EPOCHS = 50
LEARNING_RATE = 0.001
COEF_LAMBDA = 0.17


def main(training_x_fp, training_y_fp, x_values_test_fp):
    """
    main(training_x_fp, training_y_fp, test_x_fp).

    :param training_x_fp: file path to training train_x_data
    :param training_y_fp: file path to training train_y_data to the train_x_data data set
    :param x_values_test_fp: testing train_x_data fp.
    """

    train_x_data, train_y_data = DM.shuffle(DM.train_fp_to_numpy(training_x_fp, training_y_fp))

    output_dim = len(set(train_y_data))
    input_dim = len(train_x_data[0])

    perceptron = Perceptron(input_dim, output_dim)
    svm = SVM(input_dim, output_dim, COEF_LAMBDA)
    passive_aggressive = PassiveAggressive(input_dim, output_dim)

    # instantiate a model trainer with the data
    model_trainer = FitModel(train_x_data, train_y_data)

    ideal_models_array = list()

    ideal_models_array.append(model_trainer.fit_model(perceptron, EPOCHS, LEARNING_RATE))
    ideal_models_array.append(model_trainer.fit_model(svm, EPOCHS, LEARNING_RATE))
    ideal_models_array.append(model_trainer.fit_model(passive_aggressive, EPOCHS, None))

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
    main(sys.argv[1], sys.argv[2], sys.argv[3])
