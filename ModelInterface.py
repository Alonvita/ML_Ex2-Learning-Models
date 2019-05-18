# Created By Alon Vita
class Model(object):
    def predict(self, row):
        raise NotImplementedError

    def update(self, row, row_y_values, learn_rate):
        raise NotImplementedError
