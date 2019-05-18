# Created By Alon Vita
class Model(object):
    def predict(self, x):
        raise NotImplementedError

    def update(self, x, y, learn_rate):
        raise NotImplementedError
