from abc import abstractmethod, ABC


# Created By Alon Vita
class Model(ABC):
    @abstractmethod
    def predict(self, row):
        raise NotImplementedError

    @abstractmethod
    def update(self, row, row_y_values, learn_rate):
        raise NotImplementedError
