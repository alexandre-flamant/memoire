import torch


class MinMaxScaler():
    def __init__(self, n_features):
        super(MinMaxScaler, self).__init__()
        self.__n_features = n_features
        self.__max = None
        self.__min = None

    def fit(self, x):
        x = x.reshape((-1, self.__n_features))
        self.__max = x.max(dim=0).values
        self.__min = x.min(dim=0).values

    def partial_fit(self, x):
        x = x.reshape((-1, self.__n_features))
        if self.__max is None:
            self.__max = x.max(dim=0).values
            self.__min = x.min(dim=0).values
        else:
            self.__max = torch.cat((self.__max.reshape((-1, self.__n_features)), x), dim=0).max(dim=0).values
            self.__min = torch.cat((self.__min.reshape((-1, self.__n_features)), x), dim=0).min(dim=0).values

    def transform(self, x):
        if self.__max is None:
            raise Exception("Scaler has not been fit yet.")
        return (x - self.__min) / (self.__max - self.__min)

    def inverse_transform(self, x):
        if self.__max is None:
            raise Exception("Scaler has not been fit yet.")
        return x * (self.__max - self.__min) + self.__min

    def __str__(self):
        return f"MinMaxScaler(n_features={self.__n_features})\nMax: {self.__min}\nMin: {self.__max}"

    def __repr__(self):
        return self.__str__()


__all__ = ['MinMaxScaler']
