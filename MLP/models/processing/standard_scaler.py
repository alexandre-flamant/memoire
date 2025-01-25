from copy import deepcopy

import torch


class StandardScaler():
    def __init__(self, n_features):
        super(StandardScaler, self).__init__()
        self.__n_features = n_features
        self.__sum = torch.zeros(self.__n_features)
        self.__sum2 = torch.zeros(self.__n_features)
        self.__mean = torch.zeros(self.__n_features)
        self.__std = torch.zeros(self.__n_features)
        self.__n = 0

    def to(self, device):
        self_copy = deepcopy(self)
        self_copy.__sum = self.__sum.to(device)
        self_copy.__sum2 = self.__sum2.to(device)
        self_copy.__mean = self.__mean.to(device)
        self_copy.__std = self.__std.to(device)

        return self_copy

    def fit(self, x: torch.Tensor):
        self.__sum = torch.zeros(self.__n_features)
        self.__sum2 = torch.zeros(self.__n_features)
        self.__mean = torch.zeros(self.__n_features)
        self.__std = torch.zeros(self.__n_features)

        x = x.reshape(-1, self.__n_features)
        self.__n = x.shape[0]

        self.__update(x)

    def partial_fit(self, x):
        x = x.reshape(-1, self.__n_features)
        self.__n += x.shape[0]

        self.__update(x)

    def __update(self, x):
        self.__sum += torch.sum(x, dim=0)
        self.__sum2 += torch.sum(x * x, dim=0)

        self.__mean = self.__sum / self.__n
        self.__std = torch.sqrt(self.__sum2 / self.__n - self.__mean ** 2)

    def transform(self, x):
        if self.__sum is None:
            raise Exception("Standard scaler has not been fit yet.")
        return (x - self.__mean) / self.__std

    def inverse_transform(self, x):
        if self.__sum is None:
            raise Exception("Standard scaler has not been fit yet.")
        return (x * self.__std) + self.__mean

    def __str__(self):
        return f"StandardScaler(n_features={self.__n_features})\nMean: {self.__mean}\nStd: {self.__std}"

    def __repr__(self):
        return self.__str__()


__all__ = ['StandardScaler']
