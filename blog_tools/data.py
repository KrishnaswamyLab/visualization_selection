import numpy as np
import scipy.io
import pygsp
import sklearn.datasets
import abc

from . import utils


class Dataset(metaclass=abc.ABCMeta):

    def __init__(self, *args, **kwargs):
        self.c = None
        self.is_graph = False
        self.build(*args, **kwargs)
        assert self.c is None or self.X.shape[0] == len(self.c)
        if self.is_graph:
            assert self.X.shape[0] == self.X.shape[1]
            assert np.all(self.X.diagonal() == 0), self.diagonal
            assert np.max(self.X) <= 1, np.max(self.X)
            assert np.min(self.X) >= 0, np.min(self.X)
            try:
                assert (self.X - self.X.T).nnz == 0
            except AttributeError:
                assert np.all(self.X == self.X.T)

    @abc.abstractmethod
    def build(self):
        # mandatory
        self.X = np.array([])
        # optional
        self.X_raw = np.array([])
        self.c = np.empty_like(self.X)  # default: None
        self.is_graph = True  # default: False

    @property
    def name(self):
        return type(self).__name__


class grid(Dataset):

    def build(self, size=20):
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        self.X = np.hstack([np.repeat(x, size)[:, None],
                            np.tile(y, size)[:, None]])
        self.c = self.X[:, 0]


class three_blobs(Dataset):

    def build(self, dim=2, size=50):
        self.X = np.random.normal(0, 1, (size * 3, dim))
        self.X[:size, 0] += 10
        self.X[-size:, 0] += 50
        self.c = np.repeat(np.arange(3), size)


class uneven_circle(Dataset):

    def build(self, size=50):
        theta = 2 * np.pi * np.random.uniform(0, 1, size)
        self.X = np.hstack([np.cos(theta)[:, None], np.sin(theta)[:, None]])
        self.c = theta


# class tree(Dataset):
    # def build(self,size=100):
    #     # dan to add
    #     pass


class digits(Dataset):

    def build(self, digit=7):
        digits = sklearn.datasets.load_digits()
        self.X = digits['data']
        if digit is not None:
            self.X = self.X[digits['target'] == digit]
        self.X = self.X / 255


class sensor(Dataset):

    def build(self, size=100):
        G = pygsp.graphs.Sensor(N=size)
        self.X = G.W
        self.is_graph = True


class frey(Dataset):

    def build(self):
        url = 'http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat'
        filename = 'data/frey_rawface.mat'
        utils.download_file(url, filename)
        img_rows = 28
        img_cols = 20
        self.X_raw = scipy.io.loadmat(filename)
        self.X_raw = self.X_raw["ff"].T.reshape((-1, img_rows, img_cols))
        self.X_raw = self.X_raw / 255
        self.X = self.X_raw.reshape((self.X_raw.shape[0], -1))
        self.c = np.arange(self.X_raw.shape[0])


__all__ = [grid, three_blobs, uneven_circle, digits, sensor, frey]
