import numpy as np
import scipy.io
import pygsp
import sklearn.datasets
import abc
import scprep
import graphtools

from . import utils, embed


class Dataset(metaclass=abc.ABCMeta):

    def __init__(self, *args, tries=5, **kwargs):
        self.c = None
        self.is_graph = False
        self.build(*args, **kwargs)

        if self.is_graph:
            for _ in range(tries):
                self.build(*args, **kwargs)
                if np.isfinite(np.max(utils.geodesic_distance(self.X))):
                    connected = True
                    break
            if not connected:
                raise RuntimeError(
                    "Graph is not connected after {} tries".format(tries))

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

    def build(self, size=10, n_dim=3):
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        self.X_true = np.hstack([np.repeat(x, size)[:, None],
                                 np.tile(y, size)[:, None]])
        self.X = np.hstack([self.X_true, np.random.normal(
            0, 1, (self.X_true.shape[0], n_dim - self.X_true.shape[1]))])
        self.c = self.X[:, 0] + self.X[:, 1]


class three_blobs(Dataset):

    def build(self, size=50, n_dim=100):
        self.X_true = np.random.normal(0, 1, (size * 3, 2))
        self.X_true[:size, 0] += 10
        self.X_true[-size:, 0] += 50
        self.X = np.hstack([self.X_true, np.random.normal(
            0, 1, (self.X_true.shape[0], n_dim - self.X_true.shape[1]))])
        self.c = np.repeat(np.arange(3), size)


class uneven_circle(Dataset):

    def build(self, size=50, n_dim=3):
        theta = 2 * np.pi * np.random.uniform(0, 1, size)
        self.X_true = np.hstack(
            [np.cos(theta)[:, None], np.sin(theta)[:, None]])
        self.X = np.hstack([self.X_true, np.random.normal(
            0, 1, (self.X_true.shape[0], n_dim - self.X_true.shape[1]))])
        self.c = theta


class tree(Dataset):

    def build(self, size=500, seed=41):
        params = {'method': 'paths', 'batch_cells': size,
                  'path_length': 500, 'group_prob': [0.05, 0.05, .1, .3, .2, .3], 'path_from': [0, 1, 1, 2, 0, 0],
                  'de_fac_loc': 0.75, 'dropout_type': 'binomial', 'dropout_prob': 0.5, 'bcv_common': 0.05,
                  'path_skew': [0.5, 0.75, 0.25, 0.5, 0.25, 0.9], 'path_nonlinear_prob': 0.5,
                  'seed': seed, 'verbose': False}
        sim = scprep.run.SplatSimulate(**params)
        data = sim['counts']
        data_ln = scprep.normalize.library_size_normalize(data)
        data_sqrt = scprep.transform.sqrt(data_ln)
        self.X = data_sqrt
        self.c = sim['group']
        self.X = self.X[np.argsort(self.c)]
        self.c = np.sort(self.c)
        params = {'method': 'paths', 'batch_cells': size * 20,
                  'path_length': 500, 'group_prob': [0.05, 0.05, .1, .3, .2, .3], 'path_from': [0, 1, 1, 2, 0, 0],
                  'de_fac_loc': 0.75, 'dropout_type': 'binomial', 'dropout_prob': 0, 'bcv_common': 0.15,
                  'path_skew': [0.5, 0.55, 0.4, 0.5, 0.45, 0.6], 'path_nonlinear_prob': 0.5,
                  'seed': seed, 'verbose': False}
        sim = scprep.run.SplatSimulate(**params)
        data = sim['counts']
        data = data[np.argsort(sim['group'])]
        data_ln = scprep.normalize.library_size_normalize(data)
        data_sqrt = scprep.transform.sqrt(data_ln)
        G = graphtools.Graph(data_sqrt, n_pca=100, anisotropy=1)
        self.X_true = embed.PHATE(G, gamma=0)[::20]


class digits(Dataset):

    def build(self, digit=7):
        digits = sklearn.datasets.load_digits()
        self.X = digits['data']
        if digit is not None:
            self.X = self.X[digits['target'] == digit]
        self.X = self.X / 16
        self.c = self.X.sum(axis=1)
        self.X_raw = self.X.reshape(-1, 8, 8)
        self.X_raw = np.round(self.X_raw * 255).astype(np.uint8)


class sensor(Dataset):

    def build(self, size=100):
        G = pygsp.graphs.DavidSensorNet(N=size)
        self.X = G.W
        self.is_graph = True
        self.c = G.dw


class sbm(Dataset):

    def build(self, n=3, p=0.3, q=0.05, size=100):
        G = pygsp.graphs.StochasticBlockModel(N=size, k=n, p=p, q=q)
        self.X = G.W
        G_true = pygsp.graphs.StochasticBlockModel(
            N=size, k=n, p=p ** 1 / 2, q=q ** 2)
        self.X_true = G_true.W
        self.c = G.info['node_com']
        self.is_graph = True


class BarabasiAlbert(Dataset):

    def build(self, size=200):
        G = pygsp.graphs.BarabasiAlbert(size)
        self.X = G.W
        self.c = G.dw
        self.is_graph = True


class frey(Dataset):

    def build(self, size=200):
        url = 'http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat'
        filename = 'data/frey_rawface.mat'
        utils.download_file(url, filename)
        img_rows = 28
        img_cols = 20
        self.X_raw = scipy.io.loadmat(filename)
        self.X_raw = 255 - self.X_raw["ff"].T.reshape((-1, img_rows, img_cols))
        if size is not None:
            self.X_raw = self.X_raw[np.linspace(
                0, self.X_raw.shape[0] - 1, size).astype(int)]
        self.X = self.X_raw.reshape((self.X_raw.shape[0], -1))
        self.X = self.X / 255
        self.c = np.arange(self.X_raw.shape[0])


__all__ = [grid, three_blobs, uneven_circle, sbm,
           digits, frey, tree]
