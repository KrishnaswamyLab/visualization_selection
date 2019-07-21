import numpy as np
import scipy.io
import pygsp
import sklearn.datasets
import abc
import scprep
import graphtools

from . import utils, embed


class Dataset(metaclass=abc.ABCMeta):

    def __init__(self, *args, tries=5, seed=42, is_graph=False, **kwargs):
        self.c = None
        self.seed = seed
        self.is_graph = is_graph
        np.random.seed(seed)
        self.build(*args, **kwargs)
        if not hasattr(self, "X_true"):
            self.X_true = self.X
        self.validate()

    @abc.abstractmethod
    def build(self):
        # mandatory
        self.X = np.array([])
        self.name = ''
        # optional
        self.X_true = np.array([])
        self.c = np.empty_like(self.X)  # default: None

    def validate(self):
        assert self.c is None or self.X.shape[0] == len(self.c)
        assert self.X_true.shape[0] == self.X.shape[0]

    def plot_truth(self, ax):
        scprep.plot.scatter2d(self.X_true, c=self.c, ticks=False, ax=ax,
                              title=self.name, legend=False)
        # fix limits if extremely imbalanced
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if xmax - xmin > 2 * (ymax - ymin):
            ax.set_ylim((-(xmax - xmin) / 2, (xmax - xmin) / 2))


class ImageDataset(Dataset, metaclass=abc.ABCMeta):

    def plot_truth(self, ax):
        select_idx = np.random.choice(len(self.X_true), 4)
        plot_image = np.vstack([
            np.hstack([
                self.X_true[select_idx[0]],
                np.repeat(125, self.X_true.shape[1])[:, None],
                self.X_true[select_idx[1]]]),
            np.repeat(125, self.X_true.shape[2] * 2 + 1)[None, :],
            np.hstack([
                self.X_true[select_idx[2]],
                np.repeat(125, self.X_true.shape[1])[:, None],
                self.X_true[select_idx[3]]])])
        ax.imshow(plot_image, cmap='Greys', aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(self.name, fontsize='xx-large')

    def validate(self):
        assert self.c is None or self.X.shape[0] == len(self.c)
        assert self.X_true.shape[0] == self.X.shape[0]
        assert len(self.X_true.shape) >= 3


class GraphDataset(Dataset, metaclass=abc.ABCMeta):

    def __init__(self, *args, tries=5, is_graph=True, **kwargs):
        super().__init__(*args, is_graph=is_graph, **kwargs)
        for _ in range(tries):
            self.build(*args, **kwargs)
            if np.isfinite(np.max(utils.geodesic_distance(self.X))):
                connected = True
                break
        if not connected:
            raise RuntimeError(
                "Graph is not connected after {} tries".format(tries))

    def validate(self):
        assert self.X.shape[0] == self.X.shape[1]
        assert np.all(self.X.diagonal() == 0), self.diagonal
        assert np.max(self.X) <= 1, np.max(self.X)
        assert np.min(self.X) >= 0, np.min(self.X)
        try:
            assert (self.X - self.X.T).nnz == 0
        except AttributeError:
            assert np.all(self.X == self.X.T)

    def plot_truth(self, ax):
        X = embed.Spring(self.X_true, is_graph=True)
        scprep.plot.scatter2d(X, c=self.c, ticks=False, ax=ax,
                              title=self.name, legend=False)


class grid(Dataset):

    def build(self, size=10, n_dim=20, noise=0.2):
        self.name = 'Grid'

        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        self.X_true = np.hstack([np.repeat(x, size)[:, None],
                                 np.tile(y, size)[:, None]])
        self.c = self.X_true[:, 0] + self.X_true[:, 1]
        R = np.random.normal(0, 1, (self.X_true.shape[1], n_dim))
        self.X = self.X_true @ R
        self.X += np.random.normal(0, noise, self.X.shape)


class swissroll(Dataset):

    def build(self, n_samples=500, noise=0.2):
        self.name = 'Swiss roll'

        t = 1.5 * np.pi * np.random.uniform(1, 3, (n_samples, 1))
        x = t * np.cos(t)
        y = np.random.uniform(0, 26, (n_samples, 1))
        z = t * np.sin(t)

        X = np.hstack((x, y, z))
        self.X = X
        self.X += np.random.normal(0, noise, self.X.shape)
        # move origin for perspective
        X += np.array([0, 200, -50])
        self.X_true = np.hstack(
            [X[:, [0]] / (X[:, [1]]), -1 * X[:, [2]] / (X[:, [1]])])
        self.c = np.squeeze(t)


class three_blobs(Dataset):

    def build(self, size=50, n_dim=200, noise=10):
        self.name = 'Three blobs'

        self.X_true = np.random.normal(0, 1, (size * 3, 2))
        self.X_true[:size, 0] += 10
        self.X_true[-size:, 0] += 50
        R = np.random.normal(0, 1, (self.X_true.shape[1], n_dim))
        self.X = self.X_true @ R
        self.X += np.random.normal(0, noise, self.X.shape)
        self.c = np.repeat(np.arange(3), size)


class uneven_circle(Dataset):

    def build(self, size=50, n_dim=20, noise=0.4):
        self.name = 'Uneven circle'

        theta = 2 * np.pi * np.random.uniform(0, 1, size)
        self.X_true = np.hstack(
            [np.cos(theta)[:, None], np.sin(theta)[:, None]])
        R = np.random.normal(0, 1, (self.X_true.shape[1], n_dim))
        self.X = self.X_true @ R
        self.X += np.random.normal(0, noise, self.X.shape)
        self.c = theta


class tree(Dataset):

    def build(self, size=500):
        self.name = 'Tree'
        #params = {'method': 'paths', 'batch_cells': size,
        #          'path_length': 500,
        #          'path_from': [0, 1, 1, 2, 0, 0],
        #          'de_fac_loc': 1, 'path_nonlinear_prob': 0.5,
        #          'dropout_type': 'binomial', 'dropout_prob': 0.5,
        #          'path_skew': [0.5, 0.75, 0.25, 0.5, 0.25, 0.75],
        #          'group_prob': [0.15, 0.05, .1, .25, .2, .25],
        #          'seed': self.seed, 'verbose': False}
        params = {'method': 'paths', 'batch_cells': 500,
          'path_length': 500,
          'path_from': [0, 1, 1, 2, 0, 0, 2],
          'de_fac_loc': 1,
          'path_skew': [0.45, 0.7, 0.7, 0.45, 0.65, 0.5, 0.5],
          'group_prob': [0.1, 0.1, .1, .2, .2, .2,.1],
          'dropout_type': 'binomial', 'dropout_prob': 0.5,
          'seed': self.seed, 'verbose': False}
        sim = scprep.run.SplatSimulate(**params)
        data = sim['counts']
        data_ln = scprep.normalize.library_size_normalize(data)
        data_sqrt = scprep.transform.sqrt(data_ln)
        self.X = data_sqrt
        self.c = sim['group']
        self.X = self.X[np.argsort(self.c)]
        self.c = np.sort(self.c)
        expand = 4
        params = {'method': 'paths', 'batch_cells': size * expand,
                  'out_prob': 0,
                  'path_length': 500,
                  'path_from': [0, 1, 1, 2, 0, 0],
                  'de_fac_loc': 1, 'path_nonlinear_prob': 0.5,
                  'group_prob': [0.15, 0.05, .1, .25, .2, .25],
                  'dropout_type': 'binomial', 'dropout_prob': 0,
                  'path_skew': [0.5, 0.55, 0.4, 0.5, 0.45, 0.6],
                  'group_prob': [0.15, 0.05, .1, .25, .2, .25],
                  'seed': self.seed, 'verbose': False}
        sim = scprep.run.SplatSimulate(**params)
        data = sim['counts']
        data = data[np.argsort(sim['group'])]
        data_ln = scprep.normalize.library_size_normalize(data)
        data_sqrt = scprep.transform.sqrt(data_ln)
        G = graphtools.Graph(data_sqrt, n_pca=100, anisotropy=1)
        self.X_true = embed.PHATE(G, gamma=0)[::expand]


class sensor(GraphDataset):

    def build(self, size=100):
        self.name = 'Sensor network'

        G = pygsp.graphs.DavidSensorNet(N=size)
        self.X = G.W.tocsr()
        self.c = G.dw


class sbm(GraphDataset):

    def build(self, n=3, p=0.3, q=0.05, size=100, truth_weight=0.15):
        self.name = 'Stochastic block model'
        G = pygsp.graphs.StochasticBlockModel(
            N=size, k=n, p=p, q=q, seed=self.seed)
        self.X = G.W.tocsr()
        self.c = G.info['node_com']
        self.X_true = self.X.tocoo().astype(float)
        for i in range(n - 1):
            for j in range(i + 1, n):
                self.X_true.data[
                    np.isin(self.X_true.row, np.argwhere(self.c == i)) &
                    np.isin(self.X_true.col, np.argwhere(self.c == j))] *= truth_weight
                self.X_true.data[
                    np.isin(self.X_true.row, np.argwhere(self.c == j)) &
                    np.isin(self.X_true.col, np.argwhere(self.c == i))] *= truth_weight


class BarabasiAlbert(GraphDataset):

    def build(self, size=200):
        self.name = 'Barabasi Albert'

        G = pygsp.graphs.BarabasiAlbert(size)
        self.X = G.W.tocsr()
        self.c = G.dw


class digits(ImageDataset):

    def build(self, digit=7):
        self.name = 'Digits'
        digits = sklearn.datasets.load_digits()
        self.X = digits['data']
        if digit is not None:
            self.X = self.X[digits['target'] == digit]
        self.X = self.X / 16
        self.c = self.X.sum(axis=1)
        self.X_true = self.X.reshape(-1, 8, 8)
        self.X_true = np.round(self.X_true * 255).astype(np.uint8)


class frey(ImageDataset):

    def build(self, size=200):
        self.name = 'Frey faces'
        url = 'http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat'
        filename = 'data/frey_rawface.mat'
        utils.download_file(url, filename)
        img_rows = 28
        img_cols = 20
        self.X_true = scipy.io.loadmat(filename)
        self.X_true = 255 - \
            self.X_true["ff"].T.reshape((-1, img_rows, img_cols))
        if size is not None:
            self.X_true = self.X_true[np.linspace(
                0, self.X_true.shape[0] - 1, size).astype(int)]
        self.X = self.X_true.reshape((self.X_true.shape[0], -1))
        self.X = self.X / 255
        self.c = np.arange(self.X_true.shape[0])


__all__ = [swissroll, three_blobs, uneven_circle,
           digits, frey, tree]
