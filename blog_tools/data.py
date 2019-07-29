import numpy as np
import imageio
import scipy.io
import pygsp
import sklearn.datasets
import abc
import scprep
import scprep.io.hdf5 as h5
import graphtools
import pandas as pd

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
        params = {'method': 'paths', 'batch_cells': size,
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


def _sum_to_one(x):
    x = x / np.sum(x)  # fix numerical error
    x = x.round(3)
    if np.sum(x) != 1:
        x[0] += 1 - np.sum(x)
    x = x.round(3)
    return x
        

class trajectory(Dataset):

    def build(self, size=10000, method='paths', dropout=0.5, bcv=0.18, 
              # hyperparameters
              group_prob_rate=10, 
              group_prob_concentration=1,
              path_skew_shape=10,
              **kwargs):
        self.name = 'Tree'
        params = dict(method=method, seed=self.seed,
            batch_cells=size,
            n_genes=17580,
            mean_shape=6.6, mean_rate=0.45,
            lib_loc=8.4 + np.log(2), lib_scale=0.33,
            out_prob=0.016, out_fac_loc=5.4, out_fac_scale=0.90,
            bcv_common=bcv, bcv_df=21.6,
            de_prob=0.2,
            dropout_type="binomial", dropout_prob=dropout,
                     verbose=False)
        np.random.seed(self.seed)
        n_groups = np.random.poisson(group_prob_rate)
        group_prob = np.random.dirichlet(
            np.ones(n_groups) * group_prob_concentration).round(3)
        params['group_prob'] = _sum_to_one(group_prob)
        if method == 'paths':
            params['path_nonlinear_prob'] = np.random.uniform(0, 1)
            params['path_skew'] = np.random.beta(path_skew_shape, path_skew_shape, n_groups)
            params['path_from'] = [0]
            for i in range(1, n_groups):
                params['path_from'].append(np.random.randint(i))
        params.update(kwargs)
        sim = scprep.run.SplatSimulate(**params)
        data = sim['counts']
        data_ln = scprep.normalize.library_size_normalize(data)
        data_sqrt = scprep.transform.sqrt(data_ln)
        data_pca = scprep.reduce.pca(data_sqrt, 100)
        self.X = data_pca
        self.c = sim['group']
        # params['out_prob'] = 0
        params['dropout_prob'] = 0
        params['bcv_common'] = 0
        sim = scprep.run.SplatSimulate(**params)
        data = sim['counts']
        data_ln = scprep.normalize.library_size_normalize(data)
        data_sqrt = scprep.transform.sqrt(data_ln)
        data_pca = scprep.reduce.pca(data_sqrt, 100)
        self.X_true = data_pca


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
        self.c = np.arange(self.X_true.shape[0]).astype(float)

class COIL20(ImageDataset):
    
    def build(self):
        N = 20
        M = 72
        imsize = 128 * 128
        X = np.zeros((N * M, imsize))
        counter = 0
        for i in range(N):
            for j in range(M):
                X[counter, :] = imageio.imread(
                    '/data/lab/DataSets/COIL20/img/obj%d__%d.png' % (i + 1, j)).flatten()
                counter += 1
        labels = np.repeat(np.arange(1, N+1), M)
        time = np.tile(np.arange(M), N)
        self.X = X
        self.c = labels
        self.t = time
        self.X_true = self.X.reshape(-1, 128, 128)
        self.name = "COIL20"

class retina(Dataset):
    
    def build(self):
        clusters = pd.read_csv("/data/scottgigante/datasets/shekhar_retinal_bipolar/retina_clusters.tsv",
                               sep="\t", index_col=0)
        cells = pd.read_csv(
            "/data/scottgigante/datasets/shekhar_retinal_bipolar/retina_cells.csv",
            header=None,
            index_col=False).values.reshape(-1)[:-1]
        with h5.open_file("/data/scottgigante/datasets/shekhar_retinal_bipolar/retina_data.mat", 
                          'r', backend='h5py') as f:
            data = pd.DataFrame(
                np.array(h5.get_node(f, 'data')).T,
                index=cells)
        merged_data = pd.merge(data, clusters, how='left',
                               left_index=True, right_index=True)
        merged_data = merged_data.loc[~np.isnan(merged_data['CLUSTER'])]
        data = merged_data[merged_data.columns[:-2]]
        data = scprep.filter.filter_rare_genes(data)
        data = scprep.normalize.library_size_normalize(data)
        data = np.sqrt(data)
        clusters, labels = pd.factorize(
            merged_data[merged_data.columns[-1]])
        # labels = ['11', '23', '5', '4', '1', '3', '10', '6', '16_1', '2', '13', '14', '7',
        #   '12', '15_1', '9', '18', '17', '8', '15_2', '24', '21', '19', '16_2',
        #   '20', '22', '25', '26']
        cluster_assign = {
            '1': 'Rod BC',
            '2': 'Muller Glia',
            '7': 'BC1A',
            '9': 'BC1B',
            '10': 'BC2',
            '12': 'BC3A',
            '8': 'BC3B',
            '14': 'BC4',
            '3':  'BC5A',
            '13': 'BC5B',
            '6': 'BC5C',
            '11': 'BC5D',
            '5': 'BC6',
            '4': 'BC7',
            '15_1': 'BC8/9_1',
            '15_2': 'BC8/9_2',
            '16_1':  'Amacrine_1',
            '16_2':  'Amacrine_2',
            '20': 'Rod PR',
            '22': 'Cone PR',
        }
        labels = np.array(labels)
        for label, celltype in cluster_assign.items():
            labels = np.where(labels == label, celltype, labels)
        self.X = data
        self.c = labels[clusters]
        self.name = "Retinal Bipolar"

__all__ = [swissroll, three_blobs, uneven_circle, tree]
