import numpy as np
import phate
import scprep
import sklearn.decomposition
import sklearn.manifold
import umap
import pygsp
import graphtools
import networkx
import tasklogger
from . import utils


from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_array
from sklearn.utils.graph import graph_shortest_path
from sklearn.decomposition import KernelPCA


class Isomap_(sklearn.manifold.Isomap):

    def __init__(self, *args, precomputed=False, random_state=None, **kwargs):
        self.precomputed = precomputed
        self.random_state = random_state
        super().__init__(*args, **kwargs)

    def _fit_transform(self, X):
        if not self.precomputed:
            X = check_array(X, accept_sparse='csr')
            self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                          algorithm=self.neighbors_algorithm,
                                          n_jobs=self.n_jobs)
            self.nbrs_.fit(X)
            self.training_data_ = self.nbrs_._fit_X

            kng = kneighbors_graph(self.nbrs_, self.n_neighbors,
                                   mode='distance', n_jobs=self.n_jobs)

            self.dist_matrix_ = graph_shortest_path(kng,
                                                    method=self.path_method,
                                                    directed=False)
        else:
            X = check_array(X)
            self.dist_matrix_ = X

        G = self.dist_matrix_ ** 2
        G *= -0.5

        self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                     kernel="precomputed",
                                     eigen_solver=self.eigen_solver,
                                     tol=self.tol, max_iter=self.max_iter,
                                     n_jobs=self.n_jobs,
                                    random_state=self.random_state)
        self.embedding_ = self.kernel_pca_.fit_transform(G)


def PHATE(X, *args, is_graph=False, knn_dist='euclidean', verbose=0, seed=None, n_jobs=15, **kwargs):
    if knn_dist is None:
        if is_graph:
            knn_dist = 'precomputed'
    tasklogger.log_start("PHATE")
    Y = phate.PHATE(*args, knn_dist=knn_dist, verbose=verbose,
                       random_state=seed, n_jobs=n_jobs,
                       **kwargs).fit_transform(X)
    tasklogger.log_complete("PHATE")
    return Y


def UMAP(X, *args, is_graph=False, seed=None, **kwargs):
    tasklogger.log_start("UMAP")
    Y = umap.UMAP(*args, random_state=seed, **kwargs).fit_transform(X)
    tasklogger.log_complete("UMAP")
    return Y


def MDS(X, *args, is_graph=False, dissimilarity='euclidean', seed=None, n_jobs=15, **kwargs):
    if is_graph:
        X = utils.geodesic_distance(X)
        dissimilarity = 'precomputed'
    tasklogger.log_start("MDS")
    Y = sklearn.manifold.MDS(*args, dissimilarity=dissimilarity, 
                                random_state=None, n_jobs=n_jobs,
                                **kwargs).fit_transform(X)
    tasklogger.log_complete("MDS")
    return Y


def TSNE(X, *args, is_graph=False, metric='euclidean', seed=None, **kwargs):
    if is_graph:
        X = utils.geodesic_distance(X)
        metric = 'precomputed'
    tasklogger.log_start("TSNE")
    Y = sklearn.manifold.TSNE(*args, metric=metric, random_state=seed, **kwargs).fit_transform(X)
    tasklogger.log_complete("TSNE")
    return Y


def ISOMAP(X, *args, is_graph=False, seed=None, **kwargs):
    np.random.seed(seed)
    if is_graph:
        X = utils.geodesic_distance(X)
    tasklogger.log_start("ISOMAP")
    Y = Isomap_(*args, precomputed=is_graph, random_state=seed, **kwargs).fit_transform(X)
    tasklogger.log_complete("ISOMAP")
    return Y


def PCA(X, *args, is_graph=False, seed=None, **kwargs):
    X = scprep.utils.toarray(X)
    tasklogger.log_start("PCA")
    Y = sklearn.decomposition.PCA(*args, random_state=seed, **kwargs).fit_transform(X)
    tasklogger.log_complete("PCA")
    return Y


def Spring(X, *args, is_graph=False, seed=None, **kwargs):
    np.random.seed(seed)
    if not is_graph:
        G = graphtools.Graph(X, knn=3, decay=None, use_pygsp=True)
    else:
        G = pygsp.graphs.Graph(X)
    G = networkx.from_numpy_matrix(G.W.toarray())
    tasklogger.log_start("Spring")
    X = networkx.spring_layout(G, *args, **kwargs)
    tasklogger.log_complete("Spring")
    X = np.vstack(list(X.values()))
    return X


__all__ = [PCA, MDS, ISOMAP, TSNE, UMAP, PHATE]
