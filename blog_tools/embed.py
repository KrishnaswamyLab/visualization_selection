import numpy as np
import phate
import scprep
import sklearn.decomposition
import sklearn.manifold
import umap
import pygsp
import graphtools
import networkx
from . import utils


from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_array
from sklearn.utils.graph import graph_shortest_path
from sklearn.decomposition import KernelPCA


class Isomap_(sklearn.manifold.Isomap):

    def __init__(self, *args, precomputed=False, **kwargs):
        self.precomputed = precomputed
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
                                     n_jobs=self.n_jobs)
        self.embedding_ = self.kernel_pca_.fit_transform(G)


def PHATE(X, *args, is_graph=False, knn_dist='euclidean', verbose=0, **kwargs):
    if knn_dist is None:
        if is_graph:
            knn_dist = 'precomputed'
    return phate.PHATE(*args, knn_dist=knn_dist, verbose=verbose,
                       **kwargs).fit_transform(X)


def UMAP(X, *args, is_graph=False, **kwargs):
    return umap.UMAP(*args, **kwargs).fit_transform(X)


def MDS(X, *args, is_graph=False, dissimilarity='euclidean', **kwargs):
    if is_graph:
        X = utils.geodesic_distance(X)
        dissimilarity = 'precomputed'
    return sklearn.manifold.MDS(*args, dissimilarity=dissimilarity,
                                **kwargs).fit_transform(X)


def TSNE(X, *args, is_graph=False, metric='euclidean', **kwargs):
    if is_graph:
        X = utils.geodesic_distance(X)
        metric = 'precomputed'
    return sklearn.manifold.TSNE(*args, metric=metric, **kwargs).fit_transform(X)


def ISOMAP(X, *args, is_graph=False, random_state=None, **kwargs):
    np.random.seed(random_state)
    if is_graph:
        X = utils.geodesic_distance(X)
    return Isomap_(*args, precomputed=is_graph, **kwargs).fit_transform(X)


def PCA(X, *args, is_graph=False, **kwargs):
    X = scprep.utils.toarray(X)
    return sklearn.decomposition.PCA(*args, **kwargs).fit_transform(X)


def Spring(X, *args, is_graph=False, **kwargs):
    if not is_graph:
        G = graphtools.Graph(X, knn=3, decay=None, use_pygsp=True)
    else:
        G = pygsp.graphs.Graph(X)
    G = networkx.from_numpy_matrix(G.W.toarray())
    X = networkx.spring_layout(G, *args, **kwargs)
    X = np.vstack(list(X.values()))
    return X


__all__ = [PCA, MDS, ISOMAP, TSNE, UMAP, PHATE]
