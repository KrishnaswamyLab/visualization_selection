import graphtools
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def DEMaP(data, embedding, knn=30, subsample_idx=None):
    geodesic_dist = geodesic_distance(data, knn=knn)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    geodesic_dist = squareform(geodesic_dist)
    embedded_dist = pdist(embedding)
    return spearmanr(geodesic_dist, embedded_dist).correlation


def geodesic_distance(data, knn=30, distance='data'):
    G = graphtools.Graph(data, knn=knn, decay=None)
    return G.shortest_path(distance=distance)

def shared_neighbors_AUC(data, embedding, knn=200, n_jobs=20):
    return np.mean(shared_neighbors_curve(data, embedding, knn=knn, n_jobs=n_jobs))

def shared_neighbors_curve(data, embedding, knn=200, n_jobs=20):
    knn_op = neighbors.NearestNeighbors(knn, n_jobs=n_jobs)
    knn_embedding = knn_op.fit(embedding).kneighbors(embedding, return_distance=False)
    knn_data = knn_op.fit(data).kneighbors(data, return_distance=False)
    return [np.mean(np.isin(knn_embedding[:, :i], knn_data[:, :i]))
            for i in range(1, knn)]

def shared_neighbors(data, embedding, knn=200, n_jobs=20):
    knn_op = neighbors.NearestNeighbors(knn, n_jobs=n_jobs)
    knn_embedding = knn_op.fit(embedding).kneighbors(embedding, return_distance=False)
    knn_data = knn_op.fit(data).kneighbors(data, return_distance=False)
    return np.mean(np.isin(knn_embedding, knn_data))