import graphtools
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from sklearn import neighbors, cluster, metrics, svm, model_selection, mixture
from joblib import Parallel, delayed


def DEMaP(data, embedding, knn=30, subsample_idx=None, geodesic_dist=None):
    if geodesic_dist is None:
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
    return [[np.mean(np.isin(knn_embedding[j, 1:i], knn_data[j, 1:i]))
        for j in range(knn_data.shape[0])]
            for i in range(2, knn)]

def shared_neighbors(data, embedding, knn=200, n_jobs=20):
    knn_op = neighbors.NearestNeighbors(knn, n_jobs=n_jobs)
    knn_embedding = knn_op.fit(embedding).kneighbors(embedding, return_distance=False)
    knn_data = knn_op.fit(data).kneighbors(data, return_distance=False)
    return np.mean([np.isin(knn_embedding[i,1:], knn_data[i,1:]) for i in range(knn_data.shape[0])])

def compute_ari(labels, embedding, n_clusters=8, random_state=None, method='kmeans'):
    if method == 'kmeans':
        cluster_op = cluster.KMeans(n_clusters, random_state=random_state)
    elif method == 'gmm':
        cluster_op = mixture.GaussianMixture(n_clusters, random_state=random_state)
    else:
        raise NotImplementedError
    clusters = cluster_op.fit_predict(embedding)
    return metrics.adjusted_rand_score(labels, clusters)

def ari_score(labels, embedding, n_jobs=20, n_reps=100, random_state=None, method='kmeans'):
    np.random.seed(random_state)
    labels, label_names = pd.factorize(labels)
    n_clusters = len(label_names)
    ari_scores = Parallel(n_jobs)(delayed(compute_ari)(labels, embedding, n_clusters, random_state=seed) for seed in np.random.choice(2**32-1, n_reps, replace=True))
    return np.mean(ari_scores)


def compute_svm(labels, embedding, random_state=None):
    np.random.seed(random_state)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(embedding, labels, test_size=0.2, random_state=random_state)
    svm_op = svm.SVC(random_state=random_state).fit(X_train, y_train)
    return svm_op.score(X_test, y_test)


def svm_score(labels, embedding, n_jobs=10, n_reps=10, random_state=None):
    np.random.seed(random_state)
    labels, label_names = pd.factorize(labels)
    n_clusters = len(label_names)
    svm_scores = Parallel(n_jobs)(delayed(compute_svm)(labels, embedding, random_state=seed) for seed in np.random.choice(2**32-1, n_reps, replace=True))
    return np.mean(svm_scores)


def compute_onenn(labels, embedding, random_state=None):
    np.random.seed(random_state)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(embedding, labels, test_size=0.2, random_state=random_state)
    knn_op = neighbors.KNeighborsClassifier(1).fit(X_train, y_train)
    return knn_op.score(X_test, y_test)


def onenn_score(labels, embedding, n_jobs=10, n_reps=10, random_state=None):
    np.random.seed(random_state)
    labels, label_names = pd.factorize(labels)
    n_clusters = len(label_names)
    onenn_scores = Parallel(n_jobs)(delayed(compute_onenn)(labels, embedding, random_state=seed) for seed in np.random.choice(2**32-1, n_reps, replace=True))
    return np.mean(onenn_scores)
