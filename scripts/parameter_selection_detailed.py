import sys
sys.path.append("..")  # noqa
from blog_tools import data, embed
import pickle
import numpy as np
import scipy.spatial
from joblib import Parallel, delayed


dataset = data.swissroll()
results = {}


def scale(Y):
    for i in range(Y.shape[1]):
        Y[:, i] -= np.min(Y[:, i])
        Y[:, i] /= np.max(Y[:, i])
    return Y


metric = [True, False]
random_seed = [42, 43, 44, 45]
knn = [2, 3, 5, 10, 15, 25]
decay = [2, 5, 10, 20, 40, 200]
gamma = [-1, 0, 1]
min_dist = [0, 0.1, 0.2, 0.5, 0.8]
perplexity = [2, 5, 15, 30, 50, 100]
early_exaggeration = [1, 5, 12, 20, 50]
learning_rate = [5, 10, 50, 200, 500, 2000]

# PCA
algorithm = embed.PCA
results['PCA'] = {'param_names': ['seed'],
                  'params': [random_seed],
                  'default': (42,),
                  'output': {}}
for seed in random_seed:
    results['PCA']['output'][(seed,)] = scale(
        algorithm(dataset.X, seed=seed))

# MDS
algorithm = embed.MDS
results['MDS'] = {'param_names': ['seed'],
                  'params': [random_seed],
                  'default': (42,),
                  'output': {}}
for seed in random_seed:
    results['MDS']['output'][(seed,)] = scale(
        algorithm(dataset.X, seed=seed))

# ISOMAP
algorithm = embed.ISOMAP
results['ISOMAP'] = {'param_names': ['knn', 'seed'],
                     'params': [knn, random_seed],
                     'default': (5, 42),
                     'output': {}}
for k in knn:
    for seed in random_seed:
        results['ISOMAP']['output'][(k, seed)] = scale(algorithm(
            dataset.X, n_neighbors=k, seed=seed))

# TSNE
algorithm = embed.TSNE
results['TSNE'] = {'param_names': ['perplexity', 'learning_rate', 'early_exaggeration', 'seed'],
                   'params': [perplexity, learning_rate, early_exaggeration, random_seed],
                   'default': (30, 200, 12, 42),
                   'output': {}}
for p in perplexity:
    for l in learning_rate:
        for e in early_exaggeration:
            for seed in random_seed:
                results['TSNE']['output'][(p, l, e, seed)] = scale(algorithm(
                    dataset.X, perplexity=p, learning_rate=l, early_exaggeration=e, seed=seed))

# UMAP
algorithm = embed.UMAP
results['UMAP'] = {'param_names': ['knn', 'min_dist', 'seed'],
                   'params': [knn, min_dist, random_seed],
                   'default': (15, 0.1, 42),
                   'output': {}}
for k in knn:
    for m in min_dist:
        for seed in random_seed:
            complete = False
            actual_seed = seed
            while not complete:
                try:
                    results['UMAP']['output'][(k, m, seed)] = scale(algorithm(
                        dataset.X, n_neighbors=k, min_dist=m, seed=actual_seed))
                    complete = True
                except np.linalg.LinAlgError:
                    actual_seed += 100

# PHATE
algorithm = embed.PHATE
results['PHATE'] = {'param_names': ['knn', 'decay', 'gamma', 'seed'],
                    'params': [knn, decay, gamma, random_seed],
                    'default': (5, 40, 1, 42),
                    'output': {}}
for k in knn:
    for a in decay:
        for g in gamma:
            for seed in random_seed:
                results['PHATE']['output'][(k, a, g, seed)] = scale(algorithm(
                    dataset.X, knn=k, decay=a, gamma=g, seed=seed, n_jobs=10))

# Procrustes on everything
for algorithm in results.keys():
    base = list(results[algorithm]['output'].values())[0]
    for params in results[algorithm]['output'].keys():
        output = results[algorithm]['output'][params]
        _, output, _ = scipy.spatial.procrustes(base, output)
        # scale to 0, 1
        output -= np.min(output, axis=0)
        output /= np.max(output)
        results[algorithm]['output'][params] = output

# Add raw data
results['input'] = dataset.X_true
results['color'] = dataset.c


with open("../data/parameter_search_detailed.pickle", 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL-1)
