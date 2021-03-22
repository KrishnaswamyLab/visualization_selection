import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import scprep
import graphtools
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import sys
sys.path.append("..")
from blog_tools import embed, data, quantify

import pandas as pd
import numpy as np
import sys
import os
import tasklogger

from joblib import Parallel, delayed
from functools import partial

_PATH = "../{}/quant1/{{}}/{{}}.{}"
DATA_PATH = _PATH.format("data", "csv")
IMG_PATH = _PATH.format("img", "png")

def measure_method(data, data_noised, embedding, name='',
                   subsample_idx=None,
                  geodesic_dist=None, labels=None, seed=None):
    if subsample_idx is not None:
        data_noised = data_noised[subsample_idx]
    demap_score = quantify.DEMaP(
        data, embedding, knn=5, subsample_idx=subsample_idx, geodesic_dist=geodesic_dist)
    # onenn_score = quantify.shared_neighbors(data_noised, embedding, knn=2)
    # onenn_true_score = quantify.shared_neighbors(data, embedding, knn=2)
    onenn_score = quantify.onenn_score(labels, embedding)
    # auc_score = quantify.shared_neighbors_AUC(data, embedding, knn=50)
    ari_score = quantify.ari_score(labels, embedding, method='gmm')
    # svm_score = quantify.svm_score(labels, embedding)
    df = pd.DataFrame({'method': name,
                       'DEMaP': demap_score,
                      #'1nn_true': onenn_true_score,
                      'ARI': ari_score,
                      '1-nn': onenn_score,
                      #'SVM': svm_score
                      },
                      index=[''])
    return df


def measure_all_methods(load_fn,
                        dropout=None,
                        bcv=None, 
                        n_cells=None, 
                        n_genes=None,
                        n_jobs=6,
                        seed=None):
    dataset = load_fn(seed=seed, dropout=dropout, bcv=bcv, 
                          n_genes=n_genes)
    data_truth = dataset.X_true
    tasklogger.log_info("Calculating geodesic distances...")
    geodesic_dist = quantify.geodesic_distance(data_truth)
    data_noised = dataset.X
    if n_cells is not None and n_cells < Splatter.N_CELLS:
        subsample_idx = np.random.choice(
            data.shape[0], n_cells, replace=False)
    else:
        subsample_idx = None
    # embed
    tasklogger.log_info("Embedding...")
    methods = [m for m in embed.__all__ if not (m.__name__ in ['MDS', 'PHATE'])]
    embeddings = Parallel(6)(delayed(method)(data_noised, seed=seed) for method in methods)
    methods.append(embed.PHATE)
    embeddings.append(embed.PHATE(data_noised, seed=seed, n_jobs=10))
    methods.append(embed.MDS)
    embeddings.append(embed.MDS(data_noised, seed=seed, n_jobs=10))
    # plot
    tasklogger.log_info("Plotting...")
    fig, axes = plt.subplots(1, len(embeddings), figsize=(4*len(embeddings), 4))
    for embedding, ax, method in zip(embeddings, axes, methods):
        scprep.plot.scatter2d(embedding, ax=ax, label_prefix=method.__name__, 
                              ticks=False, c=dataset.c, legend=False)
    plt.tight_layout()
    fig.savefig(IMG_PATH.format(dataset.name, seed))
    # evaluate
    tasklogger.log_info("Evaluating...")
    results = [measure_method(embedding=embedding, data=data_truth, data_noised=data_noised,
                                    name=method.__name__, subsample_idx=subsample_idx,
                                    geodesic_dist=geodesic_dist, labels=dataset.c, seed=seed)
            for embedding, method in zip(embeddings, methods)]
    df = pd.concat(results)
    df = df.sort_values('DEMaP', ascending=False)
    print(df)
    return df

seed = int(sys.argv[1])
OVERWRITE = False

paths_out_path = DATA_PATH.format("paths", seed)
if OVERWRITE or not os.path.isfile(paths_out_path):
    paths_out = measure_all_methods(data.Paths, seed=seed)
    paths_out.to_csv(paths_out_path)

groups_out_path = DATA_PATH.format("groups", seed)
if OVERWRITE or not os.path.isfile(groups_out_path):
    groups_out = measure_all_methods(data.Groups, seed=int(sys.argv[1]))
    groups_out.to_csv(groups_out_path)


# for i in {0..9}; do
#   for j in {1..20}; do
#     k=$((i*20+j))
#     python run_quant.py $k &
#   done
#   wait
# done
