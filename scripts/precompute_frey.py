import sys
sys.path.append("..")  # noqa
from blog_tools import data, embed

import pickle

dataset = data.frey()
results = {}
for algorithm in embed.__all__:
    results[algorithm.__name__] = algorithm(dataset.X)

results['color'] = dataset.c
results['images'] = dataset.X_true

with open("../data/frey.pickle", 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL-1)
