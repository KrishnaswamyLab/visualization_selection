import os
import urllib.request
import graphtools


def download_file(url, filename):
    if not os.path.isfile(filename):
        try:
            os.mkdir(os.path.dirname(filename))
        except FileExistsError:
            pass
    urllib.request.urlretrieve(url, filename)


def geodesic_distance(A):
    G = graphtools.Graph(A, precomputed='adjacency')
    return G.shortest_path()
