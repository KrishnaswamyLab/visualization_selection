import numpy as np
import scipy.io
import pygsp

from . import utils


def grid(size=20):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    data = np.hstack([np.repeat(x, size)[:, None]. np.tile(y, size)[:, None]])
    return data


def three_blobs(dim=2, size=50):
    data = np.random.normal(0, 1, (size * 3, dim))
    data[:size, 0] += 10
    data[-size:, 0] += 50
    return data


def uneven_circle(size=50):
    theta = 2 * np.pi * np.random.uniform(0, 1, size)
    data = np.hstack([np.cos(theta)[:, None], np.sin(theta)[:, None]])
    return data


def tree(size=100):
    # dan to add
    pass


def sensor(size=100):
    G = pygsp.graphs.Sensor
    return G.W


def frey():
    url = 'http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat'
    filename = 'data/frey_rawface.mat'
    utils.download_file(url, filename)
    img_rows = 28
    img_cols = 20
    data = scipy.io.loadmat(filename)
    data = data["ff"].T.reshape((-1, 1, img_rows, img_cols))
    data = data / 255
    return data
