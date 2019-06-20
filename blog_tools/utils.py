import os
import urllib.requests


def download_file(url, filename):
    if not os.path.isfile(filename):
        try:
            os.mkdir(os.path.dirname(filename))
        except FileExistsError:
            pass
    urllib.request.urlretrieve(url, filename)
