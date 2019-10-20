import os
import sys
from setuptools import setup, find_packages

install_requires = [
    'numpy>=1.10.0',
    'scipy>=0.18.0',
    'scikit-learn>=0.20.0',
    'pandas>=0.25',
    'imageio',
    'pygsp',
    'scprep',
    'graphtools',
    'phate',
    'umap-learn',
    'networkx',
    'ipywidgets',
    'plotly==3.10.0',
    'numba',
    'markdown',
    'matplotlib>=3.0'
]

extras_requires = [
    'h5py',
    'rpy2>=3.0',
]

package_name = "blog_tools"

if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >=3.6 required.")

version_py = os.path.join(os.path.dirname(
    __file__), package_name, 'version.py')
version = open(version_py).read().strip().split(
    '=')[-1].replace('"', '').strip()

readme = open('README.md').read()

setup(name=package_name,
      version=version,
      description=package_name,
      author='Scott Gigante and Daniel Burkhardt, Krishnaswamy Lab, Yale University',
      author_email='krishnaswamylab@gmail.com',
      packages=find_packages(),
      license='GNU General Public License Version 2',
      install_requires=install_requires,
      long_description=readme,
      url='https://github.com/KrishnaswamyLab/visualization_selection',
      download_url="https://github.com/KrishnaswamyLab/visualization_selection/archive/v{}.tar.gz".format(
          version),
      keywords=['big-data',
                'dimensionality-reduction',
                ],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Framework :: Jupyter',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Visualization',
      ]
      )
