### Principal Components Analysis

#### How does PCA work?

PCA and related eigendecomposition methods are some of the most fundamental dimensionality reduction tools in data science. Many methods, including tSNE, UMAP, and PHATE, first reduce the data using PCA before performing further operations on the data.

You can find many rigorous descriptions of the PCA algorithm online. Here, we will focus on the intutition. The goal of PCA is to identify a set of orthogonal dimensions (each of which is a linear combination of the input features) that explain the maximum variance in the data. Generally, this is performed by taking the eigenvectors of the covariance matrix of the data and ordering them by decreasing eigenvalues. If you're not so familiar with eigendecomposition, it's important to remember that this method identifies a set of orthogonal linear combinations of the columns of a matrix. If you consider the interpretation of eigenvalues as defining the amount of scaling of an eigenvector when multiplied by a matrix, then it makes sense that the eigenvector that is scaled the most when multiplied by the covariance matrix represents an axis of maximal variance within the data.

One of the major drawbacks of PCA is that dimensions used for visualization can only represent linear combinations of the feature space. This means that if there is non-linearity in the data, PCA will struggle to accruately represent this data structure in two or three dimensions.

#### PCA on toy data cases

![PCA on toy data](img/toy_data.PCA.png)

First, let's consider the **swiss roll**. Notice how the latent dimension -- the distance from the center of the spiral -- is not well represented by these first two dimensions? This is because the noise dimension is wider than the swiss roll is tall. This brings us to our first important insight about PCA. Because PCA finds linear combinations of the data features with maximal variance, if the dimension of largest variance is noise, then PCA will show the noise.

Furthermore, remember that our goal with the swiss roll was to "unroll" the data and show the data lying on a single line or a 2d plane. Because the data is non-linear, there's no way for PCA to perform this unfurling. We'll get back to this later when we look at datasets that can unroll the roll, but for now, remember that PCA can hide non-linear relationships between the data.

Second, we look at the **three blobs** dataset. Here, PCA did a good job of capturing the positioning of the data. This should be expected because the input data is a linear projection of the latent data. PCA is essentially just a rotation, and the latent space of the data here is 2-dimensional so PCA can well represent the data.

In the case of the **uneven circle**, we see that the data structure is more or less reasonably represented due to the same reasons as in the three blobs case. However, notice that the uneven spacing of the data is lost.

Examining the **tree** dataset, we can see that PCA failed to show the six branches in two dimensions and depicts many branches overlapping. If we were to be looking only at the data without the color added, you might guess that there are only two branches in this data. This squashing of branches is because the underlying data structure does not exist as two linear combinations of the data features. Different features change between branches, so one would need one principal component per branch to represent this data. Furthermore, the non-linearity of the changes in the features of each branch causes PCA to make some branches appear curves. Although this is true with respect to the feature space, this is unfaithful to the latent data generative space. As with the Swiss Roll, PCA has no way of "unfurling" the data.
