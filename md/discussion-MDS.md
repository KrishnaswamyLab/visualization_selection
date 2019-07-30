### MDS (Multidimensional scaling)

#### How does MDS work?

Unlike the other methods here, MDS actually refers to a collection algorithms. The algorithms are often broken down into three categories: classical, metric, and non-metric. You can find full descriptions and comparison of these algorithms [elsewhere](https://www.springer.com/gp/book/9780387251509), but here we will be using a metric MDS algorithm implemented in `sklearn.manifold.MDS`. A quick intuitive explanation of metric MDS is that minimizes the difference between high-dimensional euclidean distances and low-dimensional euclidean distances. The specific algorithm used in `sklearn` is called [SMACOF](https://escholarship.org/uc/item/4ps3b5mj), which minimizes the following cost function:

$$ \sigma (X) = \sum_{i,j \in X} [d(X_{i,j}) - d(x_{i,j})]^2 $$

Here, $X$ is the data in high dimensional space and $x$ is the distance in low dimensional space and $d$ is the Euclidean distance function. To solve this problem, `sklearn` starts with a random configuration of low-dimensional embeddings, assigning each cell a random coordinate in the low-dimensional space. Next, all pairwise distances in the low-dimensional space are calculated and the ratio between the high- and low-dimensional spaces are calculated. Finally, all points are adjusted in the low-dimensional space such that the discrepancy in distances is minimized. This process is repeated until convergence. One of the nice parts about SMACOF is that it is guaranteed to monotonically decrease stress. However, there is no guarantee that the final solution is not a local minimum. To address this, `sklearn` runs MDS several times and takes the best solution.

#### MDS on toy datasets

![MDS on toy data](img/toy_data.MDS.png)

**Swiss roll** - MDS performs similarly to PCA in embedding the Swiss roll because it preserves all pairwise distances. Although MDS is non-linear, the cost function is linear across distance scales, so it will never "unroll" the swiss roll. That behavior would require the ability to understand that it doesn't matter how close the center and outside of the swiss roll are placed in the low-dimensional space.

**Three blobs** - Again, for similar reasons that PCA performs well here, MDS does a good job of embedding the three clusters. The ratio of blue:orange and blue:green blobs is preserved, and the blobs look like blobs.

**Uneven circle** - This representation is fairly accurate, although the uneven spacing is lost.

**Tree** - MDS does very poorly on the tree, perhaps because the data is very high dimensional (500 dimensions) and noisy. This is likely an initialization issue: the probability that the global minimum is reached from a set of random initializations is low. A possible solution here would be to initialize MDS using PCA or some other embedding.
