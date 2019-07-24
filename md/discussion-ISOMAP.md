### ISOMAP (ISOmetric MAPping)

#### How does ISOMAP work?

ISOMAP is conceptually similar to the classical MDS algorithm which takes an eigendecomposition of a double-centered squared Euclidean distance matrix. A drawback of the classical MDS approach is that the method is that global distances are calculated linearly in the ambient space. As we've seen above, when the data lies on a non-linear manifold, such as a swiss roll, large Euclidean distances do not correspond to manifold distances. To solve this problem, ISOMAP considers *geodesic distances* as a measure of dissimilarity between points. Geodesic distances quantify distances walking along a graph learned on the data. This is akin to taking distances along the data manifold. On the swiss roll, this means that the points farthest from eachother will be the points in the center and outside of the roll.
 
A simple approach to building a graph given is a set of data points is to connect a each point to it's *k* Nearest Neighbors. Geodesic distances along the resulting kNN graph can be calculated as the shortest path between a given pair of points. Next, the geodesic distance matrix is double-centered and eigendecomposed with the first 2 or 3 eigenvectors used for visualization. Conveniently, `sklearn` takes care of all of the graph building and shortest path calculations so that ISOMAP dimensions can be easily calculated from any dataset.

#### ISOMAP on toy data

![ISOMAP on toy data](img/toy_data.ISOMAP.png)


**Swiss roll** - As expected, ISOMAP works very well on the swiss roll. Becuase distances are taken along the data manifold, the swiss roll is effectively unrolled and we can easily identify the two meaningful axes through the data.

**Three blobs** - One of the issues with ISOMAP can be easily observed when the data exists as more than one manifold. The issue here is that the distances between the clusters is essentially undefined. The default graph learned by ISOMAP here obviously failed to connect the three blobs, so the algorithm doesn't preserve the undefined distances between them. Each of the first three eigenvectors of the dissimilarity matrix will represent the lowest frequency axis through each individual cluster. These low-frequency eigenvectors are the lines that thepoints from each blob have been projected on. If we were to look at three dimensions, we would see that the orange blob would be represented as a line orthogonal to the blue and green.

**Uneven circle** - This representation is fairly accurate, although the uneven spacing is lost because the kNN graph adjusts to varying data density along the circle.

**Tree** - One issue with ISOMAP is that it relies on accurate geodesic distances between points. Here, the kNN graph appears to have failed to capture the true geometry of the tree.
