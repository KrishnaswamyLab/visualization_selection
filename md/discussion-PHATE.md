### PHATE (Potential of Heat Affinity Transition Embedding)

#### How does PHATE work?

PHATE is a method designed by us to solve many of the issues that we described in the above visualization methods. PHATE is inspired by [diffusion maps](https://www.sciencedirect.com/science/article/pii/S1063520306000546) and was developed in collaboration with Dr. Coifman. To understand PHATE, it is useful to first describe diffusion maps. Diffusion maps aim to preserve *diffusion distances* between points. These diffusion distances approximate the rate of heat flow along a graph, and can be approximated by random walks. Dense regions of the graph have smaller diffusion distances between points than sparse regions. In diffusion maps, the Markov-normalized weight matrix $M$, called the diffusion operator, is powered to some value $t$ and then the first 2 or 3 eigenvectors of $M^t$ are used for visualization.

Although the diffusion map approach is very successful at capturing non-linear data geometries, it often fails to preserve global structure in only 2 or 3 dimensions and requires more dimensions as the dataset becomes more complex. If a dataset is distributed on a tree, diffusion maps often require one dimension per branch to preserve global distances. As such, diffusion maps is an excellent dimensionality reduction algorithm, but it is not so useful for visualization.

To solve this problem of capturing global structure in 2 or 3 dimensions, PHATE aims to preserve *potential distances* between points. The diffusion distances used in diffusion maps consider the euclidean distance between rows of $M^t$. This means that the large fold-change differences in long range distances (*i.e.* transition probabilities of 0.01 vs 0.03) are basically ignored in the face of similar fold-change differences in among local neighborhoods (*i.e.* transition probabilities of 0.1 vs 0.3). To ensure that PHATE is sensitive to such global structure, PHATE first applies a negative log transform prior to calculating euclidean distances. These potential distances are then embedded in 2 or 3 dimensions using the distance-preserving MDS algorithm described above.

#### PHATE on toy data

![PHATE on toy data](img/toy_data.PHATE.png)

**Swiss roll** - The PHATE visualization of the swiss roll looks almost as good as the one produced by ISOMAP. Here we observe that the data is indeed a sheet, but in preserving some of these global potential distances in 2 dimensions, we see that the roll is not completely flattened.

**Three blobs** - PHATE on the three blobs preserves the cluster-like nature of the data and projects each cluster onto a single axis of variation within each cluster. The green blob's axis would likely be seen in visualizing in 3 dimensions. Unfortunately, the relative positions of the blue:orange and blue:green blob is close and this visualization makes it seem that the blue and orange blob are equidistant to the green. This is likely due to the graph being sparsely connected between the blue and orange blob but largely disconnected from the green blob.

**Uneven circle** - PHATE accurately embeds the circle in two dimensions and preserves the uneven spacing of the points around the circle.

**Tree** - PHATE accurately captures the structure of the tree in two dimensions and entirely denoising the branches to thin lines.
