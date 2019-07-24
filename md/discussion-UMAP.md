### UMAP (Uniform Manifold APproximation)

#### How does UMAP work?

UMAP is a recently developed dimensionality reduction algorithm that has seen a recent explosion of popularity. UMAP shares some conceptual similarites to both tSNE and force-directed layout (a popular method for visualizing graphs). The algorithm is described in [an ArXiv pre-print](https://arxiv.org/abs/1802.03426). UMAP has an excellent set of tutorials and usage examples on github, along with stellar documentation of the code. Unfortunately, we (and [others](https://www.math.upenn.edu/~jhansen/2018/05/04/UMAP/)) find that the description of the UMAP algorithm is difficult to parse. The authors use jargon like "fuzzy topological representations" and "1-simplices" where the canonical terms "graph" and "edge" would suffice. However, the basics of the algorithm are fairly straightforward.

UMAP starts by building a graph from the data using an adjustable bandwidth kernel set using approximate nearest-neighbors. Next, UMAP uses gradient descent to minimize the cross-entropy between the edge weights in high dimensions and the edge weights constructed on a graph in low dimensions. That's it!

Let's see how UMAP performs.

#### UMAP on toy data cases

![UMAP on toy data](img/toy_data.UMAP.png)


**Swiss roll** - Here we observe that UMAP does a fair job of unrolling the swiss roll. Two issues with this embedding are that one part of the roll that is broken apart and another that is split off from the main group of points. Overall, this looks far better than t-SNE or PCA.

**Three blobs** - Because UMAP shares such conceptual similarity with t-SNE, it's not surprised that it performs similarly on these blobs. The exact ratio spacing between the blue:orange and blue:green clusters is lost, and the orangle blob, which should be on the outside of the plot, is instead place in between them. 

**Uneven circle** - Here, UMAP fails to embed the circle properly and breaks it into a single lines additionally, the uneven spacing between points is lost.


**Tree** - Although UMAP aims to perserve local and global distances, you can see that the algorithm suffers from the same difficulties as t-SNE. Because the algorithm tries to preserve local distances over global distances, the algorithm has a tendency to shatter a tree at the branching points.
