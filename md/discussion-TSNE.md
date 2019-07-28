### t-SNE (t-distributed Stochastic Neighbor Embeding)

#### How does t-SNE work?

One can find many articles with a formal treatment of the t-SNE algorithm. The intuition behind the method is that t-SNE finds a transformation of the data from the feature space to 2 or 3 dimensions such that the distances between points within any given *local neighborhood* are preseved. In t-SNE, a local neighborhood is defined using a kernel funciton defined by the Student's t distribution and distance preservation is quantified using KL-divergence between the distance matrix in high dimensions and low dimensions. The exact embedding is calculated using stochastic gradient descent. To learn more, consult the [original paper](www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf), or any number of [online explanations](http://mlexplained.com/2018/09/14/paper-dissected-visualizing-data-using-t-sne-explained/).

There are a couple of key points to consider with t-SNE. First, the embedding is calculated using only local neighborhood distances. This means that t-SNE will not preserve any information that cannot be learned using overlapping neighborhoods. As a result, the reduced dimensions learned by t-SNE *do not represent a metric space.* This means that pairwise distances computed in t-SNE space do not respect, among other things, the triangle inequality. Data points with equal values in the feature space [may not be positioned in the same point in the t-SNE space](https://datascience.stackexchange.com/questions/19025/t-sne-why-equal-data-values-are-visually-not-close). Furthermore, in a t-SNE embedding, [the distances between clusters meaningless](https://stats.stackexchange.com/questions/263539/clustering-on-the-output-of-t-sne). For a full treatment of the limitations of t-SNE, consult [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/).

#### t-SNE on toy data cases

![t-SNE on toy data](img/toy_data.TSNE.png)

**Swiss roll** - Here, t-SNE does terribly. This poor performance has been noted before ([see this comparison of SNE methods on the swiss roll](https://jlmelville.github.io/smallvis/swisssne.html)), and is even addressed on [the t-SNE FAQ](https://lvdmaaten.github.io/tsne/#faq). Laurens van der Maaten's rationale for this is that the swiss roll has low instrinsic dimensionality and so does not suffer from the so-called "crowding problem" discussed in the [original paper](www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf).  He also writes, "who cares about Swiss rolls when you can embed complex real-world data nicely?" However, we note that it is difficult to define "nicely" for real-world data.

**Three blobs** - t-SNE performs okay on this dataset. The blobs look like blobs, and the green blob is slightly closer to the blue than orange blob. However, the ratio between green/blue vs green/orange blobs is nowhere near 1:5.

**Uneven circle** - Here, the result from t-SNE is partly faithful to the original data, but the noise added to the original data is being represented as width whereas the data was originally generated one-dimensionally along the circle. Also, the uneven spacing of the points on the circle is lost in this embedding.

**Tree** - Because t-SNE only preserves local neighborhood distances, it has a tendency to shatter data distributed as a tree. This occurs because there are proportionally few cells at the points where two or more branches meet. As such, the penalty for failing to preserve the distances at these regions in the data is small. Additionally, t-SNE performs poorly with outliers, which are added by Splatter at a 5% frequency by default. Here you can see that a few of these outliers are place very far from the main data, skewing the visualization.
