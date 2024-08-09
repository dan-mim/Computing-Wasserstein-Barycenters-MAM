# MAM
The Method of Averaged Marginals (MAM) to compute the Wasserstein Barycenter problem

This repository corresponds to the code that has been created for article *Computing Wasserstein Barycenters via Operator Splitting: the Method of Averaged Marginals* published in journal [SIMODS](https://www.siam.org/publications/siam-journals/siam-journal-on-mathematics-of-data-science/).

## Abstract
The Wasserstein barycenter (WB) is an important tool for summarizing sets of probability measures. It finds applications in applied probability, clustering, image processing, etc. When the measures' supports are finite, computing a \rev{(balanced)} WB can be done by solving a linear optimization problem whose dimensions generally exceed standard solvers' capabilities. 
\rev{In the more general setting where measures have different total masses, we propose a convex nonsmooth optimization formulation for the so-called unbalanced WB problem. Due to their colossal dimensions, we introduce a decomposition scheme based on the Douglas-Rachford splitting method that can be applied to both balanced and unbalanced WB problem variants.}
Our algorithm, which has the interesting interpretation of being built upon averaging marginals, operates a series of simple (and exact) projections that can be parallelized and even randomized, making it suitable for large-scale datasets. Numerical comparisons against state-of-the-art methods on several data sets from the literature illustrate the method's performance.
