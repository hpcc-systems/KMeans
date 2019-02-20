from sklearn import cluster, datasets
import numpy as np

#Load Iris dataset
iris = datasets.load_iris()
X_iris = iris.data
Y_iris = iris.target

#Define Three initial Centroids
fix_init = X_iris[[0,50,100], : ]

#Run KMeans with 3 Clusters, threshold = 0.0 and max_iterations = 30.
#Choose algorithm "full" which is the similiar implementation to ECL implementation
k_means = cluster.KMeans(n_clusters = 3, init= fix_init, n_init=1, copy_x= True, tol=0.0,algorithm= "full" , max_iter= 30)
k_means.fit(X_iris)

#The result of the converged centroids
rst = k_means.cluster_centers_
#The converged iteration
iters = k_means.n_iter_
#The labels of each observatioin
labs = k_means.labels_

#Print out above results
print(rst)
print(iters)
print(labs)