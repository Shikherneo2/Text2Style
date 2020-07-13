import os
import sys
import time
import random
import numpy as np

from sklearn.mixture import BayesianGaussianMixture

embedding_dir = sys.argv[1]
number_of_data_points = 5000

files = [os.path.join(embedding_dir, i) for i in os.listdir(embedding_dir) if i[-3:]=="npy"]
indices = np.random.choice(len(files), number_of_data_points, replace=False)

embeddings = []
for index in indices:
	embeddings.append( np.load( files[index] ) )

print("Loaded npys")

start_time = time.time()
cluster_data = np.array(embeddings)
print(cluster_data.shape)


gmm = BayesianGaussianMixture( n_components=10, covariance_type="full", tol=1e-4, max_iter=1000, init_params="random", 
																weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=1.0/10,
																warm_start=False )

gmm.fit( cluster_data )
print(gmm.means_.shape)
print(gmm.covariances_.shape)
print(gmm.weight_concentration_)

print(gmm.lower_bound_)
print(gmm.score( cluster_data ))

end_time = time.time()

print("Time taken: ", end_time-start_time)