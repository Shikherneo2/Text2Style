# Runs GMM Decomposition on a large dataset by refining the solution on batches of data.

import os
import sys
import math
import time
import random
import numpy as np

from sklearn.mixture import BayesianGaussianMixture

embedding_dir = sys.argv[1]
output_dir = sys.argv[2]

number_of_data_points_per_iter = 5000
epochs = 10
n_comps = 7

files = [os.path.join(embedding_dir, i) for i in os.listdir(embedding_dir) if i[-3:]=="npy"]
all_indices = [i for i in range(len(files))]

embeddings = []
for file in files:
	embeddings.append( np.load( file ) )

embeddings = np.array(embeddings)
val_data = embeddings[-2500:]
print("Loaded data")

gmm = BayesianGaussianMixture( n_components=n_comps, covariance_type="full", tol=1e-4, max_iter=2500, init_params="random", 
																weight_concentration_prior_type="dirichlet_distribution", weight_concentration_prior=1e+4,
																warm_start=True )

for epoch in range(epochs):
	random.shuffle(all_indices)
	iters = math.floor(len(files)/number_of_data_points_per_iter) -1
	print("Epoch: "+str(epoch+1))
	for iter in range(iters):
		start_time = time.time()
		start_index = iter*number_of_data_points_per_iter
		end_index = start_index + number_of_data_points_per_iter
		inds_for_this_iter = all_indices[start_index:end_index]
		cluster_data = embeddings[inds_for_this_iter]
		gmm.fit( cluster_data )

		end_time = time.time()
		print( "Likelihood: "+str(gmm.score( val_data ))+", Time: "+str(end_time-start_time) )

print( "Weight Concentration : " )
print( gmm.weight_concentration_ )
np.save( open( os.path.join( output_dir, "gmm_means.npy"), "wb"), gmm.means_ )
np.save( open( os.path.join( output_dir, "gmm_covs.npy"), "wb"), gmm.covariances_ )
np.save( open( os.path.join( output_dir, "gmm_weights.npy"), "wb"), gmm.weights_ )