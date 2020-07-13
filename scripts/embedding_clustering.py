import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold
from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import SpectralClustering

# Example use
# python embedding_clustering.py "/home/sdevgupta/mine/OpenSeq2Seq/logs_mixed_phonemes/logs_highway_net/logs/train_text2_style_dataset_60K_single_batch"

embedding_dir = sys.argv[1]
files = [os.path.join(embedding_dir, i) for i in os.listdir(embedding_dir) if i[-3:]=="npy"]
random.shuffle(files)
files = files[:3000]

X = np.array([np.load(i) for i in files])

# Y = TSNE(n_components=2, n_iter=1500, perplexity=40).fit_transform(X)
# mds = manifold.MDS(2, max_iter=1000, n_init=2)
# Y = mds.fit_transform( X )

print("Starting spectral clustering")
spec = SpectralClustering(n_clusters=7, assign_labels="discretize", random_state=0).fit(Y)
labels = spec.labels_

gg = []
colors = ["red", "blue", "green", "orange", "black", "grey", "yellow","aqua", "cyan"]
for i in range(7):
  f = []
  v = []
  for index,j in enumerate(labels):
    if j==i:
      f.append( Y[index] )
      v.append( X[index] )
  f = np.array(f)
  gg.append(v)
  plt.scatter(f[:,0],f[:,1], c = colors[i] )
plt.show()

for index,g in enumerate(gg):
  gmm = BayesianGaussianMixture( n_components=1, covariance_type="full", tol=1e-4, max_iter=1000, init_params="random", 
                                weight_concentration_prior_type="dirichlet_distribution", weight_concentration_prior=None,
                                warm_start=False )
  gmm.fit(np.array(g))

  np.save( open("gmm_means_tsne"+str(index+1)+".npy", "wb"), gmm.means_ )
  np.save( open("gmm_covs_tsne"+str(index+1)+".npy", "wb"), gmm.covariances_ )