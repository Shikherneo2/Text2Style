import sys
import numpy as np
from scipy.stats import multivariate_normal

mean_filename = sys.argv[1]
covariance_filename = sys.argv[2]

mean = np.load( mean_filename )[3]
cov = np.load( covariance_filename )[3]

print(mean.shape)
print(cov.shape)

gaussian = multivariate_normal(mean, cov)
a  = gaussian.rvs()
print(a.shape)
np.save( open("gmm_rand1.npy", "wb"), a )
a  = gaussian.rvs()
np.save( open("gmm_rand2.npy", "wb"), a )