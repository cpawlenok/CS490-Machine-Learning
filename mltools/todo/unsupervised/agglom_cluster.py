################################################################################
## IMPORTS #####################################################################
################################################################################


import matplotlib.pyplot as plt
import numpy as np
import random

from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod
from utils import data
from utils.utils import optional_return


################################################################################
################################################################################
################################################################################


################################################################################
## AGGLOMERATIVE ###############################################################
################################################################################


def agglom_cluster(X, n_clust, method='means', join=None, to_return=[1,0]):
	"""
	Perform hierarchical agglomerative clustering.

	Parameters
	----------
	X : numpy array
		N x M array of Data to be clustered.
	n_clust : int
		The number of clusters into which data should be grouped.
	method : str (optional)
		str that specifies the method to use for calculating distance between
		clusters. Can be one of: 'min', 'max', 'means', or 'average'.
	join : numpy array (optional)
		N - 1 x 3 that contains a sequence of joining operations. Pass to avoid
		reclustering for new X.
	to_return : [bool] (optional)
		Array of bools that specifies which values to return. The bool
		at to_return[0] indicates whether z should be returned; the bool
		at to_return[1] indicates whether join should be returned.

	Returns
	-------
	z : numpy array
		N x 1 array of cluster assignments.
	join : numpy array (optional)
		N - 1 x 3 array that contains the sequence of joining operations 
		peformed by the clustering algorithm.
	"""
	m,n = twod(X).shape					# get data size
	D = np.zeros((m,m)) + np.inf		# store pairwise distances b/w clusters (D is an upper triangular matrix)
	z = arr(range(m))					# assignments of data
	num = np.ones(m)					# number of data in each cluster
	mu = arr(X)							# centroid of each cluster
	method = method.lower()

	if type(join) == type(None):		# if join not precomputed

		join = np.zeros((m - 1, 3))		# keep track of join sequence
		# use standard Euclidean distance
		dist = lambda a,b: np.sum(np.power(a - b, 2))
		for i in range(m):				# compute initial distances
			for j in range(i + 1, m):
				D[i][j] = dist(X[i,:], X[j,:])


		opn = np.ones(m)				# store list of clusters still in consideration
		val,k = np.min(D),np.argmin(D)	# find first join (closest cluster pair)
		
		for c in range(m - 1):
			i,j = np.unravel_index(k, D.shape)
			join[c,:] = arr([i, j, val])

			# centroid of new cluster
			mu_new = (num[i] * mu[i,:] + num[j] * mu[j,:]) / (num[i] + num[j])

			# compute new distances to cluster i
			for jj in np.where(opn)[0]:
				if jj in [i, j]:
					continue

				# sort indices because D is an upper triangluar matrix
				idxi = tuple(sorted((i,jj)))	
				idxj = tuple(sorted((j,jj)))	
					
				if method == 'min':
					D[idxi] = min(D[idxi], D[idxj])		# single linkage (min dist)
				elif method == 'max':
					D[idxi] = max(D[idxi], D[idxj])		# complete linkage (max dist)
				elif method == 'means':
					D[idxi] = dist(mu_new, mu[jj,:])	# mean linkage (dist b/w centroids)
				elif method == 'average':
					# average linkage
					D[idxi] = (num[i] * D[idxi] + num[j] * D[idxj]) / (num[i] + num[j])

			opn[j] = 0						# close cluster j (fold into i)
			num[i] = num[i] + num[j]		# update total membership in cluster i to include j
			mu[i,:] = mu_new				# update centroid list

			# remove cluster j from consideration as min
			for ii in range(m):
				if ii != j:
					# sort indices because D is an upper triangular matrix
					idx = tuple(sorted((ii,j)))	
					D[idx] = np.inf

			val,k = np.min(D), np.argmin(D)	# find next smallext pair

	# compute cluster assignments given sequence of joins
	for c in range(m - n_clust):
		z[z == join[c,1]] = join[c,0]

	uniq = np.unique(z)
	for c in range(len(uniq)):
		z[z == uniq[c]] = c

	return optional_return(to_return, twod(z).T, join)


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	np.set_printoptions(linewidth=200, precision=2)

	X,Y = data.load_data_from_csv('../data/classifier-data.csv', 4, float)
	X,Y = arr(X), arr(Y)

	z,join = agglom_cluster(X, 5, to_return=[1,1])
	print('z')
	print(z)
	print('join')
	print(join)


################################################################################
################################################################################
################################################################################
