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
## KMEANS ######################################################################
################################################################################


def kmeans(X, K, init='random', max_iter=100, do_plot=False, to_return=[1, 0, 0]):
	"""
	Perform K-means clustering on data X.

	Parameters
	----------
	X : numpy array
		N x M array containing data to be clustered.
	K : int
		Number of clusters.
	init : str or array (optional)
		Either a K x N numpy array containing initial clusters, or
		one of the following strings that specifies a cluster init
		method: 'random' (K random data points (uniformly) as clusters),
		'farthest' (choose cluster 1 uniformly, then the point farthest
		from all cluster so far, etc.), or 'k++' (choose cluster 1 
		uniformly, then points randomly proportional to distance from
		current clusters).
	max_iter : int (optional)
		Maximum number of optimization iterations.
	do_plot : bool (optional)
		Plot 2D data?
	to_return : [bool] (optional)
		Array of bools that specifies which values to return. The bool
		at to_return[0] indicates whether z should be returned; the bool
		at to_return[1] indicates whether c should be returned, etc.

	Returns
	-------
	z : numpy array
		N x 1 array containing cluster numbers of data at indices in X.
	c : numpy array (optional)
		K x M array of cluster centers.
	sumd : scalar (optional)
		Sum of squared euclidean distances.
		
	TODO: test more
	"""
	n,d = twod(X).shape							# get data size

	if type(init) is str:
		init = init.lower()
		if init == 'random':
			pi = np.random.permutation(n)
			c = X[pi[0:K],:]
		elif init == 'farthest':
			c = k_init(X, K, True)
		elif init == 'k++':
			c = k_init(X, K, False)
		else:
			raise ValueError('kmeans: value for "init" ( ' + init +  ') is invalid')
	else:
		c = init

	z,c,sumd = __optimize(X, n, K, c,  max_iter)

	return optional_return(to_return, z - 1, c, sumd)


def __optimize(X, n, K, c, max_iter):
	iter = 1
	done = (iter > max_iter)
	sumd = np.inf
	sum_old = np.inf

	z = np.zeros((n,1))

	while not done:
		sumd = 0
		
		for i in range(n):
			# compute distances for each cluster center
			dists = np.sum(np.power((c - np.tile(X[i,:], (K,1))), 2), axis=1)
			val = np.min(dists, axis=0)							# assign datum i to nearest cluster
			z[i] = np.argmin(dists, axis=0) + 1
			sumd = sumd + val

		for j in range(1, K + 1):								# now update each cluster center j...
			if np.any(z == j):
				c[j - 1,:] = np.mean(X[(z == j).flatten(),:], 0)# ...to be the mean of the assigned data...
			else:
				c[j - 1,:] = X[np.floor(np.random.rand()),:]	# ...or random restart if no assigned data

		done = (iter > max_iter) or (sumd == sum_old)
		sum_old = sumd
		iter += 1

	return z,c,sumd
			

def k_init(X, K, determ):
	"""
	Distance based initialization. Randomly choose a start point, then:
	if determ == True: choose point farthest from the clusters chosen so
	far, otherwise: randomly choose new points proportionally to their
	distance.

	Parameters
	----------
	X : numpy array
		See kmeans docstring.
	K : int
		See kmeans docstring.
	determ : bool
		See description.

	Returns
	-------
	c : numpy array
		K x M array of cluster centers.
	"""
	m,n = twod(X).shape
	clusters = np.zeros((K,n))
	clusters[0,:] = X[np.floor(np.random.rand() * m),:]			# take random point as first cluster
	dist = np.sum(np.power((X - np.ones((m,1)) * clusters[0,:]), 2), axis=1)

	for i in range(1,K):
		if determ:
			j = np.argmax(dist)									# choose farthest point...
		else:
			pr = np.cumsum(dist);								# ...or choose a random point by distance
			pr = pr / pr[-1]
			j = np.where(np.random.rand() < pr)[0][0]

		clusters[i,:] = X[j,:]									# update that cluster
		# update min distances
		new_dist = np.sum(np.power((X - np.ones((m,1)) * clusters[i,:]), 2), axis=1) 
		dist = np.minimum(dist, new_dist)

	return clusters


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	X,Y = data.load_data_from_csv('../data/classifier-data.csv', 4, float)
	X,Y = arr(X), arr(Y)

	z,c,sumd = kmeans(X, 10, to_return=[1,1,1])
	print('z')
	print(z)
	print('c')
	print(c)
	print('sumd')
	print(sumd)


################################################################################
################################################################################
################################################################################
