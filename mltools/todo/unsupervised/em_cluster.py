################################################################################
## IMPORTS #####################################################################
################################################################################


import matplotlib.pyplot as plt
import numpy as np
import random

from collections import namedtuple
from kmeans import k_init
from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod
from utils import data
from utils.utils import from_1_of_k, optional_return


################################################################################
################################################################################
################################################################################


################################################################################
## EXPECTATION-MAXIMIZATION ####################################################
################################################################################


def em_cluster(X, K, init='random', max_iter=100, tol=1e-6, do_plot=False, to_return=[1,0,0,0]):
	"""
	Perform Gaussian mixture EM (expectation-maximization) clustering on data X.

	Parameters
	----------
	X : numpy array
		N x M array containing data to be clustered.
	K : int
		Number of clusters.
	init : str or array (optional)
		Either a K x N numpy array containing initial clusters, or
		one of the following strings that specifies a cluster init
		method: 'random' (K random data points (uniformly) as clusters)
				'farthest' (choose cluster 1 uniformly, then the point farthest
					 from all cluster so far, etc.)
				'k++' (choose cluster 1 
		uniformly, then points randomly proportional to distance from
		current clusters).
	max_iter : int (optional)
		Maximum number of iterations.
	tol : scalar (optional)
		Stopping tolerance.
	do_plot : bool (optional)
		Plot if do_plot == True.
	to_return : [bool] (optional)
		Array of bools that specifies which values to return. The bool
		at to_return[0] indicates whether z should be returned; the bool
		at to_return[1] indicates whether T should be returned, etc.

	Returns
	-------
	z : numpy array
		1 x N numpy array of cluster assignments (int indices).
	T : {str -> numpy array} (optional)
		Gaussian component parameters:
			alpha : numpy array
			mu : numpy array
			sig : numpy array
	soft : numpy array (optional)
		Soft assignment probabilities (rounded for assign).
	ll : scalar (optional)
		Log-likelihood under the returned model.
	"""
	# init
	N,D = twod(X).shape					# get data size

	if type(init) is str:
		init = init.lower()
		if init == 'random':
			pi = np.random.permutation(N)
			mu = X[pi[0:K],:]
		elif init == 'farthest':
			mu = k_init(X, K, True)
		elif init == 'k++':
			mu = k_init(X, K, False)
		else:
			raise ValueError('em_cluster: value for "init" ( ' + init +  ') is invalid')
	else:
		mu = init

	sig = np.zeros((D,D,K))
	for c in range(K):
		sig[:,:,c] = np.eye(D)
	alpha = np.ones(K) / K
	R = np.zeros((N,K))

	iter,ll,ll_old = 1, np.inf, np.inf
	done = iter > max_iter
	C = np.log(2 * np.pi) * D / 2

	while not done:
		ll = 0
		for c in range(K):
			# compute log prob of all data under model c
			V = X - np.tile(mu[c,:], (N,1))			
			R[:,c] = -0.5 * np.sum((V.dot(np.linalg.inv(sig[:,:,c]))) * V, axis=1) - 0.5 * np.log(np.linalg.det(sig[:,:,c])) + np.log(alpha[c]) - C

		# avoid numberical issued by removing constant 1st
		mx = R.max(1)
		R -= np.tile(twod(mx).T, (1,K))
		# exponentiate and compute sum over components
		R = np.exp(R)
		nm = R.sum(1)
		# update log-likelihood of data
		ll = np.sum(np.log(nm) + mx)
		R /= np.tile(twod(nm).T, (1,K))		# normalize to give membership probabilities

		alpha = R.sum(0)					# total weight for each component
		for c in range(K):
			# weighted mean estimate
			mu[c,:] = (R[:,c] / alpha[c]).T.dot(X)
			tmp = X - np.tile(mu[c,:], (N,1))
			# weighted covar estimate
			sig[:,:,c] = tmp.T.dot(tmp * np.tile(twod(R[:,c]).T / alpha[c], (1,D))) + 1e-32 * np.eye(D)
		alpha /= N

		# stopping criteria
		done = (iter >= max_iter) or np.abs(ll - ll_old) < tol
		ll_old = ll
		iter += 1

	z = from_1_of_k(R)
	soft = R
	T = {'pi': alpha, 'mu': mu, 'sig': sig}
		
	return optional_return(to_return, twod(z).T, T, soft, ll)


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	X,Y = data.load_data_from_csv('../data/classifier-data.csv', 4, float)
	X,Y = arr(X), arr(Y)

	z,T,soft,ll = em_cluster(X, 5, to_return=[1,1,1,1])

	print('z')
	print(z)
	print('T')
	print(T)
	print('soft')
	print(soft)
	print('ll')
	print(ll)


################################################################################
################################################################################
################################################################################
