################################################################################
## IMPORTS #####################################################################
################################################################################


import numpy as np
import random

from csv import reader, writer
from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod
from scipy.linalg import sqrtm


################################################################################
################################################################################
################################################################################


################################################################################
## DATA MANIPULATION/CREATION  FUNCTIONS #######################################
################################################################################


def load_data_from_csv(csv_path, label_index, trans_func=lambda x: x):
	"""
	Function that loads from a CSV into main memory.

	Parameters
	----------
	csv_path : str
		Path to CSV file that contains data.
	label_indes : int
		The index in the CSV rows that contains the label
		for each data point.
	trans_func : function object
		Function that transform values in CSV, i.e.: str -> int.

	Returns
	-------
	data,labels : (list)
		Tuple that contains a list of data points (index 0) and
		a list of labels corresponding to thos data points (index 1).
	"""
	data = []
	labels = []

	with open(csv_path) as f:
		csv_data = reader(f)
	
		for row in csv_data:
			row = list(map(trans_func, row))

			labels.append(row.pop(label_index))
			data.append(row)

	return arr(data),arr(labels)


def filter_data(data, labels, filter_func):
	"""
	Function that filters data based on filter_func. Function
	iterates through data and labels and passes the values
	produced by the iterables to filter_func. If filter_func
	returns True, the values aren't included in the return
	arrays.

	Parameters
	----------
	data : array-like
		Array that contains data points.
	labels : array-like
		Array that contains labels.
	filter_func : function object
		Function that filters data/labels.

	Returns
	-------
	filtered_data,filtered_labels : (list)
		Filtered arrays.
	"""
	filtered_data,filtered_labels = [], []
	for point,label in zip(data,labels):
		if not filter_func(point,label):
			filtered_data.append(point)
			filtered_labels.append(label)

	return filtered_data,filtered_labels


def bootstrap_data(X, Y, n_boot):
	"""
	Function that resamples (bootstrap) data set: it resamples 
	data points (x_i,y_i) with replacement n_boot times.

	Parameters
	----------
	X : numpy array
		N x M numpy array that contains data points to be sampled.
	Y : numpy array
		1 x N numpy arra that contains labels that map to data 
		points in X.
	n_boot : int
		The number of samples to take.

	Returns
	-------
	(array,array)
		Tuple containing samples from X and Y.

	TODO: test more
	"""
	Y = Y.flatten()

	nx,dx = twod(X).shape
	idx = np.floor(np.random.rand(n_boot) * nx).astype(int)
	X = X[idx,:]

	ny = len(Y)
	assert ny > 0, 'bootstrap_data: Y must contain data'
	assert nx == ny, 'bootstrap_data: X and Y should have the same length'
	Y = Y[idx]

	return (X,Y)


def data_GMM(N, C, D=2, get_Z=False):
	"""
	Sample data from a Gaussian mixture model.  Draws N data x_i from a mixture
	of Gaussians, with C clusters in D dimensions.

	Parameters
	----------
	N : int
		Number of data to be drawn from a mixture of Gaussians.
	C : int
		Number of clusters.
	D : int
		Number of dimensions.
	get_Z : bool
		If True, returns a an array indicating the cluster from which each 
		data point was drawn.

	Returns
	-------
	X : numpy array
		N x D array of data.
	Z : numpy array (optional)
		1 x N array of cluster ids.

	TODO: test more
	"""
	C += 1
	pi = np.zeros(C)
	for c in range(C):
		pi[c] = gamrand(10, 0.5)
	pi = pi / np.sum(pi)
	cpi = np.cumsum(pi)

	rho = np.random.rand(D, D)
	rho = rho + twod(rho).T
	rho = rho + D * np.eye(D)
	rho = sqrtm(rho)
	
	mu = mat(np.random.randn(c, D)) * mat(rho)

	ccov = []
	for i in range(C):
		tmp = np.random.rand(D, D)
		tmp = tmp + tmp.T
		tmp = 0.5 * (tmp + D * np.eye(D))
		ccov.append(sqrtm(tmp))

	p = np.random.rand(N)
	Z = np.ones(N)

	for c in range(C - 1):
		Z[p > cpi[c]] = c
	Z = Z.astype(int)

	X = mu[Z,:]

	for c in range(C):
		X[Z == c,:] = X[Z == c,:] + mat(np.random.randn(np.sum(Z == c), D)) * mat(ccov[c])

	if get_Z:
		return (arr(X),Z)
	else:
		return arr(X)


def gamrand(alpha, lmbda):
	"""
	Gamma(alpha, lmbda) generator using the Marsaglia and Tsang 
	method (algorithm 4.33).

	Parameters
	----------
	alpha : scalar
	lambda : scalar
	
	Returns
	-------
	x : scalar

	TODO: test more
	"""
	if alpha > 1:
		d = alpha - 1 / 3
		c = 1 / np.sqrt(9 * d)
		flag = 1

		while flag:
			Z = np.random.randn()	

			if Z > -1 / c:
				V = (1 + c * Z)**3
				U = np.random.rand()
				flag = np.log(U) > (0.5 * Z**2 + d - d * V + d * np.log(V))

		return d * V / lmbda

	else:
		x = gamrand(alpha + 1, lmbda)
		return x * np.random.rand()**(1 / alpha)


def data_gauss(N0, N1=None, mu0=arr([0, 0]), mu1=arr([1, 1]), sig0=np.eye(2), sig1=np.eye(2)):
	"""
	Sample data from a Gaussian model.  	

	Parameters
	----------
	N0 : int
		Number of data to sample for class -1.
	N1 : int
		Number of data to sample for class 1.
	mu0 : numpy array
	mu1 : numpy array
	sig0 : numpy array
	sig1 : numpy array

	Returns
	-------
	X : numpy array
		Array of sampled data.
	Y : numpy array
		Array of class values that correspond to the data points in X.

	TODO: test more
	"""
	if not N1:
		N1 = N0

	d1,d2 = twod(mu0).shape[1],twod(mu1).shape[1]
	if d1 != d2 or np.any(twod(sig0).shape != arr([d1, d1])) or np.any(twod(sig1).shape != arr([d1, d1])):
		raise ValueError('data_gauss: dimensions should agree')

	X0 = np.dot(np.random.randn(N0, d1), sqrtm(sig0))
	X0 += np.ones((N0,1)) * mu0
	Y0 = -np.ones(N0)

	X1 = np.dot(np.random.randn(N1, d1), sqrtm(sig1))
	X1 += np.ones((N1,1)) * mu1
	Y1 = np.ones(N1)

	X = np.row_stack((X0,X1))
	Y = np.concatenate((Y0,Y1))

	return X,Y


def whiten(X, mu=None, sig=None):
	"""
	Function that whitens X to be zero mean, uncorrelated, and unit
	variance. For example: Xtr,m,s = whiten(Xtr); Xte = whiten(Xte, m, s)
	(whitens training data and changes test data to match)

	Parameters
	----------
	X : numpy array
	mu : numpy array
		Transform value (mean).
	sig : numpy array
		Transform value (covariance matrix).

	Returns
	-------
	to_return : numpy array or tuple
		Function has multiple return values: if mu and sig are specified, whitened
		X is returned, otherwise, whitened X and mu/sig are returned in a tuple
		(function will return always return X, and will return mu and or sig if
		one/both are unspecified).

	TODO: test more
	"""
	to_return = ()
	if type(mu) is type(None):			# because numpy complains about the truth values of arrays
		mu = np.mean(X, axis=0)
		to_return += (mu,)

	if type(sig) is type(None):			# because numpy complains about the truth values of arrays
		C = np.cov(X, rowvar=0)
		U,S,V = np.linalg.svd(C)
		sig = U * np.diag(1 / np.sqrt(np.diag(S)))
		to_return += (sig,)

	X = X - mu
	X = X.dot(sig)

	return X if len(to_return) == 0 else (X,) + to_return


def split_data(X, Y, train_fraction):
	"""
	Split data into training and test data.

	Parameters
	----------
	X : numpy array
		N x M array of data to split.
	Y : numpy arra
		1 x N array of labels that correspond to data in X.
	train_fraction : float
		Fraction of data to use for training.

	Returns
	-------
	to_return : (Xtr,Xte,Ytr,Yte) or (Xtr,Xte)
		A tuple containing the following arrays (in order): training
		data from X, testing data from X, training labels from Y
		(if Y contains data), and testing labels from Y (if Y 
		contains data).
	"""
	nx,dx = twod(X).shape
	ne = round(train_fraction * nx)

	Xtr,Xte = X[:ne,:], X[ne:,:]
	to_return = (Xtr,Xte)

	Y = arr(Y).flatten()
	ny = len(Y)

	if ny > 0:
		assert ny == nx, 'split_data: X and Y must have the same length'
		Ytr,Yte = Y[:ne], Y[ne:]
		to_return += (Ytr,Yte)

	return to_return


def shuffle_data(X, Y):
	"""
	Shuffle data in X and Y.

	Parameters
	----------
	X : numpy array
		N x M array of data to shuffle.
	Y : numpy arra
		1 x N array of labels that correspond to data in X.

	Returns
	-------
	X or (X,Y) : numpy array or tuple of arrays
		Shuffled data (only returns X and Y if Y contains data).
	
	TODO: test more
	"""
	nx,dx = twod(X).shape
	Y = arr(Y).flatten()
	ny = len(Y)

	pi = np.random.permutation(nx)
	X = X[pi,:]

	if ny > 0:
		assert ny == nx, 'shuffle_data: X and Y must have the same length'
		Y = Y[pi]
		return X,Y

	return X


def rescale(X, mu=None, scale=None):
	"""
	Shifts and scales data to be zero mean, unit variance in each dimension.

	Parameters
	----------
	X : numpy array
		N x M array that contains data to rescale.
	mu : numpy array
		1 x M array of means.
	scale : numpy array
		1 x M array of variances.

	Returns
	-------
	to_return : (X, mu, scale), (X, mu/scale), X
		If mu and/or scale aren't specified, the function returns
		the mu/and or scale used in a tuple with rescaled X. If mu 
		and scale are specified, only X is returned.

	TODO: test more
	"""
	to_return = ()
	if type(mu) is type(None):				# because numpy complains about the truth values of arrays
		mu = np.mean(X, axis=0)
		to_return += (mu,)

	if type(scale) is type(None):			# because numpy complains about the truth values of arrays
		scale = 1 / np.sqrt(np.var(X, axis=0))
		to_return += (scale,)

	X -= mu
	X *= scale

	return X if len(to_return) == 0 else (X,) + to_return


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	data,classes = load_data_from_csv('../data/classifier-data.csv', 4, float)
	data,classes = arr(data), arr(classes)

#	print('testing bootstrap_data')
#	print()
#
#	n,d = 100, 5
#	n_boot = 30
#
#	X = arr([np.random.rand(d) * 25 for i in range(n)])
#	Y = np.floor(np.random.rand(n) * 3)
#	data,classes = bootstrap_data(X, Y, n_boot)
#
#	assert len(data) == len(classes) == n_boot
#	assert d == twod(X).shape[1]
#
#	print('data')
#	print(data)
#	print('classes')
#	print(classes)
#
#	print()
#	print()
#
#	print('testing whiten')
#	print()
#
#	for i in range(1, 4):
#		print('i =', i)
#
#		Xtr,Xte,Ytr,Yte = cross_validate(data, classes, 3, i)
#		X,mu,sig = whiten(Xtr)
#
#		print('X')
#		print(X)
#		print('mu')
#		print(mu)
#		print('sig')
#		print(sig)
#
#		print()
#
#	print('testing split_data')
#	print()
#
#	Xtr,Xte,Ytr,Yte = split_data(data, classes, 0.4)
#
#	print('Xtr')
#	print(Xtr)
#	print('len(Xtr)')
#	print(len(Xtr))
#	print('Xte')
#	print(Xte)
#	print('len(Xte)')
#	print(len(Xte))
#	print('Ytr')
#	print(Ytr)
#	print('len(Ytr)')
#	print(len(Ytr))
#	print('Yte')
#	print(Yte)
#	print('len(Yte)')
#	print(len(Yte))
#
#	print('testing shuffle_data')
#	print()
#
#	X,Y = shuffle_data(data, classes)
#	print('X')
#	print(X)
#	print('Y')
#	print(Y)
#
#	print('testing rescale')
#	print()
#
#	X,mu,scale = rescale(data)
#	print('X')
#	print(X)
#	print('mu')
#	print(mu)
#	print('scale')
#	print(scale)
#
#	print('testing data_GMM')
#	print()
#
#	X = data_GMM(20, 5, D=2, get_Z=True)
#	print('X')
#	print(X)
#
#	print('testing data_gauss')
#	print()
#
#	X,Y = data_gauss(20)
#	print('X')
#	print(X)
#	print('Y')
#	print(Y)
#
	X,Y = data_GMM(1000, 2, D=4, get_Z=True)
	
	with open('../data/binary.csv', 'w') as f:
		w = writer(f)
		for x,y in zip(X,Y):
			w.writerow(list(x) + [y])


################################################################################
################################################################################
################################################################################
