################################################################################
## IMPORTS #####################################################################
################################################################################


import numpy as np
import random

from csv import reader
from data import load_data_from_csv
from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod
from numpy import column_stack as cols


################################################################################
################################################################################
################################################################################


################################################################################
## FEATURE MANIPULATION FUNCTIONS ##############################################
################################################################################


def fhash(X, K, hash=None):
	"""
	Random hash of features from data. Selects a fixed or random hash of features
	from X. 

	Parameters
	----------
	X : numpy array
		N x M numpy array containing data.
	K : int
		Number of features to select.
	hash : function object (optional)
		Hash function to use. If provided, 'hash' uses fixed hash.

	Returns
	-------
	Z : numpy array
		N x K array of hashed features of X.
	hash : hash function (optional)
		Hash function used to hash features. Only returned if 'hash' argument
		isn't provided.
	"""
	to_return = ()

	n,m = twod(X).shape

	if not hash:
		hash = lambda i: np.floor(np.random.rand(m) * K)[i]
		to_return = (hash,)

	# do the hashing
	Z = np.zeros((n,K))
	for i in range(m):
		Z[:,hash(i)] = Z[:,hash(i)] + X[:,i]

	return Z if len(to_return) == 0 else (Z,) + to_return


def fkitchensink(X, K, typ, W=None):
	"""
	Random kitchen sink features from data. Selects K random "kitchen sink"
	features of X. 

	Parameters
	----------
	X : numpy array
		N x M numpy array containing data.
	K : int
		Number of features to select.
	typ : str
		One of: 'stump', 'sigmoid', 'sinuoid', or 'linear'.
	W : numpy array (optional)
		M x K array of parameters. If provided, W uses fixed params.

	Returns
	-------
	Z : numpy array
		N x K array of features selected from X.
	W : numpy array (optional)
		M x K array of random parameters. Only returned if the argument W
		isn't provided.
	"""
	to_return = ()

	N,M = twod(X).shape
	typ = typ.lower()

	if type(W) is type(None):							# numpy complains about truth value of arrays
		if typ == 'stump':
			W = np.zeros((2,K))
			s = np.sqrt(np.var(X, axis=0))
			# random feature index 1..M
			W[0,:] = np.floor(np.random.rand(K) * M)
			W = W.astype(int)
			W[0,:] = W[0,:].astype(int)
			W[1,:] = np.random.randn(K) * s[W[0,:]]		# random threshold (w/ same variance as that feature)
		elif typ in ['sigmoid', 'sinusoid', 'linear']:
			# random direction for sigmodal ridge, random freq for sinusoids, random linear projections
			W = np.random.randn(M,K)					

		to_return = (W,)
														
	Z = np.zeros((N,K))
	
	if typ == 'stump':									# decision stump w/ random threshold
		for i in range(K):
			Z[:,i] = X[:,W[0,i]] >= W[1,i]
	elif typ == 'sigmoid':								# sigmoidal ridge w/ random direction
		Z = twod(X).dot(W)
		Z = 1 / (1 + np.exp(Z))
	elif typ == 'sinusoid':								# sinusoid w/ random frequency
		Z = np.sin(twod(X).dot(W))
	elif typ == 'linear':								# straight linear projection
		Z = twod(X).dot(W)

	return Z if len(to_return) == 0 else (Z,) + to_return


def flda(X, Y, K, T=None):
	"""
	Reduce the dimension of X to K features using (multiclass) discriminant
	analysis.

	Parameters
	----------
	X : numpy array
		N x M array of data.
	Y : numpy array
		1 x N array of labels corresponding to data in X.
	K : int
		New dimension (number of features) of X.
	T : numpy array (optional)
		The transform matrix. If this argument is provided, function uses T
		instead of computing the LDA.

	Returns
	-------
	Xlda : numpy array
	T : numpy array (optional)

	NOTE: bugs in Ihler's version
	"""
	if type(T) is not type(None):
		return np.divide(X, T)

	n,m = twod(X).shape

	c = np.unique(Y)
	nc = np.zeros(len(c))
	mu = np.zeros((len(c),n))
	sig = np.zeros((len(c),n,n))

	for i in range(len(c)):
		idx = np.where(Y == c[i])[0]
		nc[i] = len(idx)
		mu[i,:] = np.mean(X[:,idx], axis=0)
		sig[i,:,:] = np.cov(X[:,idx])

	S = (nc / n).dot(np.reshape(sig, (len(c),n * n)))
	S = np.reshape(S, (n,n))
	
	U,S,V = np.linalg.svd(X, K)					# compute SVD (Ihler uses svds here)
	Xlda = U.dot(np.sqrt(S))					# new data coefficients
	T = np.sqrt(S[0:K,0:K]).dot(twod(V).T)		# new bases for data

	return Xlda,T


def fpoly(X, degree, use_constant=True):
	"""
	Create expanded polynomial features of up to a given degree.

	Parameters
	----------
	X : numpy array
		N x M array of data.
	degree : int
		The degree.
	use_constant : bool (optional)
		If True (default), include a constant feature.

	Returns
	-------
	Xext : numpy array
		
	TODO: finish
	"""
	n,m = twod(X).shape

	if (m + 1)**(degree + 1) > 1e7:
		err_string = 'fpoly: {}**{} = too many potential output features'.format(
			m + 1, degree + 1
		)
		raise ValueError(err_string)

	if m == 1:												# faster shortcut for scalar data
		p = arr(range(0, degree + 1))
		answer = np.power(np.tile(X, (1, len(p))), np.tile(p, (n,1)))

		return answer if use_constant else answer[:,1:]

	K = 0
	for i in range((m + 1)**(degree + 1)):
		pass


def fpoly_mono(X, degree, use_constant=True):
	"""
	Create polynomial features of each individual feature (no cross products).

	Parameters
	----------
	X : numpy array
		N x M array of data.
	degree : int
		The degree.
	use_constant : bool (optional)
		If True (default), include a constant feature.

	Returns
	-------
	Xext : numpy array
		N x M * degree (+ 1 if use_constant) array of polynomial
		features from X.
		
	TODO: test more
	"""
	m,n = twod(X).shape

	if use_constant:
		Xext = np.zeros((m,n * degree + 1))
		Xext[:,0] = 1
		k = 1
	else:
		Xext = np.zeros((m,n * degree))
		k = 0

	for p in range(degree):
		for j in range(n):
			Xext[:,k] = np.power(X[:,j], p + 1)
			k += 1

	return Xext


def fpoly_pair(X, degree, use_constant=True):
	"""
	Create polynomial features of each individual feature (too many cross 
	products).

	Parameters
	----------
	X : numpy array
		N x M array of data.
	degree : int
		The degree.
	use_constant : bool (optional)
		If True (default), include a constant feature.

	Returns
	-------
	Xext : numpy array

	TODO: test more
	"""
	m,n = twod(X).shape

	npoly = np.ceil((n**(degree + 1) - 1) / (n - 1))			# ceil to fix possible roundoff error
	if use_constant:
		Xext = np.zeros((m,npoly))
		Xext[:,0] = 1
		Xcur = 1
		k = 1
	else:
		Xext = np.zeros((m,npoly - 1))
		Xcur = 1
		k = 0

	# hard coded to be a shorter length
	if degree == 2:
		Xext[:,k:k + n] = X
		k += n
		Z = np.reshape(X, (m,1,n))
		X2 = np.zeros((m,1))
		for i in range(twod(Z).shape[2]):
			X2 = cols((X2,X * Z[:,:,i]))
		X2 = X2[:,1:]
		idx = np.where((twod(arr(range(1,n + 1))).T >= arr(range(1,n + 1))).T.ravel())[0]
		K = len(idx)
		Xext[:,k:k + K] = X2[:,idx]
		return Xext[:,0:k + K]

	for p in range(degree):
		
		# workaround to make up for numpy's lack of bsxfun
		if type(Xcur) is int:
			Xcur = X * Xcur
		else:
			new_Xcur = np.zeros((m,1))
			for i in range(Xcur.shape[2]):
				new_Xcur = cols((new_Xcur, X * Xcur[:,:,i]))
			Xcur = new_Xcur[:,1:]

		Xcur = Xcur.reshape((m,np.size(Xcur) / m))
		K = Xcur.shape[1]
		Xext[:,k:k + K] = Xcur
		k = k + K
		Xcur = Xcur.reshape((m,1,K))

	return Xext


def fproject(X, K, proj=None):
	"""
	Random projection of features from data. Selects a fixed or random linear
	projection of K features from X.

	Parameters
	----------
	X : numpy array
		N x M array of data.
	K : int
		Number of features in output.
	proj : numpy array (optional)
		The projection matrix. If this argument is provided, function uses proj
		instead of random matrix.

	Returns
	-------
	X : numpy array
		N x K array of projecjtion of data in X.
	proj : numpy array (optional)
		M x K numpy array that is the project matrix. Only returned if proj 
		argument isn't provided.
	"""
	n,m = twod(X).shape

	to_return = ()
	if type(proj) is type(None):
		proj = np.random.randn(m, K)
		to_return = (proj,)

	X2 = X.dot(proj)
	
	return X2 if len(to_return) == 0 else (X2,) + to_return


def fsubset(X, K, feat=None):
	"""
	Select subset of features from data. Selects a fixed or random subset
	of K features from X.

	Parameters
	----------
	X : numpy array
		N x M array of data.
	K : int
		Number of features in output.
	feat : array like (optional)
		Flat array of indices specifying which features to select.

	Returns
	-------
	X_sub : numpy array
		N x K numpy array of data.
	feat : numpy array (optional)
		1 x M array of indices of selected features. Only returned if feat
		argument isn't provided.
	"""
	n,m = twod(X).shape

	to_return = ()
	if type(feat) is type(None):
		feat = np.random.permutation(m)
		feat = feat[0:K]
		to_return = (feat,)

	X_sub = X[:,feat]
	return X_sub if len(to_return) == 0 else (X_sub,) + to_return


def fsvd(X, K, T=None):
	"""
	Reduce the dimensionality of X to K features using singular value 
	decomposition. 

	Parameters
	----------
	X : numpy array
		N x M array of data.
	K : int
		Number of desired output features.
	T : numpy array (optional)
		Transform matrix. Including T will use T instead of computing the
		SVD.

	Returns
	-------
	Xsvd : numpy array
		N x K matrix of data.
	T : numpy array (optional)
		Transform matrix. Will only be returned if T argument isn't provided.

	TODO: finish
	"""
	n,m = twod(X).shape

	if type(T) is type(None):
		U,S,V = np.linalg.svd(X, full_matrices=False)			# compute SVD (Ihler uses svds here)
		U = U[:,:K]
		S = np.diag(S[:K])
		V = V.T[:,:K]
		Xsvd = U.dot(np.sqrt(S))								# new data coefficients
		T = np.sqrt(S[0:K,0:K]).dot(twod(V).T)					# new bases for data
		return (Xsvd,T)

	Xsvd = np.divide(X, T)										# or, use given set of bases
	return Xsvd


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	np.set_printoptions(linewidth=300, threshold=10000, precision=2)

	data,classes = load_data_from_csv('../data/classifier-data.csv', 4, float)
	data,classes = arr(data), arr(classes)

	X = fpoly_pair(data, 5)
	print('X')
	print(X)
	print('X.shape')
	print(X.shape)


################################################################################
################################################################################
################################################################################
