################################################################################
## IMPORTS #####################################################################
################################################################################


import math
import numpy as np

from numpy import atleast_2d as twod


################################################################################
################################################################################
################################################################################


################################################################################
## CLASSIFY ####################################################################
################################################################################


class Classify:

	def __init__(self):
		"""
		Constructor for base class for several different classifiers. 
		This class implements methods that generalize to different classifiers.
		"""
		self.classes = []


	def __call__(self, *args, **kwargs):
		"""
		This method provides syntatic sugar for training and prediction.
		
		To predict: classifier(X) == classifier.predict(X); first arg 
	  	should be a numpy array, other arguments can be keyword 
	  	args as necessary

		To predict soft: classifier(X, soft=True) == classifier.predict_soft(X);
	  	first arg should be a numpy array and second
	  	argument should be the keyword arg 'soft=True'

		To train: classifier(X, Y, **kwargs) == classifier.train(X, Y, **kwargs);
		first and second ars should be numpy arrays, other
		arguments can be keyword args as necessary

		Parameters
		----------
		*args : mixed
			any number of additional arguments needed for prediction or training
		**kwargs : mixed
			any number of additional arguments needed for prediction or training
		soft : bool
			determines whether soft prediction will be used if available
		"""
		if len(args) == 1 and type(args[0]) is np.ndarray:
			if 'soft' in kwargs and kwargs['soft']:
				try:
					return self.predict_soft(*args)
				except AttributeError:
					raise AttributeError('Classify.__call__: soft prediction unavailable for this classifier')
			else:
				return self.predict(*args)

		elif len(args) == 2 and type(args[0]) is np.ndarray and type(args[1]) is np.ndarray:
			self.train(*args, **kwargs)

		else:
			raise ValueError('Classify.__call__: invalid arguments')


	def auc(self, X, Y):
		"""
		This method computes the area under the roc curve on the given test data.
		This method only works on binary classifiers. 

		Paramters
		---------
		X : N x M numpy array 
			N = number of data points; M = number of features. 
		Y : 1 x N numpy array 
			Array of classes that refer to the data points in X.
		"""
		if len(self.classes) > 2:
			raise ValueError('This method can only supports binary classification ')

		try:									# compute 'response' (soft binary classification score)
			soft = self.predict_soft(X)[:,1]	# p(class = 2nd)
		except (AttributeError, IndexError):	# or we can use 'hard' binary prediction if soft is unavailable
			soft = self.predict(X)

		n,d = twod(soft).shape

		if n == 1:
			soft = soft.flatten()
		else:
			soft = soft.T.flatten()

		sorted_soft = np.sort(soft)				# sort data by score value
		indices = np.argsort(soft)				
		Y = Y[indices]
		# find ties in the sorting score
		same = np.append(np.asarray(sorted_soft[0:-1] == sorted_soft[1:]), 0)

		n = len(soft)
		rnk = self.__compute_ties(n, same)		# compute tied rank values
		
		# number of true negatives and positives
		n0 = sum(Y == self.classes[0])
		n1 = sum(Y == self.classes[1])

		if n0 == 0 or n1 == 0:
			raise ValueError('Data of both class values not found')

		# compute AUC using Mann-Whitney U statistic
		result = (np.sum(rnk[Y == self.classes[1]]) - n1 * (n1 + 1) / 2) / n1 / n0
		return result


	def confusion(self, X, Y):
		"""
		This method estimates the confusion matrix (Y x Y_hat) from test data.
		Refer to auc doc string for descriptions of X and Y.
		"""
		Y_hat = self.predict(X)
		num_classes = len(self.classes)
		indices = self.to_index(Y, self.classes) + num_classes * (self.to_index(Y_hat, self.classes) - 1)
		C = np.histogram(indices, np.asarray(range(1, num_classes**2 + 2)))[0]
		C = np.reshape(C, (num_classes, num_classes))
		return np.transpose(C)


	def err(self, X, Y):
		"""
		This method computes the error rate on test data.  

		Paramters
		---------
		X : N x M numpy array 
			N = number of data points; M = number of features. 
		Y : 1 x N numpy array    # TODO: Nx1 !!!
			array of classes that refer to the data points in X.
		"""
		Y_hat = self.predict(X)
		Y_hat = np.transpose(Y_hat)
		return np.mean(Y_hat != Y)


	def predict(self, X):
		"""
		This is an abstract predict method that must exist in order to
		implement certain Classify methods.
		"""
		pass


	def roc(self, X, Y):
		"""
		This method computes the "receiver operating characteristic" curve on
		test data.  This method is only defined for binary classifiers. Refer 
		to the auc doc string for descriptions of X and Y. Method returns
		[fpr, tpr, tnr]. Plot fpr and tpr to see the ROC curve. Plot tpr and
		tnr to see the sensitivity/specificity curve.
		"""
		if len(self.classes) > 2:
			raise ValueError('This method can only supports binary classification ')

		try:									# compute 'response' (soft binary classification score)
			soft = self.predict_soft(X)[:,1]	# p(class = 2nd)
		except (AttributeError, IndexError):
			soft = self.predict(X)				# or we can use 'hard' binary prediction if soft is unavailable

		n,d = twod(soft).shape

		if n == 1:
			soft = soft.flatten()
		else:
			soft = soft.T.flatten()

		# number of true negatives and positives
		n0 = np.sum(Y == self.classes[0])
		n1 = np.sum(Y == self.classes[1])

		if n0 == 0 or n1 == 0:
			raise ValueError('Data of both class values not found')

		# sort data by score value
		sorted_soft = np.sort(soft)
		indices = np.argsort(soft)

		Y = Y[indices]

		# compute false positives and true positive rates
		tpr = np.divide(np.cumsum(Y[::-1] == self.classes[1]), n1)
		fpr = np.divide(np.cumsum(Y[::-1] == self.classes[0]), n0)
		tnr = np.divide(np.cumsum(Y == self.classes[0]), n0)[::-1]

		# find ties in the sorting score
		same = np.append(np.asarray(sorted_soft[0:-1] == sorted_soft[1:]), 0)
		tpr = np.append([0], tpr[np.logical_not(same)])
		fpr = np.append([0], fpr[np.logical_not(same)])
		tnr = np.append([1], tnr[np.logical_not(same)])
		return [tpr, fpr, tnr]


## HELPERS #####################################################################


	def __compute_ties(self, n, same):
		rnk = list(range(n))
		i = 0
		while i < n:
			if same[i]:
				start = i
				while same[i]:
					i += 1
				for j in range(start, i + 1):
					rnk[j] = float((i + start) / 2)
			i += 1
		return np.asarray([i + 1 for i in rnk])


	def to_index(self, Y, values=None):
		values = values if values else list(np.unique(Y))
		m,n = np.asmatrix(Y).shape
		Y = Y.ravel() if m > n else Y
		m,n = np.asmatrix(Y).shape
		assert m == 1, 'Y must be discrete scalar'

		Y_ext = np.asarray([0 for i in range(n)])
		for i in range(len(values)):
			Y_ext[np.nonzero(Y == self.classes[i])[0]] = i
		return np.asarray([i + 1 for i in Y_ext])


	def to_1_of_k(self, Y, C):
		n,d = np.asmatrix(Y).shape
		Y_ext = np.zeros((d,C[-1]))
		for i in range(len(Y)):
			Y_ext[i,Y[i] - 1] = 1
		return np.asarray(Y_ext, dtype=int)


################################################################################
################################################################################
################################################################################
