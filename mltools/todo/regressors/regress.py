################################################################################
## IMPORTS #####################################################################
################################################################################


import math
import numpy as np
from numpy import asmatrix as mat


################################################################################
################################################################################
################################################################################


################################################################################
## REGRESS #####################################################################
################################################################################


class Regress:

	def __init__(self):
		"""
		Constructor for base class for several different regression learners. 
		This class implements methods that generalize to different regressors.
		"""
		pass


	def __call__(self, *args, **kwargs):
		"""
		This method provides syntatic sugar for training and prediction.
		
		To predict: regressor(X) == regressor.predict(X); first arg 
		  should be a numpy array, other arguments can be keyword 
		  args as necessary
		To train: regressor(X, Y, **kwargs) == regressor.train(X, Y, **kwargs);
		  first and second ars should be numpy arrays, other
		  arguments can be keyword args as necessary

		Args:
			This method takes any number of args or keyword args, the first
			two being numpy arrays
		"""
		if len(args) == 1 and type(args[0]) is np.ndarray:
			return self.predict(*args)

		elif len(args) == 2 and type(args[0]) is np.ndarray and type(args[1]) is np.ndarray:
			self.train(*args, **kwargs)

		else:
			raise ValueError('Regress.__call__: invalid arguments')


	def mae(self, X, Y):
		"""
		This method computes the mean absolute error of predictor object
		on test data X and Y.

		Args:
			X = N x M numpy array that contains N data points with M features
			Y = 1 x N numpy array that contains values that correspond to the
			  data points in X
		"""
		return np.mean(np.sum(abs(Y - mat(self.predict(X)).T), axis=0))


	def mse(self, X, Y):
		"""
		This method computes the mean squared error of predictor object
		on test data X and Y. Refer to mae doc string for description 
		of args.
		"""
		return np.mean(np.sum(np.power(Y - mat(self.predict(X)).T, 2), axis=0))


	def rmse(self, X, Y):
		"""
		This method computes the root mean squared error of predictor object
		on test data X and Y. Refer to mae doc string for description 
		of args.
		"""
		return np.sqrt(self.mse(X, Y))


## HELPERS #####################################################################


################################################################################
################################################################################
################################################################################
