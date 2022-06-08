################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np
import random

from classify import Classify
from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod
from numpy import concatenate as concat
from numpy import column_stack as cols
from utils.data import bootstrap_data, filter_data, load_data_from_csv, rescale
from utils.test import test_randomly
from utils.utils import from_1_of_k, to_1_of_k

from gauss_bayes_classify import GaussBayesClassify
from knn_classify import KNNClassify
from linear_classify import LinearClassify
from logistic_classify import LogisticClassify
from tree_classify import TreeClassify


################################################################################
################################################################################
################################################################################


################################################################################
## ADABOOST ####################################################################
################################################################################


class AdaBoost(Classify):

	def __init__(self, base, n, X=None, Y=None, *args, **kargs):
		"""
		Constructor for AdaBoost (adaptive boosting) object for classification. 

		Parameters
		----------
		base : class that inherits from Classify
			Base classifier type used for learning.
		n : int
			Number of base classifiers to instantiate, train, and use for
			predictions.
		X : numpy array (optional)
			N x M numpy array of training data.
		Y : numpy array (optional)
			1 x N numpy array of class labels.
		args: mixed (optional)
			Additional args needed for training.
		kargs: mixed (optional)
			Additional keywords args needed for training.
		"""
		self.n_ensemble = 0
		self.ensemble = []
		self.alpha = []
		self.base = base

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			self.train(base, n, X, Y, *args, **kargs)


	def __str__(self):
		to_return = 'AdaBoost; Type: {}'.format(str(self.base))
		return to_return


	def __repr__(self):
		to_return = 'AdaBoost; Type: {}'.format(str(self.base))
		return to_return


## CORE METHODS ################################################################


	def train(self, base, n, X, Y, *args, **kargs):
		"""
		Learn n new instances of a base class. Refer to constructor docstring
		for argument descriptions. 
		"""
		if set(np.unique(Y)) != {1, -1}:
			raise ValueError('GradBoost.train: labels must be in -1/+1 format')

		self.base = base

		N,D = twod(X).shape
		n_init = len(self.ensemble)	

		wts = np.ones((N,1))
		wts /= wts.sum()											# start with unweighted data
		for i in range(len(self.ensemble)):
			y_hat = self.ensemble[i].predict(X)						# if we already have data...
			wts = wts * np.exp(-self.alpha[-1] * 2 * (Y == y_hat))	# ...figure out the weights for the training data
			wts /= wts.sum()

		for i in range(n_init, n_init + n):
			self.ensemble.append(base(X, Y, *args, **kargs))		# train ensemble member
			y_hat = self.ensemble[-1].predict(X)					# make this entry's prediction
			err = wts.T.dot(y_hat != twod(Y).T).flatten()[0]		# calculate this entry's data weighted error
			# calculate this entry's weight
			self.alpha.append(.5 * np.log((1 - err) / (err + 1e-64)))
			wts = wts * np.exp(-self.alpha[-1] * 2 * (Y == y_hat))	# calculate the new resulting data weights
			wts /= wts.sum()										# normalize them to sum to 1
			self.n_ensemble += 1 


	def predict(self, X):
		"""
		Predict on X. See constructor docstring for argument description.
		"""
		# get ensemble response and threshold it
		y_hat = (self.predict_soft(X) > 0).astype(int)
		y_hat[y_hat == 0] = -1
		return y_hat


	def predict_soft(self, X):
		"""
		Compute the linear response of the ensemble by combining all learners.
		See constructor docstring for argument description.
		"""
		N,M = twod(X).shape
		
		pred = np.zeros((N,1))
		for l in self:									# for each learner...
			# ...make a prediction (using -1/+1 convention)
			pred = cols((pred, 2 * l.predict(X) - 1))
		pred = pred[:,1:]
		return pred * self.alpha						# return un-thresholded value


	def exp_loss(self, X, Y):
		"""
		Calculate the exponential loss function.  See constructor docstring for
		argument descriptions.
		"""
		y_hat = self.predict_soft(X)
		e = (2 * Y - 1).dot(y_hat) / len(Y)				# use mean rather than sum
		e = np.exp(-e)				
		return e


## MUTATORS ####################################################################


	def clear_ensemble(self):
		"""
		Clear ensemble of learners.
		"""
		self.n_ensemble = 0
		self.ensemble = []


## INSPECTORS ##################################################################


	def component(self, i):
		"""
		Access the component learner at index i.  See __getitem__.

		Parameters
		----------
		i : int
			Index of component learner to be accessed.

		Returns
		-------
		classifier object
			Trained component classifier at index i.
		"""
		return self[i]


	def __iter__(self):
		"""
		This method allows iteration over AdaBoost objects. Iteration 
		over AdaBoost objects allows sequential access to the learners in 
		self.ensemble.
		"""
		for learner in self.ensemble:
			yield learner


	def __getitem__(self, i):
		"""
		Indexing the AdaBoost object at index 'i' returns the learner at index 
		'i' in self.ensemble.

		Parameters
		----------
		i : int
			The index that specifies the learner to be returned.

		Returns
		-------
		self.ensemble[i] : object that inherits from Classify
			Learner in ensemble at index i.
		"""
		if type(i) is not int:
			raise TypeError('AdaBoost.__getitem__: argument \'i\' must be of type int')

		return self.ensemble[i]


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	data,predictions = load_data_from_csv('../data/binary.csv', -1, float)
	data,predictions = arr(data), arr(predictions)
	data,predictions = bootstrap_data(data, predictions, 150)

	bases = [GaussBayesClassify, KNNClassify, TreeClassify]

	def test(trd, trc, ted, tec):
		X1,Y1 = bootstrap_data(trd, trc, round(len(trd) * .5))
		X2,Y2 = bootstrap_data(trd, trc, round(len(trd) * .5))

		base = bases[np.random.randint(len(bases))]

		ab = AdaBoost(base, 10, X1, Y1)

		print('ab', '\n')
		print(ab)

		err = ab.err(ted, tec)
		print('err =', err)

		print('ab.n_ensemble')
		print(ab.n_ensemble)

		ab.train(base, 7, X2, Y2)

		print('ab.n_ensemble')
		print(ab.n_ensemble)

		err = ab.err(ted, tec)
		print('err =', err)

		return err

	avg_err = test_randomly(data, predictions, 0.8, test=test, end=5)

	print('avg_err')
	print(avg_err)


################################################################################
################################################################################
################################################################################
