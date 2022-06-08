################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np
import random

from regress import Regress
from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod
from numpy import concatenate as concat
from numpy import column_stack as cols
from utils.data import bootstrap_data, filter_data, load_data_from_csv, rescale
from utils.test import test_randomly
from utils.utils import from_1_of_k, to_1_of_k

from regressors.linear_regress import LinearRegress
from regressors.logistic_regress import LogisticRegress
from regressors.knn_regress import KNNRegress
from regressors.tree_regress import TreeRegress


################################################################################
################################################################################
################################################################################


################################################################################
## GRADBOOST ###################################################################
################################################################################


class GradBoost(Regress):

	def __init__(self, base, n, X=None, Y=None, *args, **kargs):
		"""
		Constructor for gradient boosting ensemble for regression.

		Parameters
		----------
		base : class that inherits from Regress
			Base regressor type used for learning.
		n : int
			Number of base regressors to instantiate, train, and use for
			predictions.
		X : numpy array (optional)
			N x M numpy array of training data.
		Y : numpy array (optional)
			1 x N numpy array of prediction values.
		args: mixed (optional)
			Additional args needed for training.
		kargs: mixed (optional)
			Additional keywords args needed for training.
		"""
		self.ensemble = []
		self.alpha = []
		self.const = 0
		self.n_use = 0
		
		self.base = base

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			self.train(base, n, X, Y, *args, **kargs)

	
	def __repr__(self):
		to_return = 'GradBoost; Type: {}'.format(str(self.base))
		return to_return


	def __str__(self):
		to_return = 'GradBoost; Type: {}'.format(str(self.base))
		return to_return


## CORE METHODS ################################################################


	def train(self, base, n, X, Y, *args, **kargs):
		"""
		Learn n new instances of base class. Refer to constructor docstring for
		descriptions of arguments.
		"""
		self.base = base

		N,D = twod(X).shape
		n_init = self.n_use
		step = 1

		if n_init == 0:
			# skip training, use constant predictor; set to local var
			self.const = np.mean(Y)								# (specialized to quadratic loss)

		y_hat = np.zeros(N) + self.const						# figure out current prediction value
		for i in range(n_init):									# if we already have learners...
			yi = self[i].predict(X).flatten()					# ...figure out prediction for the
			y_hat += (self.alpha[i] * yi)						# training data

		for i in range(n_init, n_init + n):
			Ri = (Y - y_hat) + 1e-64							# compute residuals (specialized to quadratic loss)
			self.ensemble.append(base(X, Ri, *args, **kargs))	# fit a model to the gradient residual
			yi = self[-1].predict(X)
			# minimize loss over alpha (specialized to quadratic loss)
			min_loss = step * np.divide((Ri.dot(yi)), (twod(yi).T.dot(yi)))
			self.alpha.append(min_loss.flatten()[0])			 
			y_hat = (twod(y_hat).T + self.alpha[-1] * yi).flatten()
			self.n_use += 1


	def predict(self, X):
		"""
		Predict on X. Refer to constructor docstring for description of X.
		"""
		N,D = twod(X).shape
		Y_te = np.zeros((N,1)) + self.const

		for i,l in enumerate(self):
			yi = l.predict(X)									# figure out current prediction value
			# if we already have learners, figure out the prediction on the training data
			Y_te = Y_te + self.alpha[i] * yi			

		return Y_te


## MUTATORS ####################################################################


	def clear_ensemble(self):
		"""
		Clear ensemble of learners.
		"""
		self.ensemble = []
		self.n_use = 0


	def set_n(self, n):
		"""
		Set the current number of predictors in the ensemble.

		Parameters
		----------
		n : int
			The new number of predictors.
		"""
		if type(n) is not int:
			raise TypeError('GradBoost.set_n: \'n\' (' + n + ') must be of type int')

		self.n_use = n


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


	def get_n(self):
		"""
		Get the current number of predictors in the ensemble.

		Returns
		-------
		self.n_use : int
			Number of predictors in the ensemble.
		"""
		return self.n_use


	def __iter__(self):
		"""
		This method allows iteration over GradBoost objects. Iteration 
		over GradBoost objects allows sequential access to the learners in 
		self.ensemble.
		"""
		for learner in self.ensemble:
			yield learner


	def __getitem__(self, i):
		"""
		Indexing the GradBoost object at index 'i' returns the learner at index 
		'i' in self.ensemble.

		Parameters
		----------
		i : int
			The index that specifies the learner to be returned.
		"""
		if type(i) is not int:
			raise TypeError('GradBoost.__getitem__: argument \'i\' must be of type int')

		return self.ensemble[i]


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	data,predictions = load_data_from_csv('../data/regressor-data.csv', -1, float)
	data,predictions = arr(data), arr(predictions)
	data,predictions = bootstrap_data(data, predictions, 1000)

	# bases = [LinearRegress, LogisticRegress, KNNRegress, TreeRegress]
	bases = [KNNRegress]

	def test(trd, trc, ted, tec):
		X1,Y1 = bootstrap_data(trd, trc, round(len(trd) * .5))
		X2,Y2 = bootstrap_data(trd, trc, round(len(trd) * .5))

		base = bases[np.random.randint(len(bases))]
		gd = GradBoost(base, 10, X1, Y1)

		print('gd', '\n')
		print(gd, '\n')

		err = gd.mse(ted, tec)
		print('gd.mse =', err, '\n')

		print('gd.predict(ted)')
		print(gd.predict(ted))

		print('gd.get_n()')
		print(gd.get_n())

		gd.train(base, 6, X2, Y2)

		print('gd.get_n()')
		print(gd.get_n())

		err = gd.mse(ted, tec)
		print('gd.mse =', err, '\n')

		print('gd.predict(ted)')
		print(gd.predict(ted))

		return err

	avg_err = test_randomly(data, predictions, 0.8, test=test, end=1)

	print('avg_err')
	print(avg_err)


################################################################################
################################################################################
###############################################################################
