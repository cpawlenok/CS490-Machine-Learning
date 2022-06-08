################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np

from numpy import asarray as arr
from numpy import asmatrix as mat
from regress import Regress
from utils.data import load_data_from_csv
from utils.test import test_randomly


################################################################################
################################################################################
################################################################################


################################################################################
## KNNREGRESS ##################################################################
################################################################################


class KNNRegress(Regress):

	def __init__(self, X=None, Y=None, K=1, alpha=0):
		"""
		Constructor for KNNRegressor (k-nearest-neighbor regression model).  

		Parameters
		----------
		X : numpy array
			N x M array of N training instances with M features. 
		Y : numpy array
			1 x N array that contains the values that correspond to instances 
			in X.
		K : int 
			That sets the number of neighbors to used for predictions.
		alpha : scalar 
			Weighted average coefficient (Gaussian weighting; alphs = 0 -> 
			simple average).
		"""
		self.K = K
		self.X_train = []
		self.Y_train = []
		self.alpha = alpha

		if type(X) == np.ndarray and type(Y) == np.ndarray:
			self.train(X, Y)


	def __repr__(self):
		str_rep = 'KNNRegress, K={}{}'.format(
			self.K, ', weighted (alpha=' + str(self.alpha) + ')' 
			if self.alpha else '')
		return str_rep


	def __str__(self):
		str_rep = 'KNNRegress, K={}{}'.format(
			self.K, ', weighted (alpha=' + str(self.alpha) + ')' 
			if self.alpha else '')
		return str_rep


## CORE METHODS ################################################################
			

	def train(self, X, Y):
		"""
		This method "trains" the KNNRegressor: it stores the input data.
		Refer to constructor docstring for descriptions of X and Y.
		"""
		self.X_train = np.asarray(X)
		self.Y_train = np.asarray(Y)


	def predict(self, X):
		"""
		This method makes a nearest neighbor prediction on test data X.
	
		Parameters
		----------
		X : numpy array 
			N x M numpy array that contains N data points with M features. 
		"""
		n_tr,m_tr = mat(self.X_train).shape										# get size of training data
		n_te,m_te = mat(X).shape												# get size of test data

		assert m_tr == m_te, 'KNNRegress.predict: training and prediction data must have same number of features'

		Y_te = np.tile(self.Y_train[0], (n_te, 1))								# make Y_te the same data type as Y_train
		K = min(self.K, n_tr)													# can't have more than n_tr neighbors

		for i in range(n_te):
			dist = np.sum(np.power((self.X_train - X[i]), 2), axis=1)			# compute sum of squared differences
			sorted_dist = np.sort(dist, axis=0)[:K]								# find nearest neihbors over X_train and...
			sorted_idx = np.argsort(dist, axis=0)[:K]							# ...keep nearest K data points
			wts = np.exp(-self.alpha * sorted_dist)
			Y_te[i] = mat(wts) * mat(self.Y_train[sorted_idx]).T / np.sum(wts)	# weighted average

		return Y_te


## MUTATORS ####################################################################


	def set_alpha(self, alpha):
		"""
		Set weight parameter.  
		
		Parameters
		----------
		alpha : scalar (int or float)
		"""
		if type(alpha) not in [int, float]:
			raise TypeError('KNNRegress.set_alpha: alpha must be of type int or float')
		self.alpha = alpha

	
	def set_K(self, K):
		"""
		Set K. 

		Parameters
		----------
		K : int
		"""
		if type(K) not in [int, float]:
			raise TypeError('KNNRegress.set_K: K must be of type int or float')
		self.K = int(K)


## INSPECTORS ##################################################################


	def get_K(self):
		return self.K


## HELPERS #####################################################################


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

## RANDOM TESTING ##############################################################

	data,predictions = load_data_from_csv('../data/regressor-data.csv', -1, float)
	data,predictions = arr(data), arr(predictions)

	def test(trd, trc, ted, tec):
		print('knnr', '\n')
		knnr = KNNRegress(trd, trc)
		print(knnr, '\n')
		err = knnr.mae(ted, tec)
		print(err, '\n')
		return err

	avg_err = test_randomly(data, predictions, 0.8, test)

	print('avg_err')
	print(avg_err)

## DETERMINISTIC TESTING #######################################################

#	data = [[float(val) for val in row[:-1]] for row in csv.reader(open('../data/regressor-data.csv'))]
#	trd = np.asarray(data[0:40] + data[50:90] + data[100:140])
#	ted = np.asarray(data[40:50] + data[90:100] + data[140:150])
#	trd2 = np.asarray(data[150:180] + data[200:230] + data[250:280])
#	ted2 = np.asarray(data[180:200] + data[230:250] + data[280:300])
#	trd3 = np.asarray(data[300:320] + data[350:370] + data[400:420])
#	ted3 = np.asarray(data[320:350] + data[370:400] + data[420:450])
#	predictions = [float(row[-1].lower()) for row in csv.reader(open('../data/regressor-data.csv'))]
#	trp = np.asarray(predictions[0:40] + predictions[50:90] + predictions[100:140])
#	tep = np.asarray(predictions[40:50] + predictions[90:100] + predictions[140:150])
#	trp2 = np.asarray(predictions[150:180] + predictions[200:230] + predictions[250:280])
#	tep2 = np.asarray(predictions[180:200] + predictions[230:250] + predictions[280:300])
#	trp3 = np.asarray(predictions[300:320] + predictions[350:370] + predictions[400:420])
#	tep3 = np.asarray(predictions[320:350] + predictions[370:400] + predictions[420:450])
#	
#	print('kr', '\n')
#	kr = KNNRegress(trd, trp, K=5, alpha=1.2)
#	print(kr, '\n')
#	print(kr.predict(ted), '\n')
#	print(kr.mae(ted, tep), '\n')
#	print(kr.mse(ted, tep), '\n')
#	print(kr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('kr', '\n')
#	kr = KNNRegress(trd2, trp2, K=3, alpha=1.6)
#	print(kr, '\n')
#	print(kr.predict(ted2), '\n')
#	print(kr.mae(ted2, tep2), '\n')
#	print(kr.mse(ted2, tep2), '\n')
#	print(kr.rmse(ted2, tep2), '\n')
#
#	print()
#
#	print('kr', '\n')
#	kr = KNNRegress(trd3, trp3, K=7, alpha=0.6)
#	print(kr, '\n')
#	print(kr.predict(ted3), '\n')
#	print(kr.mae(ted3, tep3), '\n')
#	print(kr.mse(ted3, tep3), '\n')
#	print(kr.rmse(ted3, tep3), '\n')
#
#	print()
#
#	print('kr', '\n')
#	kr = KNNRegress(trd, trp, K=9, alpha=3.2)
#	print(kr, '\n')
#	print(kr.predict(ted2), '\n')
#	print(kr.mae(ted2, tep2), '\n')
#	print(kr.mse(ted2, tep2), '\n')
#	print(kr.rmse(ted2, tep2), '\n')
#
#	print()
#
#	print('kr', '\n')
#	kr = KNNRegress(trd2, trp2, K=11, alpha=2.6)
#	print(kr, '\n')
#	print(kr.predict(ted), '\n')
#	print(kr.mae(ted, tep), '\n')
#	print(kr.mse(ted, tep), '\n')
#	print(kr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('kr', '\n')
#	kr = KNNRegress(trd3, trp3, K=2, alpha=1.7)
#	print(kr, '\n')
#	print(kr.predict(ted2), '\n')
#	print(kr.mae(ted2, tep2), '\n')
#	print(kr.mse(ted2, tep2), '\n')
#	print(kr.rmse(ted2, tep2), '\n')


################################################################################
################################################################################
################################################################################



