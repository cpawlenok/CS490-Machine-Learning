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
from numpy import newaxis as nxs
from utils.data import filter_data, load_data_from_csv
from utils.test import test_randomly


################################################################################
################################################################################
################################################################################


################################################################################
## LOGISTICMSECLASSIFY #########################################################
################################################################################


class LogisticMSEClassify(Classify):
	
	def __init__(self, X=None, Y=None, stepsize=.01, tolerance=1e-4, max_steps=5000, init='zeros'):
		"""
		Constructor for LogisticMSEClassifier (logistic classifier with MSE loss function.).

		Parameters
		----------
		X : N x M numpy array 
			N = number of data points; M = number of features.
		Y : 1 x N numpy array 
			Class labels that relate to the data points in X.
		stepsize : scalar
			Step size for gradient descent (decreases as 1/iter).
		tolerance : scalar
			Tolerance for stopping criterion.
		max_steps : int
			Max number of steps to take before training stops.
		init : str
			Initialization method; one of the following strings:
			'keep' (to keep current value), 'zeros' (init to all-zeros), 'randn' (init at random),
			and 'linreg' (init w/ small linear regression).
		"""
		self.wts = []								# linear weights on features (1st is constant)
		self.classes = arr([-1, 1])					# list of class values used in input

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			self.train(X, Y, stepsize, tolerance, max_steps, init.lower())


	def __str__(self):
		to_return = 'Logistic Binary Classifier; {} classes, {} features\n{}'.format(
			len(self.classes), len(np.asmatrix(self.wts).T) - 1, self.wts)
		return to_return

	def __repr__(self):
		to_return = 'Logistic Binary Classifier; {} classes, {} features\n{}'.format(
			len(self.classes), len(np.asmatrix(self.wts).T) - 1, self.wts)
		return to_return


## CORE METHODS ################################################################

	
	def train(self, X, Y, stepsize=.01, tolerance=1e-4, max_steps=5000, init='zeros'):
		"""
		This method trains the logistic MSE classifier. See constructor doc
		string for argument descriptions.
		"""
		n,d = mat(X).shape
		X_train = np.concatenate((np.ones((n,1)), X), axis=1)

		if np.logical_or(Y == 1, Y == -1).sum() != len(Y):			# check for correct binary labeling
			raise ValueError('LogisticMSEClassify.train: Y values must be +1/-1')

		self.wts = self.__init_weights(X, Y, init.lower())

		iter = 1													# it number
		done = 0													# end of loop flag
		surr = np.zeros((1, max_steps + 1)).ravel()					# surrogate loss values
		err = np.zeros((1, max_steps + 1)).ravel()					# misclassification rate values

		while not done:
			step_i = stepsize / iter								# step size evolution

			for i in range(n):										# stochastic gradient update (one pass)
				resp = X_train[i,:].dot(self.wts.T)					# compute linear response
				y_hat_i = np.sign(resp)								# compute prediction for X_i

				# compute gradient of loss function
				sig = 1 / (1 + np.exp(-resp))
				soft = 2 * sig - 1
				grad = (soft - Y[i]) * sig * (1 - sig) * X_train[i,:]

				# take a step down gradient
				self.wts = self.wts - step_i * grad

			# compute error values
			# misclassification rate
			err[iter] = np.mean(Y != np.sign(X_train * self.wts))											
			# logistic MSE surrogate
			surr[iter] = np.mean((Y[nxs].T - 2 / (1 + np.exp(mat(-X_train) * mat(self.wts).T)) + 1)[0] ** 2)				


			# compute stopping conditions
			done = iter > 0 and ((np.abs(surr[iter] - surr[iter - 1]) < tolerance) or iter >= max_steps)
			iter += 1												# increment iter


	def predict(self, X):
		"""
		This method makes predictions on test data X. Refer to
		constructor doc string for description of X.
		"""
		t = self.wts[:,0] + X.dot(self.wts[:,1:].T)
		t = (t >= 0) + 0
		return self.classes[t]


	def predict_soft(self, X):
		"""
		This method performs "soft" prediction on X (predicts real
		valued numbers). Refer to constructor doc string for description
		of X.
		"""
		return self.wts[:,0] + X.dot(self.wts[:,1:].T)


## MUTATORS ####################################################################


	def set_weights(self, wts):
		"""
		Set weights of the classifier. 

		Parameters
		----------
		wts : list or a numpy array
		"""
		if type(wts) not in [list, np.ndarray] or len(wts) == 0:
			raise TypeError('LogisticClassify.set_weights: classes should be a list/numpy array with a length of at least 1')
		self.wts = np.asarray(wts)


## INSPECTORS ##################################################################


	def get_weights(self):
		return self.wts


## HELPERS #####################################################################


	def __init_weights(self, X, Y, init='zeros'):
		"""
		This method is a helper that initializes classifier weights in one of
		four ways: zeros (all zeros), randn (random), linreg (small linear regression,
		or keep (no change). Refer to constructor doc string for further 
		description of arguments. Used in:
			train
		"""
		n,d = np.asmatrix(X).shape
		C = len(list(np.unique(Y)))

		if init == 'zeros':	
			wts = np.zeros((C - 1, d + 1))
		elif init == 'randn':
			wts = np.random.randn(C - 1, d + 1)
		elif init == 'linreg':
			wts = self.__init_regress(X, Y, n, d, C)
		elif init == 'keep':
			wts = np.zeros((C - 1, d + 1)) if len(self.wts) != d + 1 else self.wts
		else:
			raise ValueError('LogisticMSEClassify.__init_weights: ' + str(init) + ' is not a valid option for init')

		return wts


	def __init_regress(self, X, Y, n, d, C):
		"""
		Helper method that initializes wts using "small" linear regression. Used in:
			__init_weights
		"""
		wts = np.zeros((C, d + 1))
		indices = np.random.permutation(n)
		#indices = np.asarray(range(n))
		indices = indices[range(min(max(4 * d, 100), n))]
		X_train = np.concatenate((np.ones((len(indices),1)), X[indices,:]), axis=1)
		inv_cov = np.linalg.inv(np.dot(X_train.T, X_train) + .1 * np.eye(d + 1))

		for c in range(1, C + 1):
			wts[c - 1,:] = np.dot(np.dot((2 * (Y[indices] == c) - 1), X_train), inv_cov)

		wts = wts - wts[0,:]
		wts = wts[1:,:] / 2
		
		return wts


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

## RANDOM TESTING ##############################################################

#	data,classes = load_data_from_csv('../data/classifier-data.csv', 4, float)
#	bd1,bc1 = filter_data(data, classes, lambda x,y: y == 0)
#	bd2,bc2 = filter_data(data, classes, lambda x,y: y == 1)
#	bd3,bc3 = filter_data(data, classes, lambda x,y: y == 2)
#
#	bd1,bc1 = arr(bd1), arr(list(map(lambda x: -1 if x != 1 else 1, bc1)))
#	bd2,bc2 = arr(bd2), arr(list(map(lambda x: -1 if x != 0 else 1, bc2)))
#	bd3,bc3 = arr(bd3), arr(list(map(lambda x: -1 if x != 1 else 1, bc3)))
#
#	start = 0
#	#start = int(len(bd1) * .1)
#	end = len(bd1)
#
#	avg_err = 0
#
#	for i in range(start, end):
#		indexes = range(end)
#		train_indexes = random.sample(indexes, int(.8 * end))
#		test_indexes = list(set(indexes) - set(train_indexes))
#
#		trd,trc = bd1[train_indexes], bc1[train_indexes]
#		ted,tec = bd1[test_indexes], bc1[test_indexes]
#		print('lc', '\n')
#		lc = LogisticMSEClassify(trd, trc)
#		print(lc, '\n')
#	#	print(lc.predict(ted), '\n')
#	#	print(lc.predict_soft(ted), '\n')
#	#	print(lc.confusion(ted, tec), '\n')
#		print(lc.err(ted, tec), '\n')
#		avg_err += lc.err(ted, tec)
#
#	print(avg_err / end)

## DETERMINISTIC TESTING #######################################################

	data = [[float(val) for val in row[:-1]] for row in csv.reader(open('../data/classifier-data.csv'))]
	trd = np.asarray(data[0:40] + data[50:90] + data[100:140])
	ted = np.asarray(data[40:50] + data[90:100] + data[140:150])
	classes = [float(row[-1].lower()) for row in csv.reader(open('../data/classifier-data.csv'))]
	trc = np.asarray(classes[0:40] + classes[50:90] + classes[100:140])
	tec = np.asarray(classes[40:50] + classes[90:100] + classes[140:150])

	btrd = trd[0:80,:]
	bted = ted[0:20,:]
	btrc = trc[0:80]
	btec = tec[0:20]

	btrd2 = trd[40:120,:]
	bted2 = ted[10:30,:]
	btrc2 = trc[40:120]
	btec2 = tec[10:30]

	btrc = arr(list(map(lambda x: -1 if x != 1 else 1, btrc)))
	btec = arr(list(map(lambda x: -1 if x != 1 else 1, btec)))

	print('lc', '\n')
	lc = LogisticMSEClassify(bted, btec)
	print(lc, '\n')
	print(lc.predict(btrd), '\n')
	print(lc.predict_soft(ted), '\n')
	#print(lc.confusion(ted, tec), '\n')
	print(lc.err(btrd, btrc), '\n')

################################################################################
################################################################################
################################################################################
