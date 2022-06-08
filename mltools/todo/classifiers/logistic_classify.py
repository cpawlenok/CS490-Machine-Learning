################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np

from classify import Classify
from numpy import asarray as arr
from numpy import asmatrix as mat
from utils.data import filter_data, load_data_from_csv
from utils.test import test_randomly


################################################################################
################################################################################
################################################################################


################################################################################
## LOGISTICCLASSIFY ############################################################
################################################################################


class LogisticClassify(Classify):

	def __init__(self, X=None, Y=None, init='zeros', stop_iter=1000, stop_tol=-1, reg=0.0, 
		stepsize=1, train_soft=False):
		"""
		Constructor for LogisticClassifier (logistic classifier; linear classifier with
		saturated output).

		Parameters
		----------
		X : N x M numpy array 
			N = number of data points; M = number of features.
		Y : 1 x N numpy array 
			Class labels that relate to the data points in X.
		init : str
			Initialization method for the weights. One of 'zeros', 
			'random', 'regress', or 'bayes'.
		stop_iter : int 
			Maximum number of iterations through data before stoping.
		stop_tol : float 
			Tolerance for the stopping criterion.
		reg : float 
			L2 regularization value.
		stepsize : scalar (int or float) 
			Step size for gradient descent.
		"""
		self.wts = []
		self.classes = []

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			if train_soft:
				self.train_soft(X, Y, init.lower(), stop_iter, stop_tol, stepsize)
			else:
				self.train(X, Y, init.lower(), stop_iter, stop_tol, reg, stepsize)


	def __repr__(self):
		to_return = 'Logistic Classifier; {} classes, {} features\n{}'.format(
			len(self.classes), len(np.asmatrix(self.wts).T) - 1, self.wts)
		return to_return


	def __str__(self):
		to_return = 'Logistic Classifier; {} classes, {} features\n{}'.format(
			len(self.classes), len(np.asmatrix(self.wts).T) - 1, self.wts)
		return to_return


## CORE METHODS ################################################################


	def train(self, X, Y, init='zeros', stop_iter=1000, stop_tol=-1, reg=0.0, 
		stepsize=1):
		"""
		This method trains the logistic classifier. See constructor doc
		string for argument descriptions.
		"""
		self.classes = list(np.unique(Y))

		n,d = np.asmatrix(X).shape
		X_train = np.concatenate((np.ones((n,1)), X), axis=1)

		Y_copy = np.asarray(Y)
		Y = self.to_index(Y_copy)
		C = len(self.classes)

		self.wts = self.__init_weights(X, Y, init) if len(self.wts) == 0 else self.wts
		assert np.asmatrix(self.wts).shape == (C - 1, d + 1), 'Weights are not sized correctly for these data'

		wts_old = self.__gradient_descent(X, X_train, Y, Y_copy, stop_iter, stepsize, reg, stop_tol, n)


	def train_soft(self, X, Y, init, stop_iter, stop_tol, stepsize):
		"""
		This method trains the logistic classifier. Soft training assumes data is already
		in canonical form, but possibly real valued. Refer to constructor doc string 
		for description of arguments.
		"""
		n,d = np.asmatrix(X).shape
		X_train = np.concatenate((np.ones((n,1)), X), axis=1)
		self.classes = [0, 1]

		old_wts = self.__init_wts(init, X_train, Y, n, d)

		iter, done = 1, 0
		mse, err = np.zeros((1, stop_iter + 1)).ravel(), np.zeros((1, stop_iter + 1)).ravel()

		while not done:
			Y_hat = self.__logistic(X)
			mse[iter] = np.mean(np.power(Y - Y_hat, 2))
			err[iter] = np.mean(Y != (Y_hat > .5))

			for i in range(n):
				Y_hat_i = self.__logistic(X[i,:])
				grad = (Y_hat_i - Y[i]) * Y_hat_i * (1 - Y_hat_i) * X_train[i,:]
				self.wts = self.wts - stepsize / iter * grad

			if iter > 1:
				done = (abs(mse[iter] - mse[iter - 1]) < tolerance) or iter > max_steps
			iter += 1


	def predict(self, X):
		"""
		This method makes predictions on test data X. Refer to
		constructor doc string for description of X.
		"""
		pred = np.argmax(self.__logistic(X), axis=1)
		pred = np.asarray([[self.classes[i[0]]] for i in pred])
		return pred


	def predict_soft(self, X):
		"""
		This method performs "soft" prediction on X (predicts real
		valued numbers). Refer to constructor doc string for description
		of X.
		"""
		return self.__logistic(X)


## MUTATORS ####################################################################


	def set_classes(self, classes):
		"""
		Set classes of the classifier. 

		Parameters
		----------
		classes : list
		"""
		if type(classes) is not list or len(classes) == 0:
			raise TypeError('LogisticClassify.set_classes: classes should be a list with a length of at least 1')
		self.classes = classes


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


	def get_classes(self):
		return self.classes


	def get_weights(self):
		return self.wts


## HELPERS #####################################################################


	def __gradient_descent(self, X, X_train, Y, Y_copy, stop_iter, stepsize, reg, stop_tol, n):
		"""
		This is a helper method that performs stochastic gradient descent. Used in:
			train
		"""
		iter, done = 1, 0
		j_sur, j_01 = np.zeros((1,stop_iter + 1)).ravel(), np.zeros((1,stop_iter + 1)).ravel()

		while not done:
			step = stepsize / iter
			j_sur[iter] = self.__nll(X, Y) + reg * np.sum(np.power(self.wts[:], 2))
			j_01[iter] = self.err(X, Y_copy)

			for i in range(n):
				z = np.concatenate(([0], np.asarray(np.dot(X_train[i,:], self.wts.T)).ravel()))
				s = np.exp(z - np.max(z))
				grad = np.asmatrix(s).T / np.sum(s) * X_train[i,:]
				grad[Y[i] - 1,:] = grad[Y[i] - 1,:] - X_train[i,:]
				grad = grad[1:,:]
				grad = grad + reg * self.wts
				self.wts = self.wts - step * grad

			done = iter >= stop_iter or (iter > 1 and (abs(j_sur[iter] - j_sur[iter - 1]) < stop_tol))

			wts_old = self.wts
			iter += 1

		return self.wts


	def __init_weights(self, X, Y, init='zeros'):
		"""
		This method is a helper that initializes classifier weights in one of
		four ways: zeros (all zeros), random (random, Gaussian coefficients),
		regress (simple one-versus-all linear regression on a subsample),
		or bayes (simple Gaussian Bayes model). Refer to constructor doc
		string for further description of arguments. Used in:
			train
		"""
		n,d = np.asmatrix(X).shape
		C = len(list(np.unique(Y)))

		if init == 'zeros':	
			wts = np.zeros((C - 1, d + 1))
		elif init == 'random':
			wts = np.random.randn(C - 1, d + 1)
		elif init == 'regress':
			wts = self.__init_regress(X, Y, n, d, C)
		elif init == 'bayes':
			wts = self.__init_bayes(X, Y, n, d, C)
		else:
			raise ValueError('LogisticClassify.__init_weights: ' + str(init) + 
				' is not a valid option for init')

		return wts


	def __init_bayes(self, X, Y, n, d, C):
		"""
		This method initializes weights using a Gaussian Bayes with 
		equal covariances. Used in:
			__init_weights
		"""
		wts = np.zeros((C,d + 1))
		mu = np.zeros((C,d))
		sig = .1 * np.eye(d)

		for c in range(1, C + 1):
			mu[c - 1,:] = np.mean(X[Y == c,:], axis=0)
			sig = sig + np.dot(np.mean(Y == c), np.cov(X[Y == c,:], rowvar=False))

		sig = np.linalg.inv(sig)
		wts[:,1:] = np.dot(mu, sig)
		wts[:,0] = -.5 * np.sum(mu * wts[:,1:], axis=1)
		wts = wts - wts[0,:]
		wts = wts[1:,:]
	
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


	def __logistic(self, X):
		"""
		This is a helper method that evaluates the (multi-)logistic function
		for weights self.wts and data X. Used in:
			predict_soft
		"""
		n,d = np.asmatrix(X).shape
		X = np.concatenate((np.ones((n,1)), X), axis=1)
		z = np.dot(X, np.asmatrix(self.wts).T)
		z = np.concatenate((np.zeros((n,1)), z), axis=1)
		max_z_col = np.max(z, axis=1)
		z = z - max_z_col
		s = np.exp(z)
		s = s / np.sum(s, axis=1)
		return s


	def __nll(self, X, Y):
		"""
		This is a helper method that computes logistic negative
		log-likelihood loss. Used in:
			train
		"""
		Y = self.to_1_of_k(Y, list(np.unique(Y)))
		n,d = np.asmatrix(Y).shape
		y_hat = self.predict_soft(X)
		int_Y = np.zeros((n, 1))

		for i in range(n):
			for j in range(d):
				if Y[i,j]:
					int_Y[i] = y_hat[i,j]

		return -np.mean(np.log(int_Y))


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

## RANDOM TESTING ##############################################################

	data,classes = load_data_from_csv('../data/classifier-data.csv', 4, float)
	data,classes = arr(data), arr(classes)

	bd1,bc1 = filter_data(data, classes, lambda x,y: y == 2)
	bd1,bc1 = np.array(bd1), np.array(bc1)

	def test(trd, trc, ted, tec):
		print('lc', '\n')
		lc = LogisticClassify(trd, trc)
		print(lc, '\n')
	#	print(lc.predict(ted), '\n')
	#	print(lc.predict_soft(ted), '\n')
	#	print(lc.confusion(ted, tec), '\n')
	#	print(lc.auc(ted, tec), '\n')
	#	print(lc.roc(ted, tec), '\n')
		err = lc.err(ted, tec)
		print(err, '\n')
		return err

	avg_err = test_randomly(data, classes, 0.8, test)

	print('avg_err')
	print(avg_err)

## DETERMINISTIC TESTING #######################################################
#
#	data = [[float(val) for val in row[:-1]] for row in csv.reader(open('../data/classifier-data.csv'))]
#	trd = np.asarray(data[0:40] + data[50:90] + data[100:140])
#	ted = np.asarray(data[40:50] + data[90:100] + data[140:150])
#	classes = [float(row[-1].lower()) for row in csv.reader(open('../data/classifier-data.csv'))]
#	trc = np.asarray(classes[0:40] + classes[50:90] + classes[100:140])
#	tec = np.asarray(classes[40:50] + classes[90:100] + classes[140:150])
#
#	btrd = trd[0:80,:]
#	bted = ted[0:20,:]
#	btrc = trc[0:80]
#	btec = tec[0:20]
#
#	btrd2 = trd[40:120,:]
#	bted2 = ted[10:30,:]
#	btrc2 = trc[40:120]
#	btec2 = tec[10:30]
#
#	print('lc', '\n')
#	lc = LogisticClassify(trd, trc)
#	print(lc, '\n')
#	print(lc.predict(ted), '\n')
#	#print(lc.predict_soft(ted), '\n')
#	print(lc.confusion(ted, tec), '\n')
#	print(lc.err(ted, tec), '\n')
#
#	print()
#
#	print('lc2', '\n')
#	lc2 = LogisticClassify(trd, trc, 'regress')
#	print(lc2, '\n')
#	print(lc2.predict(ted), '\n')
#	#print(lc2.predict_soft(ted), '\n')
#	print(lc2.confusion(ted, tec), '\n')
#	print(lc2.err(ted, tec), '\n')
#
#	print()
#
#	print('lc3', '\n')
#	lc3 = LogisticClassify(trd, trc, 'bayes')
#	print(lc3, '\n')
#	print(lc3.predict(ted), '\n')
#	#print(lc3.predict_soft(ted), '\n')
#	print(lc3.confusion(ted, tec), '\n')
#	print(lc3.err(ted, tec), '\n')
#
#	print()
#
#	print('ilc', '\n')
#	ilc = LogisticClassify(ted, tec)
#	print(ilc, '\n')
#	print(ilc.predict(trd), '\n')
#	#print(ilc.predict_soft(trd), '\n')
#	print(ilc.confusion(trd, trc), '\n')
#	print(ilc.err(trd, trc), '\n')
#
#	print()
#
#	print('ilc2', '\n')
#	ilc2 = LogisticClassify(ted, tec, 'regress')
#	print(ilc2, '\n')
#	print(ilc2.predict(trd), '\n')
#	#print(ilc2.predict_soft(trd), '\n')
#	print(ilc2.confusion(trd, trc), '\n')
#	print(ilc2.err(trd, trc), '\n')
#
#	print()
#
#	print('ilc3', '\n')
#	ilc3 = LogisticClassify(ted, tec, 'bayes')
#	print(ilc3, '\n')
#	print(ilc3.predict(trd), '\n')
#	#print(ilc3.predict_soft(trd), '\n')
#	print(ilc3.confusion(trd, trc), '\n')
#	print(ilc3.err(trd, trc), '\n')
#
#	print()
#
#	print('bc', '\n')
#	bc = LogisticClassify(btrd, btrc)
#	print(bc, '\n')
#	print(bc.predict(bted), '\n')
#	#print(bc.predict_soft(bted), '\n')
#	print(bc.auc(bted, btec), '\n')
#	print(bc.confusion(bted, btec), '\n')
#	print(bc.err(bted, btec), '\n')
#	print(bc.roc(bted, btec), '\n')
#
#	print()
#
#	print('bc2', '\n')
#	bc2 = LogisticClassify(btrd, btrc, 'regress')
#	print(bc2, '\n')
#	print(bc2.predict(bted), '\n')
#	#print(bc2.predict_soft(bted), '\n')
#	print(bc2.auc(bted, btec), '\n')
#	print(bc2.confusion(bted, btec), '\n')
#	print(bc2.err(bted, btec), '\n')
#	print(bc2.roc(bted, btec), '\n')
#
#	print()
#
#	print('bc3', '\n')
#	bc3 = LogisticClassify(btrd, btrc, 'bayes')
#	print(bc3, '\n')
#	print(bc3.predict(bted), '\n')
#	#print(bc3.predict_soft(bted), '\n')
#	print(bc3.auc(bted, btec), '\n')
#	print(bc3.confusion(bted, btec), '\n')
#	print(bc3.err(bted, btec), '\n')
#	print(bc3.roc(bted, btec), '\n')
#
#	print()
#
#	print('bc4', '\n')
#	bc4 = LogisticClassify(btrd2, btrc2)
#	print(bc4, '\n')
#	print(bc4.predict(bted2), '\n')
#	#print(bc4.predict_soft(bted2), '\n')
#	print(bc4.auc(bted2, btec2), '\n')
#	print(bc4.confusion(bted2, btec2), '\n')
#	print(bc4.err(bted2, btec2), '\n')
#	print(bc4.roc(bted2, btec2), '\n')
#
#	print()
#
#	print('bc5', '\n')
#	bc5 = LogisticClassify(btrd2, btrc2, 'regress')
#	print(bc5, '\n')
#	print(bc5.predict(bted2), '\n')
#	#print(bc5.predict_soft(bted2), '\n')
#	print(bc5.auc(bted2, btec2), '\n')
#	print(bc5.confusion(bted2, btec2), '\n')
#	print(bc5.err(bted2, btec2), '\n')
#	print(bc5.roc(bted2, btec2), '\n')
#
#	print()
#
#	print('bc6', '\n')
#	bc6 = LogisticClassify(btrd2, btrc2, 'bayes')
#	print(bc6, '\n')
#	print(bc6.predict(bted2), '\n')
#	#print(bc6.predict_soft(bted2), '\n')
#	print(bc6.auc(bted2, btec2), '\n')
#	print(bc6.confusion(bted2, btec2), '\n')
#	print(bc6.err(bted2, btec2), '\n')
#	print(bc6.roc(bted2, btec2), '\n')
#
#	print()
#
#	print('ibc', '\n')
#	ibc = LogisticClassify(bted, btec)
#	print(ibc, '\n')
#	print(ibc.predict(btrd), '\n')
#	#print(ibc.predict_soft(btrd), '\n')
#	print(ibc.auc(btrd, btrc), '\n')
#	print(ibc.confusion(btrd, btrc), '\n')
#	print(ibc.err(btrd, btrc), '\n')
#	print(ibc.roc(btrd, btrc), '\n')
#
#	print()
#
#	print('ibc2', '\n')
#	ibc2 = LogisticClassify(bted, btec, 'regress')
#	print(ibc2, '\n')
#	print(ibc2.predict(btrd), '\n')
#	#print(ibc2.predict_soft(btrd), '\n')
#	print(ibc2.auc(btrd, btrc), '\n')
#	print(ibc2.confusion(btrd, btrc), '\n')
#	print(ibc2.err(btrd, btrc), '\n')
#	print(ibc2.roc(btrd, btrc), '\n')
#
#	print()
#
#	print('ibc3', '\n')
#	ibc3 = LogisticClassify(bted, btec, 'bayes')
#	print(ibc3, '\n')
#	print(ibc3.predict(btrd), '\n')
#	#print(ibc3.predict_soft(btrd), '\n')
#	print(ibc3.auc(btrd, btrc), '\n')
#	print(ibc3.confusion(btrd, btrc), '\n')
#	print(ibc3.err(btrd, btrc), '\n')
#	print(ibc3.roc(btrd, btrc), '\n')
#
#	print()
#
#	print('ibc4', '\n')
#	ibc4 = LogisticClassify(bted2, btec2)
#	print(ibc4, '\n')
#	print(ibc4.predict(btrd2), '\n')
#	#print(ibc4.predict_soft(btrd2), '\n')
#	print(ibc4.auc(btrd2, btrc2), '\n')
#	print(ibc4.confusion(btrd2, btrc2), '\n')
#	print(ibc4.err(btrd2, btrc2), '\n')
#	print(ibc4.roc(btrd2, btrc2), '\n')
#
#	print()
#
#	print('ibc5', '\n')
#	ibc5 = LogisticClassify(bted2, btec2, 'regress')
#	print(ibc5, '\n')
#	print(ibc5.predict(btrd2), '\n')
#	#print(ibc5.predict_soft(btrd2), '\n')
#	print(ibc5.auc(btrd2, btrc2), '\n')
#	print(ibc5.confusion(btrd2, btrc2), '\n')
#	print(ibc5.err(btrd2, btrc2), '\n')
#	print(ibc5.roc(btrd2, btrc2), '\n')
#
#	print()
#
#	print('ibc6', '\n')
#	ibc6 = LogisticClassify(btrd2, btrc2, 'bayes')
#	print(ibc6, '\n')
#	print(ibc6.predict(btrd2), '\n')
#	#print(ibc6.predict_soft(btrd2), '\n')
#	print(ibc6.auc(btrd2, btrc2), '\n')
#	print(ibc6.confusion(btrd2, btrc2), '\n')
#	print(ibc6.err(btrd2, btrc2), '\n')
#	print(ibc6.roc(btrd2, btrc2), '\n')
#
#
################################################################################
################################################################################
################################################################################
