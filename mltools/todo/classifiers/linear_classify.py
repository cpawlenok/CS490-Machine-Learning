################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np

from classify import Classify


################################################################################
################################################################################
################################################################################


################################################################################
## LINEARCLASSIFY ##############################################################
################################################################################


class LinearClassify(Classify):

	def __init__(self, X=None, Y=None, method='perceptron', reg=0.1, stepsize=0.01, tolerance=1e-4, 
		max_steps=5000, init='zeros', train_soft=False):
		"""
		Constructor for LinearClassifier (perceptron). 

		Parameters
		----------
		X : N x M numpy array 
			N = number data points; M = number of features. 
		Y : 1 x N numpy array 
			Contains class labels that relate the the data points in X.  
		method : str
			One of 'perceptron', 'logisticmse', or 'hinge'.
		reg : scalar (int or float) or 1 x M + 1 numpy array (where first entry is constant feature)
			L2 regularization penalty parameter.
		stepsize : float 
			Step size for gradient descent.
		tolerance : float
			Tolerance for stopping criterion.
		max_steps : int 
			Maximum number of steps to take before stopping.
		init : str
			One of 'keep' (keep current weights), 'zeros' (all-zeros), 'randn' (random), 
			or 'linreg' (linear regression solution).
		train_soft : bool 
			Determines the training method for the classifier. 

		TODO:
			Fix auc
			Fix roc
			Test train_soft
		"""
		self.wts = []
		self.classes = []
		self.method = method.lower()

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			if train_soft:
				self.train_soft(X, Y, stepsize, tolerance, max_steps, init.lower())
			else:
				self.train(X, Y, method, reg, stepsize, tolerance, max_steps, init.lower())


	def __repr__(self):
		return 'Linear Binary Classifier; {} features\n{}'.format(
			len(self.wts), self.wts)


	def __str__(self):
		return 'Linear Binary Classifier; {} features\n{}'.format(
			len(self.wts), self.wts)


## CORE METHODS ################################################################


	def train(self, X, Y, method='perceptron', reg=0.1, stepsize=0.01, tolerance=1e-4, 
		max_steps=5000, init='zeros'):
		"""
		This method trains the linear classifer. Refer to constructor 
		doc string for argument descriptions.
		"""
		n,d = np.asmatrix(X).shape
		X_train = np.concatenate((np.ones((n,1)), X), axis=1)

		if np.isscalar(reg):
			reg = reg + np.asarray([0 for i in range(d + 1)])
			reg[0] = 0

		self.classes = list(np.unique(Y))
		assert len(self.classes) == 2, 'Linear classifier should have two classes'
		Y = np.asarray([-1 if Y[i] == self.classes[0] else 1 for i in range(len(Y))])

		old_wts = self.__init_wts(init, X_train, Y, n, d)

		self.__gradient_descent(max_steps, stepsize, method, n, X, X_train, Y, reg, tolerance)


	def train_soft(self, X, Y, stepsize=.01, tolerance=1e-4, max_steps=5000, init='zeros'):
		"""
		This method trains the linear classifier. Soft training assumes data is already
		in canonical form, but possibly real valued. Refer to constructor doc string 
		for description of arguments.
		"""
		n,d = np.asmatrix(X).shape
		X_train = np.concatenate((np.ones((n,1)), X), axis=1)
		self.classes = [0, 1]

		old_wts = self.__init_wts(init, X_train, Y, n, d)

		iter, done = 1, 0
		mse, err = np.zeros((1, max_steps + 1)).ravel(), np.zeros((1, max_steps + 1)).ravel()

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
		This method makes predictions on test data X.

		Parameters
		----------
		X : N x M numpy array 
			N = number data points; M = number of features. 
		"""
		indices = np.sign(self.wts[0] + np.dot(X, self.wts[1:])) / 2 + .5
		Y_te = np.asarray([self.classes[int(i)] for i in indices])
		return np.asarray(np.asmatrix(Y_te).T)


	def predict_soft(self, X):
		"""
		This method makes "soft" predictions on test data X (real valued #s).  
		Refer to predict doc string for a description of X.
		"""
		Y_hat = self.__logistic(X) * 2 - 1
		return np.asarray(np.asmatrix(Y_hat).T)


## MUTATORS ####################################################################


	def set_classes(self, classes):
		"""
		Set classes of the classifier. 

		Parameters
		----------
		classes : list
		"""
		if type(classes) is not list or len(classes) == 0:
			raise TypeError('LinearClassify.set_classes: classes should be a list with a length of at least 1')
		self.classes = classes


	def set_weights(self, wts):
		"""
		Set weights of the classifier. 

		Parameters
		----------
		wts : list or a numpy array
		"""
		if type(wts) not in [list, np.ndarray] or len(wts) == 0:
			raise TypeError('LinearClassify.set_weights: classes should be a list/numpy array with a length of at least 1')
		self.wts = np.asarray(wts)


## INSPECTORS ##################################################################


	def get_classes(self):
		return self.classes


	def get_weights(self):
		return self.wts


## HELPERS #####################################################################


	def __init_wts(self, init, X_train, Y, n, d):
		"""
		Helper method that intializes self.wts. Used in: 
			train
			train_soft
		"""
		if init == 'keep':
			self.wts = self.wts if len(self.wts) == d + 1 else np.asarray([0 for i in range(d + 1)])
		elif init == 'zeros':
			self.wts = np.asarray([0 for i in range(d + 1)])
		elif init == 'randn':
			self.wts = np.random.randn(1, d + 1).ravel()
		elif init == 'linreg':
			indices = np.random.permutation(n).ravel()
			indices = indices[0:min(max(4 * d, 100), n)]
			self.wts = np.dot(np.dot(Y[indices], X_train[indices,:]), np.linalg.inv(
				np.dot(X_train[indices,:].T, X_train[indices,:]) + .1 * np.eye(d + 1)))
		else:
			raise ValueError('LinearClassify.__init_wts: ' + str(init) + ' is not a valid option for the init parameter')

		return 0 * self.wts + np.inf


	def __take_step(self, response, X_train, Y, reg, y_hat_i, i, step_i):
		"""
		Helper method that takes a step in gradient descent. Used in:
			__gradient_descent
		"""
		if self.method == 'logisticmse':
			sig = 1 / (1 + np.exp(-response))
			grad = (2 * sig - 1 - Y[i]) * sig * (1 - sig) * X_train[i,:] + reg * self.wts
		elif self.method == 'hinge':
			grad = (Y[i] * response < 1) * (-Y[i] * X_train[i,:]) + reg * self.wts
		elif self.method == 'perceptron':
			grad = (y_hat_i - Y[i]) * X_train[i,:]
		else:
			raise ValueError('LinearClassify.__take_step: ' + str(self.method) + ' is not a valid option for the method parameter')

		self.wts = self.wts - step_i * grad


	def __update_surrogate_loss(self, X, X_train, Y, it, errs, surr):
		"""
		Helper method that updates the surrogate loss matrix. Used in:
			__gradient_descent
		"""
		if self.method == 'logisticmse':
			surr[it] = np.mean(np.power(Y - self.__logistic(X), 2))
			return (surr,errs)
		elif self.method == 'hinge':
			surr[it] = np.mean(np.maximum(0, 1 - Y * (np.dot(X_train, self.wts))))
			return (surr,errs)
		elif self.method == 'perceptron':
			surr[it] = np.inf if errs[it] != 0 else 0
			return (surr, errs)
		else:
			raise ValueError('LinearClassify.__update: ' + str(method) + ' is not a valid option for the method parameter')


	def __gradient_descent(self, max_steps, stepsize, method, n, X, X_train, Y, reg, tolerance):
		"""
		Helper method that implements stochastic gradient descent. Used in:
			train
		"""
		it, done = 1, 0
		surr, errs = np.zeros((1, max_steps + 1)).ravel(), np.zeros((1, max_steps + 1)).ravel()

		while not done:
			step_i = stepsize if method.lower() == 'perceptron' else stepsize / it
			for i in range(n):
				response = np.dot(X_train[i,:], self.wts.T)
				y_hat_i = np.sign(response)
				self.__take_step(response, X_train, Y, reg, y_hat_i, i, step_i)
			errs[it] = np.mean(Y != np.sign(np.dot(X_train, self.wts)))
			surr,errs = self.__update_surrogate_loss(X, X_train, Y, it, errs, surr)
			done = it > 1 and ((abs(surr[it] - surr[it - 1]) < tolerance) or (it > max_steps - 1))
			it += 1


	def __logistic(self, X):
		"""
		Helper method that calculates the predicted value for Y based on X. Used in:
			train_soft
			predict_soft
			__update_surrogate_loss
		"""
		n,d = np.asmatrix(X).shape
		X = np.concatenate((np.ones((n,1)), X), axis=1) if n > 1 else np.append([1], X)
		f = np.dot(X, self.wts)
		value = 1 / (1 + np.exp(-f))
		return value


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

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

	print('lc1', '\n')
	lc1 = LinearClassify(btrd, btrc, method='perceptron', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='zeros')
	print(lc1, '\n')
	print(lc1.predict(bted), '\n')
	print(lc1.predict_soft(bted), '\n')
	print(lc1.auc(bted, btec), '\n')
	print(lc1.err(bted, btec), '\n')
	print(lc1.roc(bted, btec), '\n')
	print(lc1.confusion(bted, btec), '\n')

	print()

	print('lc2', '\n')
	lc2 = LinearClassify(btrd2, btrc2, method='perceptron', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='zeros')
	print(lc2, '\n')
	print(lc2.predict(bted2), '\n')
	print(lc2.predict_soft(bted2), '\n')
	print(lc2.auc(bted2, btec2), '\n')
	print(lc2.err(bted2, btec2), '\n')
	print(lc2.roc(bted2, btec2), '\n')
	print(lc2.confusion(bted2, btec2), '\n')

	print()

	print('lc3', '\n')
	lc3 = LinearClassify(btrd, btrc, method='perceptron', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='linreg')
	print(lc3, '\n')
	print(lc3.predict(bted), '\n')
	print(lc3.predict_soft(bted), '\n')
	print(lc3.auc(bted, btec), '\n')
	print(lc3.err(bted, btec), '\n')
	print(lc3.roc(bted, btec), '\n')
	print(lc3.confusion(bted, btec), '\n')

	print()

	print('lc4', '\n')
	lc4 = LinearClassify(btrd2, btrc2, method='perceptron', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='linreg')
	print(lc4, '\n')
	print(lc4.predict(bted2), '\n')
	print(lc4.predict_soft(bted2), '\n')
	print(lc4.auc(bted2, btec2), '\n')
	print(lc4.err(bted2, btec2), '\n')
	print(lc4.roc(bted2, btec2), '\n')
	print(lc4.confusion(bted2, btec2), '\n')

	print()

	print('lc5', '\n')
	lc5 = LinearClassify(btrd, btrc, method='logisticmse', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='zeros')
	print(lc5, '\n')
	print(lc5.predict(bted), '\n')
	print(lc5.predict_soft(bted), '\n')
	print(lc5.auc(bted, btec), '\n')
	print(lc5.err(bted, btec), '\n')
	print(lc5.roc(bted, btec), '\n')
	print(lc5.confusion(bted, btec), '\n')

	print()

	print('lc6', '\n')
	lc6 = LinearClassify(btrd2, btrc2, method='logisticmse', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='zeros')
	print(lc6, '\n')
	print(lc6.predict(bted2), '\n')
	print(lc6.predict_soft(bted2), '\n')
	print(lc6.auc(bted2, btec2), '\n')
	print(lc6.err(bted2, btec2), '\n')
	print(lc6.roc(bted2, btec2), '\n')
	print(lc6.confusion(bted2, btec2), '\n')

	print()

	print('lc7', '\n')
	lc7 = LinearClassify(btrd, btrc, method='logisticmse', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='linreg')
	print(lc7, '\n')
	print(lc7.predict(bted), '\n')
	print(lc7.predict_soft(bted), '\n')
	print(lc7.auc(bted, btec), '\n')
	print(lc7.err(bted, btec), '\n')
	print(lc7.roc(bted, btec), '\n')
	print(lc7.confusion(bted, btec), '\n')

	print()

	print('lc8', '\n')
	lc8 = LinearClassify(btrd2, btrc2, method='logisticmse', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='linreg')
	print(lc8, '\n')
	print(lc8.predict(bted2), '\n')
	print(lc8.predict_soft(bted2), '\n')
	print(lc8.auc(bted2, btec2), '\n')
	print(lc8.err(bted2, btec2), '\n')
	print(lc8.roc(bted2, btec2), '\n')
	print(lc8.confusion(bted2, btec2), '\n')

	print()

	print('lc9', '\n')
	lc9 = LinearClassify(btrd, btrc, method='hinge', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='zeros')
	print(lc9, '\n')
	print(lc9.predict(bted), '\n')
	print(lc9.predict_soft(bted), '\n')
	print(lc9.auc(bted, btec), '\n')
	print(lc9.err(bted, btec), '\n')
	print(lc9.roc(bted, btec), '\n')
	print(lc9.confusion(bted, btec), '\n')

	print()

	print('lc10', '\n')
	lc10 = LinearClassify(btrd2, btrc2, method='hinge', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='zeros')
	print(lc10, '\n')
	print(lc10.predict(bted2), '\n')
	print(lc10.predict_soft(bted2), '\n')
	print(lc10.auc(bted2, btec2), '\n')
	print(lc10.err(bted2, btec2), '\n')
	print(lc10.roc(bted2, btec2), '\n')
	print(lc10.confusion(bted2, btec2), '\n')

	print()

	print('lc11', '\n')
	lc11 = LinearClassify(btrd, btrc, method='hinge', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='linreg')
	print(lc11.predict(bted), '\n')
	print(lc11.predict_soft(bted), '\n')
	print(lc11.auc(bted, btec), '\n')
	print(lc11.err(bted, btec), '\n')
	print(lc11.roc(bted, btec), '\n')
	print(lc11.confusion(bted, btec), '\n')

	print()

	print('lc12', '\n')
	lc12 = LinearClassify(btrd2, btrc2, method='hinge', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='linreg')
	print(lc12.predict(bted2), '\n')
	print(lc12.predict_soft(bted2), '\n')
	print(lc12.auc(bted2, btec2), '\n')
	print(lc12.err(bted2, btec2), '\n')
	print(lc12.roc(bted2, btec2), '\n')
	print(lc12.confusion(bted2, btec2), '\n')

	print()

	print('lc13', '\n')
	lc13 = LinearClassify(bted, btec, method='hinge', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='linreg')
	print(lc13.predict(btrd), '\n')
	print(lc13.predict_soft(btrd), '\n')
	print(lc13.auc(btrd, btrc), '\n')
	print(lc13.err(btrd, btrc), '\n')
	print(lc13.roc(btrd, btrc), '\n')
	print(lc13.confusion(btrd, btrc), '\n')

	print()

	print('lc14', '\n')
	lc14 = LinearClassify(bted2, btec2, method='hinge', reg=.1, stepsize=.01, tolerance=1e-4, max_steps=5000, init='linreg')
	print(lc14.predict(btrd2), '\n')
	print(lc14.predict_soft(btrd2), '\n')
	print(lc14.auc(btrd2, btrc2), '\n')
	print(lc14.err(btrd2, btrc2), '\n')
	print(lc14.roc(btrd2, btrc2), '\n')
	print(lc14.confusion(btrd2, btrc2), '\n')


################################################################################
################################################################################
################################################################################
