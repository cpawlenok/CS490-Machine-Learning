################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np
from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import concatenate as concat
from numpy import exp as e
from regress import Regress


################################################################################
################################################################################
################################################################################


################################################################################
## NNETREGRESS #################################################################
################################################################################


class NNetRegress(Regress):

	def __init__(self, X=None, Y=None, sizes=[], init='zeros', stepsize=.01, tolerance=1e-4, max_steps=5000, activation='logistic'):
		"""
		Constructor for NNetRegressor (neural network classifier).
		Member weights are [W1 ... WL-1] where W1 is Nh1 x N1

		Parameters
		----------
		X : numpy array 
			N x M numpy array that contains N data points with M features.
		Y : numpy array 
			1 x N numpy array that contains N values that relate to data 
			points in X.
		sizes = array of ints 
			[N1, Nh1 ... Nout] where Nout is # of outputs.
		init : str
			One of 'keep', 'zeros', or 'logistic; init method for weights'.
		stepsize : scalar
			Step size for gradient descent (descreases as 1 / iter).
		tolerance : scalar
			Tolerance for stopping criterion.
		max_steps : int
			Maximum number of steps before stopping.
		activation : string
			One of 'logistic', 'htangent', or 'custom'; init method for 
			activation functions.
		"""
		self.wts = arr([], dtype=object)
		self.activation = activation
		self.sig = None
		self.d_sig = None
		self.sig_0 = None
		self.d_sig_0 = None

		self.set_activation(activation)								# default to logistic activation
		self.init_weights(sizes, init, X, Y)						# default initialization

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			self.train(X, Y, init, stepsize, tolerance, max_steps)	# train if data available


	def __repr__(self):
		to_return = 'Multi-layer Perceptron (Neural Network) Regressor\n[{}]'.format(
			self.get_layers())
		return to_return


	def __str__(self):
		to_return = 'Multi-layer Perceptron (Neural Network) Regressor\n[{}]'.format(
			self.get_layers())
		return to_return


## CORE METHODS ################################################################


	def train(self, X, Y, init='zeros', stepsize=.01, tolerance=1e-4, max_steps=500):
		"""
		This method trains the neural network. Refer to constructor
		doc string for descriptions of arguments.
		"""
		n,d = mat(X).shape												# d = number of features; n = number of training data
		L = len(self.wts) + 1											# number of layers

		# define desired activation function and its derivative for training
		sig, d_sig, sig_0, d_sig_0 = self.sig, self.d_sig, self.sig_0, self.d_sig_0

		# outer loop of gradient descent
		iter = 1														# iteration number
		done = 0														# end of loop flag
		surr = np.zeros((1, max_steps + 1)).ravel()						# surrogate loss values
		errs = np.zeros((1, max_steps + 1)).ravel()						# error rate values

		while not done:
			step_i = stepsize / iter

			# stochastic gradient update
			for i in range(n):
				A,Z = self.__responses(self.wts, X[i,:], sig, sig_0)	# compute all layers' responses, then backdrop
				delta = (Z[L - 1] - Y[i]) * d_sig_0(Z[L - 1])			# take derivative of output layer
	
				for l in range(L - 2, 0, -1):
					grad = arr(mat(delta).T * mat(Z[l]))				# compute gradient on current layer weights
					delta = np.dot(delta, self.wts[l]) * d_sig(Z[l])	# propagate gradient downwards
					delta = delta[1:]									# discard constant feature
					self.wts[l] = self.wts[l] - step_i * grad			# take gradient step on current layer weights

			# compute current error values
			errs[iter] = self.mse(X, Y)									# surrogate (mse on output)

			# check stopping conditions
			done = iter > 1 and (abs(errs[iter] - errs[iter - 1]) < tolerance or iter >= max_steps)

			iter += 1
			wts_old = self.wts


	def predict(self, X):
		"""
		This method makes predictions on test data 'X'.

		Parameters
		----------
		X : numpy array 
			N x M numpy array; N = number of data points; M = number of 
			features.
		"""
		L = len(self.wts)
		Z = arr(concat((np.ones((mat(X).shape[0],1)), mat(X)), axis=1))					# initialize to input features + constant

		for l in range(L - 2):
			Z = arr(mat(Z) * mat(self.wts[l]).T)										# compute linear response of next layer
			Z = arr(concat((np.ones((mat(Z).shape[0],1)), mat(self.sig(Z))), axis=1))	# initialize to input features + constant

		Z = arr(mat(Z) * mat(self.wts[L - 1].T))										# compute output layer linear response
		return self.sig_0(Z)															# output layer activation function


## MUTATORS ####################################################################


	def set_activation(self, method, sig=None, d_sig=None, sig_0=None, d_sig_0=None):
		"""
		This method sets activation functions. 

		Parameters
		----------
		method : str
			One of 'logistic', 'htangent', or 'custom'.
		sig : function object
			Pass only when method == 'custom'.
		d_sig : function object
			Pass only when method == 'custom'.
		sig_0 : function object
			Pass only when method == 'custom' (output layer activation function).
		d_sig_0 : function object
			Pass only when method == 'custom' (output layer activation function).

		TODO:
			add multilogit?
		"""
		method = method.lower()
		self.activation = method

		if method == 'logistic':
			self.sig = lambda z: 2 * 1 / (1 + e(-z.astype(np.float64))) - 1
			self.d_sig = lambda z: 2 * e(-z) / np.power(1 + e(-z), 2)
			self.sig_0 = lambda z: z
			self.d_sig_0 = lambda z: 1 + 0 * z

		elif method == 'htangent':
			self.sig = lambda z: np.tanh(z)
			self.d_sig = lambda z: 1 - np.power(np.tanh(z), 2)
			self.sig_0 = lambda z: z
			self.d_sig_0 = lambda z: 1 + 0 * z

		elif method == 'custom':
			self.sig = sig
			self.d_sig = d_sig
			self.sig_0 = sig_0
			self.d_sig_0 = d_sig_0

		else:
			raise ValueError('NNetRegress.set_activation: \'' + method + '\' is not a valid argument for method')


	def set_layers(self):
		pass


	def set_weights(self):
		pass


	def init_weights(self, sizes, init='zeros', X=None, Y=None):
		"""
		This method initializes the weights of the neural network.
		Set layer sizes to S = [Ninput, N1, N2, ... Noutput] and set
		using 'fast' method ('none', 'random', 'zeros'). Refer to
		constructor doc string for argument descriptions.

		TODO:
			implement autoenc
			implement regress
		"""
		init = init.lower()

		if init == 'none':
			pass				# no init: do nothing

		elif init == 'zeros':
			self.wts = arr([np.zeros((sizes[i + 1], sizes[i] + 1)) for i in range(len(sizes) - 1)], dtype=object)

		elif init == 'random':
			self.wts = arr([.25 * np.random.randn(sizes[i + 1], sizes[i] + 1) for i in range(len(sizes) - 1)], dtype=object)

		elif init == 'autoenc':
			pass

		elif init == 'regress':
			pass

		else:
			raise ValueError('NNetRegress.init_weights: \'' + init + '\' is not a valid argument for init')


## INSPECTORS ##################################################################


	def get_layers(self):
		"""
		This method returns the layer sizes of the neural net.
		"""
		S = [mat(self.wts[l]).shape[1] for l in range(len(self.wts) - 1)]
		S.append(mat(self.wts[-1].shape[1]))
		return arr(S)


	def get_weights(self):
		return self.wts


## HELPERS #####################################################################


	def __responses(self, wts, X, sig, sig_0):
		"""
		This is a helper method that gets the linear sum from previous layer
		(A) and saturated activation responses (Z) for a data point. Used in:
			train
		"""
		L = len(wts) + 1
		A = [None for i in range(L)]
		Z = arr([None for i in range(L)], dtype=object).ravel()
		A[0] = arr([1])
		Z[0] = arr(concat((np.ones((mat(X).shape[0],1)), mat(X)), axis=1))				# compute linear combination of inputs

		for l in range(1, L - 1):
			A[l] = arr(mat(Z[l - 1]) * mat(wts[l - 1]).T)								# compute linear combination of previous layer
			Z[l] = arr(concat((np.ones((mat(X).shape[0],1)), mat(sig(A[l]))), axis=1))	# pass through activation function and add constant feature

		A[L - 1] = arr(mat(Z[L - 2]) * mat(wts[L - 2]).T)
		Z[L - 1] = sig_0(A[L - 1])														# output layer

		return (A,Z)
	

################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	np.set_printoptions(linewidth=200)

	data = [[float(val) for val in row[:-1]] for row in csv.reader(open('../data/regressor-data.csv'))]
	trd = np.asarray(data[0:40] + data[50:90] + data[100:140])
	ted = np.asarray(data[40:50] + data[90:100] + data[140:150])
	trd2 = np.asarray(data[150:180] + data[200:230] + data[250:280])
	ted2 = np.asarray(data[180:200] + data[230:250] + data[280:300])
	trd3 = np.asarray(data[300:320] + data[350:370] + data[400:420])
	ted3 = np.asarray(data[320:350] + data[370:400] + data[420:450])
	predictions = [float(row[-1].lower()) for row in csv.reader(open('../data/regressor-data.csv'))]
	trp = np.asarray(predictions[0:40] + predictions[50:90] + predictions[100:140])
	tep = np.asarray(predictions[40:50] + predictions[90:100] + predictions[140:150])
	trp2 = np.asarray(predictions[150:180] + predictions[200:230] + predictions[250:280])
	tep2 = np.asarray(predictions[180:200] + predictions[230:250] + predictions[280:300])
	trp3 = np.asarray(predictions[300:320] + predictions[350:370] + predictions[400:420])
	tep3 = np.asarray(predictions[320:350] + predictions[370:400] + predictions[420:450])

	print('nnr', '\n')
	nnr = NNetRegress(trd, trp, [4,4,4], init='random')
	print(nnr, '\n')


################################################################################
################################################################################
################################################################################
