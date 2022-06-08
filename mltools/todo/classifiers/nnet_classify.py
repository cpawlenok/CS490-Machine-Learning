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
from utils.data import bootstrap_data, load_data_from_csv, rescale, split_data
from utils.utils import from_1_of_k, to_1_of_k


################################################################################
################################################################################
################################################################################


################################################################################
## NNETCLASSIFY ################################################################
################################################################################


class NNetClassify(Classify):

	def __init__(self, X, Y, sizes, init='zeros', stepsize=.01, tolerance=1e-4, max_steps=5000,
		activation='logistic'):
		"""
		Constructor for NNetClassifier (neural net classifier).

		Parameters
		----------
		X : numpy array
			N x M array that contains N data points with M features.
		Y : numpy array
			Array taht contains class labels that correspond
		  	to the data points in X. 
		sizes : [Nin, Nh1, ... , Nout] 
			Nin is the number of features, Nout is the number of outputs, 
			which is the number of classes. Member weights are {W1, ... , WL-1},
		  	where W1 is Nh1 x Nin, etc.
		init : str 
			'none', 'zeros', or 'random'.  inits the neural net weights.
		stepsize : scalar
			The stepsize for gradient descent (decreases as 1 / iter).
		tolerance : scalar 
			Tolerance for stopping criterion.
		max_steps : int 
			The maximum number of steps before stopping. 
		activation : str 
			'logistic', 'htangent', or 'custom'. Sets the activation functions.
		"""
		self.classes = []
		self.wts = arr([], dtype=object)
		self.set_activation(activation.lower())
		self.init_weights(sizes, init.lower(), X, Y)

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			self.train(X, Y, init, stepsize, tolerance, max_steps)


	def __repr__(self):
		to_return = 'Multi-layer Perceptron (Neural Network) Classifier\n[{}]'.format(
			self.get_layers())
		return to_return


	def __str__(self):
		to_return = 'Multi-layer Perceptron (Neural Network) Classifier\n[{}]'.format(
			self.get_layers())
		return to_return


## CORE METHODS ################################################################


	def train(self, X, Y, init='zeros', stepsize=.01, tolerance=1e-4, max_steps=5000):
		"""
		This method trains the neural network. Refer to constructor
		doc string for descriptions of arguments.
		"""
		if self.wts[0].shape[1] - 1 != len(X[0]):
			raise ValueError('NNetClassify.__init__: sizes[0] must == len(X) (number of features)')

		if len(np.unique(Y)) != self.wts[-1].shape[0]:
			raise ValueError('NNetClassify.__init__: sizes[-1] must == the number of classes in Y')

		self.classes = self.classes if self.classes else np.unique(Y)

		# convert Y to 1-of-K format
		Y_tr_k = to_1_of_k(Y)

		n,d = mat(X).shape													# d = dim of data, n = number of data points
		nc = len(self.classes)												# number of classes
		L = len(self.wts) 													# get number of layers

		# define desired activation function and it's derivative (for training)
		sig,d_sig, = self.sig, self.d_sig
		sig_0,d_sig_0 = self.sig_0, self.d_sig_0

		# outer loop of stochastic gradient descent
		iter = 1															# iteration number
		done = 0															# end of loop flag

		surr = np.zeros((1, max_steps + 1)).ravel()							# surrogate loss values
		err = np.zeros((1, max_steps + 1)).ravel()							# misclassification rate values

		while not done:
			step_i = stepsize / iter										# step size evolution; classic 1/t decrease
			
			# stochastic gradient update (one pass)
			for i in range(n):
				A,Z = self.__responses(self.wts, X[i,:], sig, sig_0)		# compute all layers' responses, then backdrop
				delta = (Z[L] - Y_tr_k[i,:]) * arr(d_sig_0(Z[L]))			# take derivative of output layer

				for l in range(L - 1, -1, -1):
					grad = mat(delta).T * mat(Z[l])							# compute gradient on current layer wts
					delta = np.multiply(delta.dot(self.wts[l]), d_sig(Z[l]))# propagate gradient downards
					delta = delta[:,1:]										# discard constant feature
					self.wts[l] = self.wts[l] - step_i * grad				# take gradient step on current layer wts

			err[iter] = self.err_k(X, Y_tr_k)								# error rate (classification)
			surr[iter] = self.mse_k(X, Y_tr_k)								# surrogate (mse on output)

			print('surr[iter]')
			print(surr[iter])
			print('iter')
			print(iter)

			# check if finished
			done = (iter > 1) and (np.abs(surr[iter] - surr[iter - 1]) < tolerance) or iter >= max_steps
			iter += 1


	def predict(self, X):
		"""
		Make prediction on test data X. See constructor docstring for
		argument description.
		"""
		R = self.predict_soft(X)											# compute soft output values
		Y = R.argmax(1)														# get index of maximum response
		return self.classes[Y]												# convert to saved class values


	def predict_soft(self, X):
		"""
		Make 'soft' (non-class) prediction of nnet on test data X.
		See constructor docstring for argument description.
		"""
		L = len(self.wts)
		Z = cols((np.ones((mat(X).shape[0],1)),X))							# init input features + constant

		for l in range(L - 1):
			Z = mat(Z) * mat(self.wts[l]).T									# compute linear response of next layer
			Z = cols((np.ones((mat(Z).shape[0],1)),Z))						# activation function + constant

		Z = mat(Z) * mat(self.wts[L - 1]).T									# compute output layer linear response
		return self.sig_0(Z)												# output layer activation function


	def err_k(self, X, Y):
		"""
		Compute misclassification error. Assumes Y in 1-of-k form.
		See constructor doc string for argument descriptions.
		"""
		Y_hat = self.predict(X)
		return np.mean(Y_hat != from_1_of_k(Y))
	def log_likelihood(self, X, Y):
		"""
		Compute the emperical avg. log likelihood of 'obj' on test data (X,Y).
		See constructor doc string for argument descriptions.
		"""
		r,c = twod(Y).shape
		if r == 1 and c != 1:
			Y = twod(Y).T

		soft = self.predict_soft(X)
		return np.mean(np.sum(np.log(np.power(soft, Y, )), 1), 0)


	def mse(self, X, Y):
		"""
		Compute mean squared error of predictor 'obj' on test data (X,Y).
		See constructor doc string for argument descriptions.
		"""
		return mse_k(X, to_1_of_K(Y))


	def mse_k(self, X, Y):
		"""
		Compute mean squared error of predictor; assumes Y is
		in 1-of-k format. Refer to constructor docstring for
		argument descriptions.
		"""
		return np.power(Y - self.predict_soft(X), 2).sum(1).mean(0)


## MUTATORS ####################################################################


	def set_activation(self, method, sig=None, d_sig=None, sig_0=None, d_sig_0=None):
		"""
		This method sets the activation functions. 

		Parameters
		----------
		method : str
			'logistic' , 'htanget', or 'custom'
		sig, d_sig, sig_0, and d_sig_0 : function objects
			Optional arguments intended for use with the 
			'custom' method option. They should be functions. 
			sig_0 and d_sig_0 are the output layer activation functions.
		"""
		method = method.lower()

		if method == 'logistic':
			self.sig = lambda z: twod(1 / (1 + np.exp(-z)))
			self.d_sig = lambda z: twod(np.multiply(self.sig(z), (1 - self.sig(z))))
			self.sig_0 = self.sig
			self.d_sig_0 = self.d_sig
		elif method == 'htangent':
			self.sig = lambda z: twod(np.tanh(z))
			self.d_sig = lambda z: twod(1 - np.power(np.tanh(z), 2))
			self.sig_0 = self.sig
			self.d_sig_0 = self.d_sig
		elif method == 'custom':
			self.sig = sig
			self.d_sig = d_sig
			self.sig_0 = sig_0
			self.d_sig_0 = d_sig_0
		else:
			raise ValueError('NNetClassify.set_activation: ' + str(method) + ' is not a valid option for method')

		self.activation = method


	def set_classes(self, classes):
		"""
		Set classes of the classifier. 

		Parameters
		----------
		classes : list
		"""
		if type(classes) is not list or len(classes) == 0:
			raise TypeError('NNetClassify.set_classes: classes should be a list with a length of at least 1')
		self.classes = classes


	def set_layers(self, sizes, init='random'):
		"""
		Set layers sizes to sizes.

		Parameters
		----------
		sizes : [int]
			List containing sizes.
		init : str (optional)
			Weight initialization method.
		"""
		self.init_weights(sizes, init, None, None)


	def set_weights(self, wts):
		"""
		Set weights of the classifier.

		Parameters
		----------
		wts : list
		"""
		self.wts = wts


	def init_weights(self, sizes, init, X, Y):
		"""
		This method initializes the weights of the neural network and
		sets layer sizes to S=[Ninput, N1, N2, ... , Noutput]. Refer
		to constructor doc string for descritpions of arguments.
		"""
		init = init.lower()

		if init == 'none':
			pass
		elif init == 'zeros':
			self.wts = arr([np.zeros((sizes[i + 1],sizes[i] + 1)) for i in range(len(sizes) - 1)], dtype=object)
		elif init == 'random':
			self.wts = arr([.0025 * np.random.randn(sizes[i+1],sizes[i]+1) for i in range(len(sizes) - 1)], dtype=object)
		else:
			raise ValueError('NNetClassify.init_weights: ' + str(init) + ' is not a valid option for init')


## INSPECTORS ##################################################################


	def get_classes(self):
		return self.classes


	def get_layers(self):
		S = arr([mat(self.wts[i]).shape[1] - 1 for i in range(len(self.wts))])
		S = concat((S, [mat(self.wts[-1]).shape[0]]))
		return S


	def get_weights(self):
		return self.wts


## HELPERS #####################################################################


	def __responses(self, wts, X_in, sig, sig_0):
		"""
		Helper function that gets linear sum from previous layer (A) and
		saturated activation responses (Z) for a data point. Used in:
			train
		"""
		L = len(wts)
		constant_feat = np.ones((mat(X_in).shape[0],1)).flatten()	# constant feature
		# compute linear combination of inputs
		A = [arr([1])]
		Z = [concat((constant_feat, X_in))]

		for l in range(1, L):
			A.append(Z[l - 1].dot(wts[l - 1].T))					# compute linear combination of previous layer
			# pass through activation function and add constant feature
			Z.append(cols((np.ones((mat(A[l]).shape[0],1)),sig(A[l]))))

		A.append(arr(mat(Z[L - 1]) * mat(wts[L - 1]).T))
		Z.append(arr(sig_0(A[L])))									# output layer (saturate for classifier, not regressor)

		return A,Z


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

## RANDOM TESTING ##############################################################

	X,Y = load_data_from_csv('../data/gauss.csv', 4, float)
	X,Y = bootstrap_data(X, Y, 100000)
	# X,mu,scale = rescale(X)
	Xtr,Xte,Ytr,Yte = split_data(X, Y, .8)
	
	nc = NNetClassify(Xtr, Ytr, [4,5,5,5,5,5,5,5,4], init='random', max_steps=5000, activation='htangent')
	print(nc.get_weights())
	print(nc)
	print(nc.predict(Xte))
	print(nc.predict_soft(Xte))
	print(nc.err(Xte, Yte))

## DETERMINISTIC TESTING #######################################################

#	data = [[float(val) for val in row[:-1]] for row in csv.reader(open('../data/classifier-data.csv'))]
#	trd = np.asarray(data[0:40] + data[50:90] + data[100:140])
#	ted = np.asarray(data[40:50] + data[90:100] + data[140:150])
#	classes = [float(row[-1].lower()) for row in csv.reader(open('../data/classifier-data.csv'))]
#	trc = np.asarray(classes[0:40] + classes[50:90] + classes[100:140])
#	tec = np.asarray(classes[40:50] + classes[90:100] + classes[140:150])
#
#	trd,mu,scale = rescale(trd)
#	ted,mu,scale = rescale(ted)
#
#	print('nc')
#	nc = NNetClassify(trd, trc, [4,5,5,5,3], init='random', max_steps=5000, activation='htangent')
#	print(nc.get_weights())
#	print(nc)
#	print(nc.predict(ted))
#	print(nc.predict_soft(ted))
#	print(nc.err(ted, tec))


################################################################################
################################################################################
################################################################################
