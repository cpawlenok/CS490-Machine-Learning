################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np

from numpy import asarray as arr
from numpy import asmatrix as mat
from regress import Regress
from knn_regress import KNNRegress
from linear_regress import LinearRegress
from logistic_regress import LogisticRegress
from tree_regress import TreeRegress


################################################################################
################################################################################
################################################################################


################################################################################
## BAGGEDREGRESS ###############################################################
################################################################################


class BaggedRegress(Regress):

	def __init__(self, base=None, n=0, X=None, Y=None, **kwargs):
		"""
		Constructor for BaggedRegress. Regressor uses
		n learners of type 'base'.

		Parameters
		----------
		base : regressor object
			Batch regressor object type, i.e. 'LinearRegress'.
		n : int 
			Number of regressors to use.
		X : numpy array 
			N x M numpy array; N = number of data points, M = number of 
			features.
		Y : numpy array 
			1 x N numpy array; N = number of values relating to data points 
			in X.
		**kwargs : mixed
			Any number of additional arguments need for training
			learners of type 'base'.
		"""
		self.bag = []
		self.base = base

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			self.train(base, n, X, Y, **kwargs)
		

	def __repr__(self):
		to_return = 'Bagged Regressor; Type: {}'.format(str(self.base))
		return to_return


	def __str__(self):
		to_return = 'Bagged Regressor; Type: {}'.format(str(self.base))
		return to_return


## CORE METHODS ################################################################


	def train(self, base, n, X, Y, **kwargs):
		"""
		This method batch trains n learners of type 'base'. Refer to
		constructor doc string for argument descriptions.
		"""
		self.base = base

		m,d = np.asmatrix(X).shape						# get data size								

		for i in range(n):
			# choose a random sample with replacement...
			subset = np.asarray(np.ceil((m - 1) * np.random.rand(1, m - 1)), dtype=int).ravel()
			X_sub = X[subset,:]						
			Y_sub = Y[subset]							

			# ...and train learner #i with those data
			regressor = base(X_sub, Y_sub, **kwargs)
			self.bag.append(regressor)
	

	def predict(self, X):
		"""
		This method makes predictions on X. 

		Parameters
		----------
		X : numpy array 
			N x M numpy array; N = number of data points (not necessarily
		  	the same as in train), M = number of features.
		"""
		n,m = np.asmatrix(X).shape
		b = len(self.bag)								# have 'b' learners in the ensemble

		predictions = np.zeros((n,1))
		for i in range(b):								# make all 'b' predictions (0/1)...
			predictions = np.concatenate((predictions, self.bag[i].predict(X)), axis=1)
		predictions = predictions[:,1:]

		preds_mean = mat(np.mean(predictions, axis=1))	
		# make sure predictions return as column vector
		preds_mean = preds_mean.T if len(preds_mean) < len(preds_mean.T) else preds_mean
		return arr(preds_mean)							# ...and average the collection of predictions


## MUTATORS ####################################################################


	def set_component(self, i, base):
		"""
		This method sets the learner at index 'i' to 'base'.

		Parameters
		----------
		i : int
			The index of self.bag where 'base' will be placed.
		base : regressor object
		"""
		self[i] = base


	def __setitem__(self, i, base):
		"""
		This method provides syntactic sugar for the method
		set_component. See set_component doc string for 
		argument descriptions.
		"""
		if type(i) is not int:
			raise TypeError('BaggedRegressor.set_component: invalid value for argument \'i\'')
		if Regress not in type(base).__bases__:
			raise TypeError('BaggedRegressor.set_component: invalid value for argument \'base\'')

		try:
			self.bag[i] = base
		except IndexError:
			self.bag.append(base)


## INSPECTORS ##################################################################


	def get_component(self, i):
		return self[i]


	def __getitem__(self, i):
		"""
		This method provides syntactic sugar for the
		get_component method. Indexing the BaggedRegressor
		at index 'i' returns the learner at index 'i' in
		self.bag.

		Parameters
		----------
		i = int
			The index that specifies the learner to be returned.
		"""
		if type(i) is not int:
			raise TypeError('BaggedRegressor.get_component: argument \'i\' must be of type int')

		return self.bag[i]


	def __iter__(self):
		"""
		This method allows iteration over BaggedRegressor
		objects. Iteration over BaggedRegressor objects allows
		sequential access to the learners in self.bag.
		"""
		for learner in self.bag:
			yield learner


	def __len__(self):
		return len(self.bag)


## HELPERS #####################################################################


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

	print('knn_br', '\n')
	knn_br = BaggedRegress(KNNRegress, 20, trd, trp)
	print(knn_br, '\n')
	print(knn_br.predict(ted), '\n')
	print(knn_br.mae(ted, tep), '\n')
	print(knn_br.mse(ted, tep), '\n')
	print(knn_br.rmse(ted, tep), '\n')
	knn_br.set_component(10, LinearRegress(trd, trp))
	print(knn_br.get_component(10), '\n')
	knn_br[11] = LinearRegress(trd, trp)
	print(knn_br[11])

	print()

	print('lin_br', '\n')
	lin_br = BaggedRegress(LinearRegress, 20, trd, trp)
	print(lin_br, '\n')
	print(lin_br.predict(ted), '\n')
	print(lin_br.mae(ted, tep), '\n')
	print(lin_br.mse(ted, tep), '\n')
	print(lin_br.rmse(ted, tep), '\n')
	lin_br.set_component(10, KNNRegress(trd, trp))
	print(lin_br.get_component(10), '\n')
	lin_br[11] = KNNRegress(trd, trp)
	print(lin_br[11])

	print()

	print('log_br', '\n')
	log_br = BaggedRegress(LogisticRegress, 20, trd, trp)
	print(log_br, '\n')
	print(log_br.predict(ted), '\n')
	print(log_br.mae(ted, tep), '\n')
	print(log_br.mse(ted, tep), '\n')
	print(log_br.rmse(ted, tep), '\n')
	log_br.set_component(10, LinearRegress(trd, trp))
	print(log_br.get_component(10), '\n')
	log_br[11] = LinearRegress(trd, trp)
	print(log_br[11])

	print()

	print('tree_br', '\n')
	tree_br = BaggedRegress(TreeRegress, 20, trd, trp)
	print(tree_br, '\n')
	print(tree_br.predict(ted), '\n')
	print(tree_br.mae(ted, tep), '\n')
	print(tree_br.mse(ted, tep), '\n')
	print(tree_br.rmse(ted, tep), '\n')
	tree_br.set_component(10, LogisticRegress(trd, trp))
	print(tree_br.get_component(10), '\n')
	tree_br[11] = LogisticRegress(trd, trp)
	print(tree_br[11])

	print()


################################################################################
################################################################################
################################################################################
