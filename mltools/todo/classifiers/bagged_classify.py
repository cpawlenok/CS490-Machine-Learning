################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np

from classify import Classify
from gauss_bayes_classify import GaussBayesClassify
from knn_classify import KNNClassify
from linear_classify import LinearClassify
from logistic_classify import LogisticClassify
from numpy import asarray as arr
from tree_classify import TreeClassify
from utils.data import filter_data, load_data_from_csv
from utils.test import test_randomly


################################################################################
################################################################################
################################################################################


################################################################################
## BAGGEDCLASSIFY ##############################################################
################################################################################


class BaggedClassify(Classify):

	def __init__(self, base=None, n=0, X=None, Y=None, **kwargs):
		"""
		Constructor for BaggedClassifier. Classifier uses
		n learners of type 'base'.

		Parameters
		----------
		base : object that inherits from 'Classify'
			batch classifier object type, i.e. 'LinearClassify'
		n : int
			number of classifiers to use
		X : N x M numpy array
			N = number of data points, M = number of features
		Y : 1 x N numpy array
			N = number of class labels relating to data points in X
		**kwargs : mixed
			any number of additional arguments need for training learners of type 'base'
		"""
		self.bag = []
		self.classes = []
		self.base = base

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			self.train(base, n, X, Y, **kwargs)
		

	def __repr__(self):
		to_return = 'Bagged Classifier; Type: {}'.format(str(self.base))
		return to_return


	def __str__(self):
		to_return = 'Bagged Classifier; Type: {}'.format(str(self.base))
		return to_return


## CORE METHODS ################################################################


	def train(self, base, n, X, Y, **kwargs):
		"""
		This method batch trains n learners of type 'base'. Refer to
		constructor doc string for argument descriptions.
		"""
		self.base = base

		self.classes = list(np.unique(Y)) if len(self.classes) == 0 else self.classes
		m,d = np.asmatrix(X).shape							# get data size

		for i in range(n):
			# choose random sample with replacement...
			subset = np.asarray(np.ceil((m - 1) * np.random.rand(1, m - 1)), dtype=int)
			X_sub = X[subset,:][0]											
			Y_sub = Y[subset][0]

			classifier = base(X_sub, Y_sub, **kwargs)		# ...and train learner 'i' with those data
			self.bag.append(classifier)
	

	def predict(self, X):
		"""
		This method makes predictions on X. 

		Parameters
		----------
		X : N x M numpy array
			N = number of data points (not necessarily the same as in train), M = number of featurs
		"""
		n,m = np.asmatrix(X).shape
		b = len(self.bag)									# 'b' learners in the ensemble

		predictions = np.zeros((n,1))
		for i in range(b):									# make all b predictions
			predictions = np.concatenate((predictions, self.bag[i].predict(X)), axis=1)
		predictions = predictions[:,1:]

		C = len(self.classes)
		nc = np.zeros((n,C))
		for c in range(C):									# count how many instances of each class there are
			nc[:,c] = np.sum(predictions == self.classes[c], axis=1)

		indices = np.argmax(nc, axis=1)
		return np.asarray([[self.classes[i]] for i in indices])


## MUTATORS ####################################################################


	def set_classes(self, classes):
		"""
		Set classes of the classifier. 

		Parameters
		----------
		classes : list 
		"""
		if type(classes) is not list or len(classes) == 0:
			raise TypeError('BaggedClassify.set_classes: classes should be a list with a length of at least 1')
		self.classes = classes


	def set_component(self, i, base):
		"""
		This method sets the learner at index 'i' to 'base'.

		Parameters
		----------
		i : int
			the index of self.bag where 'base' will be placed
		base : object that inherits from 'Classify'
		"""
		self[i] = base


	def __setitem__(self, i, base):
		"""
		This method provides syntactic sugar for the
		method set_component. See set_component doc string
		for argument descriptions.
		"""
		if type(i) is not int:
			raise TypeError('BaggedClassify.set_component: invalid value for argument \'i\'')
		if Classify not in type(base).__bases__:
			raise TypeError('BaggedClassify.set_component: invalid value for argument \'base\'')

		try:
			self.bag[i] = base
		except IndexError:
			self.bag.append(base)


## INSPECTORS ##################################################################


	def get_classes(self):
		return self.classes


	def get_component(self, i):
		return self[i]


	def __getitem__(self, i):
		"""
		This method provides syntactic sugar for the
		get_component method. Indexing the BaggedClassifier
		at index 'i' returns the learner at index 'i' in
		self.bag.

		Parameters
		----------
		i : int
			the index that specifies the learner to be returned
		"""
		if type(i) is not int:
			raise TypeError('BaggedClassifier.__getitem__: argument \'i\' must be of type int')

		return self.bag[i]


	def __iter__(self):
		"""
		This method allows iteration over BaggedClassifier
		objects. Iteration over BaggedClassifier objects allows
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

## RANDOM TESTING ##############################################################

	data,classes = load_data_from_csv('../data/classifier-data.csv', 4, float)
	data,classes = arr(data), arr(classes)

	bd1,bc1 = filter_data(data, classes, lambda x,y: y == 2)
	bd1,bc1 = np.array(bd1), np.array(bc1)

	bases = [GaussBayesClassify, KNNClassify, TreeClassify] 
	bases += [LinearClassify, LogisticClassify]

	def test(trd, trc, ted, tec):
		print('bc', '\n')
		bc = BaggedClassify(bases[np.random.randint(len(bases))], 20, trd, trc)
		print(bc, '\n')
#		print(bc.predict(ted), '\n')
#		print(bc.predict_soft(ted), '\n')
#		print(bc.confusion(ted, tec), '\n')
		print(bc.auc(ted, tec), '\n')
		print(bc.roc(ted, tec), '\n')
		err = bc.err(ted, tec)
		print(err, '\n')
		return err

	avg_err = test_randomly(bd1, bc1, 0.8, test)

	print('avg_err')
	print(avg_err)


## DETERMINISTIC TESTING #######################################################

#	np.set_printoptions(linewidth=200)
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
#	print('knn_bc', '\n')
#	knn_bc = BaggedClassify(KNNClassify, 20, trd, trc)
#	print(knn_bc, '\n')
#	print(knn_bc.predict(ted), '\n')
#	print(knn_bc.err(ted, tec), '\n')
#	print(knn_bc.confusion(ted, tec), '\n')
#
#	print()
#
#	print('knn_ibc', '\n')
#	knn_ibc = BaggedClassify(KNNClassify, 30, ted, tec)
#	print(knn_ibc, '\n')
#	print(knn_ibc.predict(trd), '\n')
#	print(knn_ibc.err(trd, trc), '\n')
#	print(knn_ibc.confusion(trd, trc), '\n')
#
#	print()
#
#	print('knn_bbc', '\n')
#	knn_bbc = BaggedClassify(KNNClassify, 40, btrd, btrc)
#	print(knn_bbc, '\n')
#	print(knn_bbc.predict(bted), '\n')
#	print(knn_bbc.auc(bted, btec), '\n')
#	print(knn_bbc.err(bted, btec), '\n')
#	print(knn_bbc.confusion(bted, btec), '\n')
#	print(knn_bbc.roc(bted, btec), '\n')
#
#	print()
#
#	print('knn_ibbc', '\n')
#	knn_ibbc = BaggedClassify(KNNClassify, 50, bted, btec)
#	print(knn_ibbc, '\n')
#	print(knn_ibbc.predict(btrd), '\n')
#	print(knn_ibbc.auc(btrd, btrc), '\n')
#	print(knn_ibbc.err(btrd, btrc), '\n')
#	print(knn_ibbc.confusion(btrd, btrc), '\n')
#	print(knn_ibbc.roc(btrd, btrc), '\n')
#
#	print()
#
#	print('knn_bbc2', '\n')
#	knn_bbc2 = BaggedClassify(KNNClassify, 40, btrd2, btrc2)
#	print(knn_bbc2, '\n')
#	print(knn_bbc2.predict(bted2), '\n')
#	print(knn_bbc2.auc(bted2, btec2), '\n')
#	print(knn_bbc2.err(bted2, btec2), '\n')
#	print(knn_bbc2.confusion(bted2, btec2), '\n')
#	print(knn_bbc2.roc(bted2, btec2), '\n')
#
#	print()
#
#	print('knn_ibbc2', '\n')
#	knn_ibbc2 = BaggedClassify(KNNClassify, 50, bted2, btec2)
#	print(knn_ibbc2, '\n')
#	print(knn_ibbc2.predict(btrd2), '\n')
#	print(knn_ibbc2.auc(btrd2, btrc2), '\n')
#	print(knn_ibbc2.err(btrd2, btrc2), '\n')
#	print(knn_ibbc2.confusion(btrd2, btrc2), '\n')
#	print(knn_ibbc2.roc(btrd2, btrc2), '\n')
#
#	print()
#
#	print('dt_bc', '\n')
#	dt_bc = BaggedClassify(TreeClassify, 20, trd, trc)
#	print(dt_bc, '\n')
#	print(dt_bc.predict(ted), '\n')
#	print(dt_bc.err(ted, tec), '\n')
#	print(dt_bc.confusion(ted, tec), '\n')
#
#	print()
#
#	print('dt_ibc', '\n')
#	dt_ibc = BaggedClassify(TreeClassify, 30, ted, tec)
#	print(dt_ibc, '\n')
#	print(dt_ibc.predict(trd), '\n')
#	print(dt_ibc.err(trd, trc), '\n')
#	print(dt_ibc.confusion(trd, trc), '\n')
#
#	print()
#
#	print('dt_bbc', '\n')
#	dt_bbc = BaggedClassify(TreeClassify, 40, btrd, btrc)
#	print(dt_bbc, '\n')
#	print(dt_bbc.predict(bted), '\n')
#	print(dt_bbc.auc(bted, btec), '\n')
#	print(dt_bbc.err(bted, btec), '\n')
#	print(dt_bbc.confusion(bted, btec), '\n')
#	print(dt_bbc.roc(bted, btec), '\n')
#
#	print()
#
#	print('dt_ibbc', '\n')
#	dt_ibbc = BaggedClassify(TreeClassify, 50, bted, btec)
#	print(dt_ibbc, '\n')
#	print(dt_ibbc.predict(btrd), '\n')
#	print(dt_ibbc.auc(btrd, btrc), '\n')
#	print(dt_ibbc.err(btrd, btrc), '\n')
#	print(dt_ibbc.confusion(btrd, btrc), '\n')
#	print(dt_ibbc.roc(btrd, btrc), '\n')
#
#	print()
#
#	print('dt_bbc2', '\n')
#	dt_bbc2 = BaggedClassify(TreeClassify, 40, btrd2, btrc2)
#	print(dt_bbc2, '\n')
#	print(dt_bbc2.predict(bted2), '\n')
#	print(dt_bbc2.auc(bted2, btec2), '\n')
#	print(dt_bbc2.err(bted2, btec2), '\n')
#	print(dt_bbc2.confusion(bted2, btec2), '\n')
#	print(dt_bbc2.roc(bted2, btec2), '\n')
#
#	print()
#
#	print('dt_ibbc2', '\n')
#	dt_ibbc2 = BaggedClassify(TreeClassify, 50, bted2, btec2)
#	print(dt_ibbc2, '\n')
#	print(dt_ibbc2.predict(btrd2), '\n')
#	print(dt_ibbc2.auc(btrd2, btrc2), '\n')
#	print(dt_ibbc2.err(btrd2, btrc2), '\n')
#	print(dt_ibbc2.confusion(btrd2, btrc2), '\n')
#	print(dt_ibbc2.roc(btrd2, btrc2), '\n')
#
#	print()


################################################################################
################################################################################
################################################################################
