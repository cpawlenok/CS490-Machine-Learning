################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np

from classify import Classify
from numpy import array as arr
from utils.data import filter_data, load_data_from_csv
from utils.test import test_randomly


################################################################################
################################################################################
################################################################################


################################################################################
## TREECLASSIFY ################################################################
################################################################################


class TreeClassify(Classify):

	def __init__(self, X=None, Y=None, min_parent=2, max_depth=np.inf, n_features=None):
		"""
		Constructor for TreeClassifier (decision tree classifier).

		Parameters
		----------
		X : numpy array 
			N x M numpy array which contains N data points with M features.
		Y : numpy array
			N x 1 numpy array which contains class labels for the data points in X. 
		min_parent : int 
			The minimum number of data required to split a node. 
		max_depth : int 
			The maximum depth of the decision tree. 
		n_features : int 
			The number of available features for splitting at each node.
		"""
		self.L = np.asarray([0])
		self.R = np.asarray([0])
		self.F = np.asarray([0])
		self.T = np.asarray([0])
	
		self.classes = []

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			self.train(X, Y, min_parent, max_depth, n_features)

	
	def __repr__(self):
		to_return = 'Decision Tree Classifier; {} features\nThresholds: {}'.format(
			len(self.classes), str(self.T) if len(self.T) < 4 else 
			'[{0:.2f}, {1:.2f} ... {2:.2f}, {3:.2f}]'
			 .format(self.T[0], self.T[1], self.T[-1], self.T[-2]))
		return to_return


	def __str__(self):
		to_return = 'Decision Tree Classifier; {} features\nThresholds: {}'.format(
			len(self.classes), str(self.T) if len(self.T) < 4 else 
			'[{0:.2f}, {1:.2f} ... {2:.2f}, {3:.2f}]'
			 .format(self.T[0], self.T[1], self.T[-1], self.T[-2]))
		return to_return


## CORE METHODS ################################################################


	def train(self, X, Y, min_parent=2, max_depth=np.inf, n_features=None):
		"""
		This method trains a random forest classification tree.
		Refer to constructor doc string for description of arguments.
		"""
		n,d = np.asmatrix(X).shape
		n_features = min(n_features if n_features else d, d)
		min_score = -1

		self.classes = list(np.unique(Y)) if len(self.classes) == 0 else self.classes
		Y = self.to_index(Y)

		sz = min(2 * n, 2**(max_depth + 1))
		L = np.zeros((1,sz)).ravel()
		R, F, T = np.zeros((1,sz)).ravel(), np.zeros((1,sz)).ravel(), np.zeros((1,sz)).ravel()

		L, R, F, T, last = self.__dectree_train(X, Y, L, R, F, T, 1, 0, min_parent, max_depth, min_score, n_features)

		self.L = L[1:last]
		self.R = R[1:last]
		self.F = F[1:last]
		self.T = T[1:last]

		self.L = np.asarray([i - 1 for i in self.L])
		self.R = np.asarray([i - 1 for i in self.R])


	def predict(self, X):
		"""
		This method makes predictions on the test data X.

		Parameters
		----------
		X : numpy array
			N x M numpy array which contains N data points with M features. 
			N doesn't necessarily have to be the same as for X in train.
		"""
		Y_te = self.__dectree_test(X, self.L, self.R, self.F, self.T, 0).T.ravel()
		return np.asarray([[self.classes[int(i)]] for i in Y_te])


## MUTATORS ####################################################################


	def set_classes(self, classes):
		"""
		Set classes of the classifier. 

		Parameters
		----------
		classes : list 
			List of class labels.
		"""
		if type(classes) is not list or len(classes) == 0:
			raise TypeError('TreeClassify.set_classes: classes should be a list with a length of at least 1')
		self.classes = classes


## INSPECTORS ##################################################################

	
	def get_classes(self):
		return self.classes


## HELPERS #####################################################################


	def __dectree_train(self, X, Y, L, R, F, T, next, depth, min_parent, max_depth, min_score, n_features):
		"""
		This is a recursive helper method that recusively trains the decision tree. Used in:
			train
		"""
		n,d = np.asmatrix(X).shape
		num_classes = max(Y)

		if n < min_parent or depth >= max_depth or np.all(Y == Y[0]):
			assert n != 0, ('TreeClassify.__dectree_train: tried to create size zero node')
			F[next] = -1
			tmp = np.sum(self.to_1_of_k(Y, range(1, num_classes + 1)), axis=0)
			nc = np.max(tmp)
			T[next] = np.argmax(tmp)
			next += 1
			return (L,R,F,T,next)

		best_val = -np.inf
		best_feat = -1
		try_feat = np.random.permutation(d)
		#try_feat = range(d)

		for i_feat in try_feat[0:n_features]:
			dsorted = np.asarray(np.sort(X[:,i_feat].T)).ravel()
			pi = np.argsort(X[:,i_feat].T)
			tsorted = Y[pi].ravel()
			can_split = np.append(np.asarray(dsorted[:-1] != dsorted[1:]), 0)

			if not np.any(can_split):
				continue

			y_left = np.cumsum(self.to_1_of_k(tsorted, range(1, num_classes + 1)), axis=0)
			y_right = np.asarray([y_left[-1,:] - y_left[i,:] for i in range(len(y_left))])

			y_left = np.asarray([y_left[:,i].T / np.asarray(range(1, n + 1)) for i in range(len(y_left.T))]).T
			y_right = np.asarray([y_right[:,i].T / np.asarray(list(range(n - 1, 0, -1)) + [1]) for i in range(len(y_right.T))]).T

			h_root = np.dot(-y_left[-1,:], np.log(y_left[-1,:] + np.spacing(1)).T) 
			h_left = -np.sum(y_left * np.log(y_left + np.spacing(1)), axis=1)
			h_right = -np.sum(y_right * np.log(y_right + np.spacing(1)), axis=1)

			IG = h_root - (np.divide(range(1,n+1), n) * h_left + np.divide(range(n - 1, -1, -1), n) * h_right)
			val = np.max((IG + np.spacing(1)) * can_split)
			index = np.argmax((IG + np.spacing(1)) * can_split)

			if val > best_val:
				best_val = val
				best_feat = i_feat
				best_thresh = (dsorted[index] + dsorted[index + 1]) / 2

		if best_feat == -1:
			F[next] = -1
			tmp = np.sum(self.to_1_of_k(Y, range(1, num_classes + 1)), axis=0)
			nc = np.max(tmp)
			T[next] = np.argmax(tmp)
			next += 1
			return (L,R,F,T,next)

		F[next] = best_feat
		T[next] = best_thresh

		go_left = X[:,F[next]] < T[next]
		my_idx = next
		next += 1

		L[my_idx] = next
		L,R,F,T,next = self.__dectree_train(X[go_left,:], Y[go_left], L, R, F, T, 
			next, depth + 1, min_parent, max_depth, min_score, n_features)

		R[my_idx] = next
		L,R,F,T,next = self.__dectree_train(X[np.logical_not(go_left),:], Y[np.logical_not(go_left)], L, R, F, T, 
			next, depth + 1, min_parent, max_depth, min_score, n_features)

		return (L,R,F,T,next)


	def __dectree_test(self, X, L, R, F, T, pos):
		"""
		This is a recursive helper method that finds class labels
		in the decision tree for prediction. Used in:
			predict
		"""
		y_hat = np.zeros((len(np.asmatrix(X)), 1))
		
		if F[pos] == -1:
			y_hat[:] = T[pos]
		else:
			go_left = X[:,F[pos]] < T[pos]
			y_hat[go_left] = self.__dectree_test(X[go_left,:], L, R, F, T, L[pos])
			y_hat[np.logical_not(go_left)] = self.__dectree_test(X[np.logical_not(go_left),:], L, R, F, T, R[pos])

		return y_hat


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
		print('tc', '\n')
		tc = TreeClassify(trd, trc)
		print(tc, '\n')
	#	print(tc.predict(ted), '\n')
	#	print(tc.predict_soft(ted), '\n')
	#	print(tc.confusion(ted, tec), '\n')
	#	print(tc.auc(ted, tec), '\n')
	#	print(tc.roc(ted, tec), '\n')
		err = tc.err(ted, tec)
		print(err, '\n')
		return err

	avg_err = test_randomly(data, classes, 0.8, test)

	print('avg_err')
	print(avg_err)

## DETERMINISTIC TESTING #######################################################

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
#	print('tc', '\n')
#	tc = TreeClassify(trd, trc)
#	print(tc, '\n')
#	print(tc.predict(ted), '\n')
#	print(tc.err(ted, tec), '\n')
#	print(tc.confusion(ted, tec), '\n')
#
#	print()
#
#	print('itc', '\n')
#	itc = TreeClassify(ted, tec)
#	print(itc, '\n')
#	print(itc.predict(trd), '\n')
#	print(itc.err(trd, trc), '\n')
#	print(itc.confusion(trd, trc), '\n')
#
#	print()
#
#	print('btc', '\n')
#	btc = TreeClassify(btrd, btrc)
#	print(btc, '\n')
#	print(btc.predict(bted), '\n')
#	print(btc.auc(bted, btec), '\n')
#	print(btc.err(bted, btec), '\n')
#	print(btc.roc(bted, btec), '\n')
#	print(btc.confusion(bted, btec), '\n')
#
#	print()
#
#	print('ibtc', '\n')
#	ibtc = TreeClassify(bted, btec)
#	print(ibtc, '\n')
#	print(ibtc.predict(btrd), '\n')
#	print(ibtc.auc(btrd, btrc), '\n')
#	print(ibtc.err(btrd, btrc), '\n')
#	print(ibtc.roc(btrd, btrc), '\n')
#	print(ibtc.confusion(btrd, btrc), '\n')
#
#	print()
#
#	print('btc2', '\n')
#	btc2 = TreeClassify(btrd2, btrc2)
#	print(btc2, '\n')
#	print(btc2.predict(bted2), '\n')
#	print(btc2.auc(bted2, btec2), '\n')
#	print(btc2.err(bted2, btec2), '\n')
#	print(btc2.roc(bted2, btec2), '\n')
#	print(btc2.confusion(bted2, btec2), '\n')
#
#	print()
#
#	print('ibtc2', '\n')
#	ibtc2 = TreeClassify(bted2, btec2)
#	print(ibtc2, '\n')
#	print(ibtc2.predict(btrd2), '\n')
#	print(ibtc2.auc(btrd2, btrc2), '\n')
#	print(ibtc2.err(btrd2, btrc2), '\n')
#	print(ibtc2.roc(btrd2, btrc2), '\n')
#	print(ibtc2.confusion(btrd2, btrc2), '\n')


################################################################################
################################################################################
################################################################################
