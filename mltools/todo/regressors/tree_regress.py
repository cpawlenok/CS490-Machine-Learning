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
## TREEREGRESS #################################################################
################################################################################


class TreeRegress(Regress):

	def __init__(self, X=None, Y=None, min_parent=2, max_depth=np.inf, min_score=-1, n_features=None):
		"""
		Constructor for TreeRegressor (decision tree regression model).

		Parameters
		----------
		X : numpy array 
			N x M numpy array which contains N data points with M features.
		Y : numpy array 
			1 x N numpy array that contains values the relate to the data
		  	points in X.
		min_parent : int 
			Minimum number of data required to split a node. 
		min_score : int 
			Minimum value of score improvement to split a node.
		max_depth : int 
			Maximum depth of the decision tree. 
		n_features : int 
			Number of available features for splitting at each node.
		"""
		self.L = arr([0])			# indices of left children
		self.R = arr([0])			# indices of right children
		self.F = arr([0])			# feature to split on (-1 = leaf = predict)
		self.T = arr([0])			# threshold to split on (prediction value if leaf)
	
		if type(X) is np.ndarray and type(Y) is np.ndarray:					
			self.train(X, Y, min_parent, max_depth, min_score, n_features)	# train if data is provided

	
	def __repr__(self):
		to_return = 'Decision Tree Regressor \nThresholds: {}'.format(
			str(self.T) if len(self.T) < 4 else 
			'[{0:.2f}, {1:.2f} ... {2:.2f}, {3:.2f}]'
			.format(self.T[0], self.T[1], self.T[-1], self.T[-2]))
		return to_return


	def __str__(self):
		to_return = 'Decision Tree Regressor \nThresholds: {}'.format(
			str(self.T) if len(self.T) < 4 else 
			'[{0:.2f}, {1:.2f} ... {2:.2f}, {3:.2f}]'
			.format(self.T[0], self.T[1], self.T[-1], self.T[-2]))
		return to_return


## CORE METHODS ################################################################


	def train(self, X, Y, min_parent=2, max_depth=np.inf, min_score=-1, n_features=None):
		"""
		This method trains a random forest regression model.
		Refer to constructor doc string for description of arguments.
		"""
		n,d = mat(X).shape
		n_features = min(n_features if n_features else d, d)

		sz = min(2 * n, 2**(max_depth + 1))		
		L = np.zeros((1,sz)).ravel()					# allocate memory for binary trees with 
		R = np.zeros((1,sz)).ravel()					# given max_depth and >= 1 datum per leaf
		F = np.zeros((1,sz)).ravel()							
		T = np.zeros((1,sz)).ravel()

		L, R, F, T, last = self.__dectree_train(X, Y, L, R, F, T, 1, 0, min_parent, max_depth, min_score, n_features)

		self.L = L[1:last]								# store returned data into object
		self.R = R[1:last]								
		self.F = F[1:last]
		self.T = T[1:last]

		self.L = arr([i - 1 for i in self.L])			# must subtract 1 from each index to make indices 
		self.R = arr([i - 1 for i in self.R])			# consistent with Python's zero-based indexing


	def predict(self, X):
		"""
		This method makes predictions on the test data X.

		Parameters
		----------
		X : numpy array 
			N x M numpy array which contains N data points with
		  	M features. N doesn't necessarily have to be the same as
		  	for X in train.
		"""
		return self.__dectree_test(X, self.L, self.R, self.F, self.T, 0)


## MUTATORS ####################################################################


## INSPECTORS ##################################################################

	
## HELPERS #####################################################################


	def __dectree_train(self, X, Y, L, R, F, T, next, depth, min_parent, max_depth, min_score, n_features):
		"""
		This is a recursive helper method that recusively trains the decision tree. Used in:
			train

		TODO:
			compare for numerical tolerance
		"""
		n,d = mat(X).shape

		# check leaf conditions...
		if n < min_parent or depth >= max_depth or np.var(Y) < min_score:
			assert n != 0, ('TreeRegress.__dectree_train: tried to create size zero node')
			return self.__output_leaf(Y, n, L, R, F, T, next)

		best_val = np.inf
		best_feat = -1
		try_feat = np.random.permutation(d)

		# ...otherwise, search over (allowed) features
		for i_feat in try_feat[0:n_features]:
			dsorted = arr(np.sort(X[:,i_feat].T)).ravel()						# sort data...
			pi = np.argsort(X[:,i_feat].T)										# ...get sorted indices...
			tsorted = Y[pi].ravel()												# ...and sort targets by feature ID
			can_split = np.append(arr(dsorted[:-1] != dsorted[1:]), 0)			# which indices are valid split points?

			if not np.any(can_split):
				continue

			# find min weighted variance among split points
			val,idx = self.__min_weighted_var(tsorted, can_split, n)

			# save best feature and split point found so far
			if val < best_val:
				best_val = val
				best_feat = i_feat
				best_thresh = (dsorted[idx] + dsorted[idx + 1]) / 2

		# if no split possible, output leaf (prediction) node
		if best_feat == -1:			
			return self.__output_leaf(Y, n, L, R, F, T, next)

		# split data on feature i_feat, value (tsorted[idx] + tsorted[idx + 1]) / 2
		F[next] = best_feat
		T[next] = best_thresh
		go_left = X[:,F[next]] < T[next]
		my_idx = next
		next += 1

		# recur left
		L[my_idx] = next	
		L,R,F,T,next = self.__dectree_train(X[go_left,:], Y[go_left], L, R, F, T, 
			next, depth + 1, min_parent, max_depth, min_score, n_features)

		# recur right
		R[my_idx] = next	
		L,R,F,T,next = self.__dectree_train(X[np.logical_not(go_left),:], Y[np.logical_not(go_left)], L, R, F, T, 
			next, depth + 1, min_parent, max_depth, min_score, n_features)

		return (L,R,F,T,next)


	def __dectree_test(self, X, L, R, F, T, pos):
		"""
		This is a recursive helper method that finds leaf nodes
		in the decision tree for prediction. Used in:
			predict
		"""
		y_hat = np.zeros((len(mat(X)), 1))
		
		if F[pos] == -1:
			y_hat[:] = T[pos]
		else:
			go_left = X[:,F[pos]] < T[pos]
			y_hat[go_left] = self.__dectree_test(X[go_left,:], L, R, F, T, L[pos])
			y_hat[np.logical_not(go_left)] = self.__dectree_test(X[np.logical_not(go_left),:], L, R, F, T, R[pos])

		return y_hat


	def __output_leaf(self, Y, n, L, R, F, T, next):
		"""
		This is a helper method that handles leaf node termination
		conditions. Used in:
			__dectree_train
		"""
		F[next] = -1
		T[next] = np.mean(Y)		
		next += 1
		return (L,R,F,T,next)


	def __min_weighted_var(self, tsorted, can_split, n):
		"""
		This is a helper method that finds the minimum weighted variance
		among all split points. Used in:
			__dectree_train
		"""
		# compute mean up to and past position j (for j = 0..n)
		y_cum_to = np.cumsum(tsorted, axis=0)
		y_cum_pa = y_cum_to[-1] - y_cum_to
		mean_to = y_cum_to / arr(range(1, n + 1))		
		mean_pa = y_cum_pa / arr(list(range(n - 1, 0, -1)) + [1])

		# compute variance up to, and past position j (for j = 0..n)
		y2_cum_to = np.cumsum(np.power(tsorted, 2), axis=0)
		y2_cum_pa = y2_cum_to[-1] - y2_cum_to
		var_to = (y2_cum_to - 2 * mean_to * y_cum_to + list(range(1, n + 1)) * np.power(mean_to, 2)) / list(range(1, n + 1))
		var_pa = (y2_cum_pa - 2 * mean_pa * y_cum_pa + list(range(n - 1, -1, -1)) * np.power(mean_pa, 2)) / arr(list(range(n - 1, 0, -1)) + [1])
		var_pa[-1] = np.inf

		# find minimum weighted variance among all split points
		weighted_variance = arr(range(1, n + 1)) / n * var_to + arr(range(n - 1, -1, -1)) / n * var_pa
		val = np.nanmin((weighted_variance + 1) / (can_split + 1e-100))			# nan versions of min functions must be used to ignore nans
		idx = np.nanargmin((weighted_variance + 1) / (can_split + 1e-100))		# find only splittable points

		return (val,idx)


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	np.set_printoptions(linewidth=200)

## RANDOM TESTING ##############################################################

	data,predictions = load_data_from_csv('../data/regressor-data.csv', -1, float)
	data,predictions = arr(data), arr(predictions)

	def test(trd, trc, ted, tec):
		print('tr', '\n')
		tr = TreeRegress(trd, trc)
		print(tr, '\n')
		err = tr.mae(ted, tec)
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
#	print('tr', '\n')
#	tr = TreeRegress(trd, trp)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd2, trp2)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd3, trp3)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd, trp, min_parent=1)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd2, trp2, min_parent=3)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd3, trp3, min_parent=5)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd, trp, max_depth=10)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd2, trp2, max_depth=100)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd3, trp3, max_depth=500)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd, trp, min_score=10)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd2, trp2, min_score=50)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd3, trp3, min_score=100)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd, trp, n_features=1)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd2, trp2, n_features=2)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd3, trp3, n_features=3)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd, trp, min_parent=1, max_depth=10000, min_score=.001, n_features=3)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd2, trp2, min_parent=10, max_depth=20, min_score=1, n_features=1)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')
#
#	print()
#
#	print('tr', '\n')
#	tr = TreeRegress(trd3, trp3, min_parent=1, max_depth=20, min_score=.001, n_features=1)
#	print(tr, '\n')
#	print(tr.predict(ted), '\n')
#	print(tr.mae(ted, tep), '\n')
#	print(tr.mse(ted, tep), '\n')
#	print(tr.rmse(ted, tep), '\n')


################################################################################
################################################################################
################################################################################
