################################################################################
## IMPORTS #####################################################################
################################################################################


#import data
import numpy as np
import random

from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod


################################################################################
################################################################################
################################################################################


################################################################################
## TESTING FUNCTIONS ###########################################################
################################################################################


def cross_validate(X, Y, n_folds, i_fold):
	"""
	Function that splits data for n-fold cross validation.

	Parameters
	----------
	X : numpy array
		N x M numpy array that contains data points.
	Y : numpy array
		1 x N numpy array that contains labels that correspond to
		data points in X.
	n_folds : int
		Total number of data folds.
	i_fold : int
		The fold for which the current call of cross_validate
		will partition.

	Returns
	-------
	to_return : (Xtr,Xte,Ytr,Yte)
		Tuple that contains (in this order) training data from X, testing
		data from X, training labels from Y, and testing labels from Y.
	"""
	Y = arr(Y).flatten()

	nx,dx = twod(X).shape
	ny = len(Y)
	idx = range(nx)

	if ny > 0:
		assert nx == ny, 'cross_validate: X and Y must have the same length'

	n = np.fix(nx / n_folds)

	te_start = int((i_fold - 1) * n)
	te_end = int((i_fold * n)) if (i_fold * n) <= nx else int(nx)
	test_range = list(range(te_start, te_end))
	train_range = sorted(set(idx) - set(test_range))

	to_return = (X[train_range,:], X[test_range,:])
	if ny > 0:
		to_return += (Y[train_range], Y[test_range])

	return to_return


def test_randomly(data, labels, mix=0.8, end=0, test=lambda x: 1.0, *args):
	"""
	Function that performs random tests using data/labels.

	Parameters
	----------
	data : numpy array
		N x M array of data points used for training/testing learner.
		N = number of data; M = number of features.
	labels : numpy array
		1 x N array of class/regression labels used for training/testing learner.
	mix : float
		The percentage of data to use for training (1 - mix = percentage of data
		used for testing).
	end : int
		The number of tests to run.
	test : function object
		A function that takes at least four arguments (arrays containing data/labels
		for testing/training) and performs tests. This function should return an
		error value for one experiment.
	args : mixed
		Any additional arguments needed for testing.

	Returns
	-------
	float
		Average error value of all tests performed.
	"""
	start = 0
	end = len(data) if end == 0 else end

	avg_err = 0

	for i in range(start, end):
		indexes = range(len(data))
		train_indexes = random.sample(indexes, int(mix * len(data)))
		test_indexes = list(set(indexes) - set(train_indexes))

		trd,trc = data[train_indexes], labels[train_indexes]
		ted,tec = data[test_indexes], labels[test_indexes]
		avg_err += test(trd, trc, ted, tec, *args)

	return avg_err / end


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	pass
#	data,classes = data.load_data_from_csv('../data/classifier-data.csv', 4, float)
#	data,classes = arr(data), arr(classes)
#
#	for i in range(1,6):
#		Xtr,Xte,Ytr,Yte = cross_validate(data, classes, 5, i)
#		print('i =', i)
#		print('len(Xtr)')
#		print(len(Xtr))
#		print('len(Xte)')
#		print(len(Xte))
#		print('len(Ytr)')
#		print(len(Ytr))
#		print('len(Yte)')
#		print(len(Yte))
#		print()


################################################################################
################################################################################
################################################################################
