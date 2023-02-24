import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv
import random
from time import perf_counter
import logging
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


logger = logging.getLogger("MainLogger")
logging.basicConfig(level=logging.INFO)


ORIG_DATAPATH = "../Data/wine_quality/winequality-whites.csv"
FIX_DATAPATH = "../Data/wine_quality/fixed-winequality-whites.csv"

warnings.filterwarnings("ignore", message="")

fig = plt.figure()
ax = plt.axes()

STEP_SIZE = 5
RUNS = 1

"""
Generally, info on sklearn methods, xgboost, and numpy APIs was taken from the API-specific online documentation.
"""




def timed(func):
	"""Wrapper to time function runtime"""
	
	# Info on decorators: https://realpython.com/primer-on-python-decorators/
	def wrapper(*args, **kwargs):
		runtime = perf_counter()
		out = func(*args, **kwargs)
		logger.info("{0} function took {1:.2f} secs".format(func.__name__, (perf_counter()-runtime)))
		return out
	return wrapper


def get_data(fname=""):
	"""Returns 2D numpy arrays of the training and test data"""
	out = None
	with open(fname, 'r') as file:
		csv_reader = csv.reader(file, delimiter=',')
		out = np.array(list(csv_reader))
	columns = out[0,1:]
	data = out[1:,1:-1].astype(float)
	labels = out[1:, -1:].astype(int).flatten()
	return data, labels, columns

# @timed
def SVM(X, y, seed=None) -> SVR:
	"""Returns a fit sklearn Linear SVM Model"""
	# SVM = SVR()
	SVM = LinearSVR()
	SVM.fit(X, y)
	return SVM

# @timed
def XGB(X, y, seed) -> xgb.XGBRegressor:
	"""Returns a fit sklearn XGB Model"""
	GBDT = xgb.XGBRegressor(random_state=seed, n_estimators=1500)
	GBDT.fit(X, y)
	return GBDT

# @timed
def RF(X, y, seed) -> RandomForestRegressor:
	"""Returns a fit sklearn RF Model"""
	RF = RandomForestRegressor(random_state=seed)
	RF.fit(X, y)
	return RF

# @timed
def MLR(X, y, seed=None) -> LinearRegression:
	"""Returns a fit sklearn Linear Model"""
	MLR = LinearRegression()
	MLR.fit(X, y)
	return MLR


# @timed
def HR(X, y, seed) -> HuberRegressor:
	"""Returns a fit sklearn Linear Model"""
	HR = HuberRegressor(max_iter=int(1e3))
	HR.fit(X, y)
	return HR


def test_model(model, test_data, test_labels):
	"""Simple method to test a model"""
	pred_labels = model.predict(test_data)
	MSE = mean_squared_error(test_labels, pred_labels)
	print(type(model), " gives MSE: %.3f" % MSE)


def train_and_get_test_MSE(func, train_data, train_labels, test_data, test_labels, seed, subarray_size=None):
	"""Trains a model with the given arguments, and returns the test MSE"""
	if subarray_size:
		# generate k-sized subarrays
		# seed2 = random.randint(0, 1e9)
		# subarray_indices = get_subset_array(train_data.shape[0], subarray_size)
		# model = func(X=train_data[subarray_indices,:], y=train_labels[subarray_indices], seed=seed2)
		model = func(X=train_data[:subarray_size,:], y=train_labels[:subarray_size], seed=seed)
	else:
		model = func(X=train_data, y=train_labels, seed=seed)
	return mean_squared_error(test_labels, model.predict(test_data))


def get_mean_and_std(runs=30, *args, **kwargs):
	"""Get the mean and std of some function"""
	mse_list = np.array([train_and_get_test_MSE(*args, **kwargs) for _ in range(runs)])
	return np.mean(mse_list), np.std(mse_list)


def get_subset_array(size, ind):
	"""Returns a random subset of indices of an array of some size"""
	arr = np.arange(size)
	np.random.shuffle(arr)
	return arr[:ind]


def save_plot(fname):
	"""Saves the plot, and labels it"""
	plt.ylabel('MSE')
	plt.xlabel('Train Dataset Size')
	# plt.ylim(0, 2.5)
	plt.title("MSE vs the K-Size Training Data")
	plt.legend(loc='upper right')
	from io import BytesIO
	figfile = BytesIO()
	plt.savefig(figfile, format='png', dpi=200)
	try:
		f = open(fname + '.png', 'wb')
		f.write(figfile.getvalue())
	finally:
		f.close()
	plt.clf()
	fig = plt.figure()
	ax = plt.axes()


def plot(means, stds, k, label, runs):
	"""Plot the given MSE values with a bounds for the known std"""
	plt.plot(k, means, lw=2.0, label=label)
	stds = np.array(stds)
	means = np.array(means)
	# print(stds)
	if runs > 1:
		ax.fill_between(k, means+stds, means-stds, alpha=0.3)
	# plt.errorbar(k, means, yerr=stds, color="black", capsize=3, lw=1.0)
	# save_plot(fname, fig)



def get_k_shot_perf(func, train_data, train_labels, test_data, test_labels, seed, label, fname=None, step=100, runs=30):
	"""Trains the model on subsets of the training data to test few-shot performance."""
	k = []
	means = []
	stds = []
	# for i in tqdm(range(2, train_data.shape[0])):
	for i in tqdm(range(train_data.shape[0], 15, -1*step)):  # 20
	# for i in range(1, train_data.shape[0], step):
		# # Test on subarrays
		mean, std = get_mean_and_std(func=func, runs=runs, train_data=train_data, train_labels=train_labels, 
			test_data=test_data, test_labels=test_labels, subarray_size=i, seed=seed)
		# Append values to lists
		k.append(i)
		means.append(mean)
		stds.append(std)
	print(label, means, k)
	plot(means, stds, k, label=label, runs=runs)


def save_data(data, fname):
	"""Saves data using pickle"""
	with open(fname, 'wb') as file:
		pickle.dump(data, file)


def load_data(fname):
	"""Loads data using pickle"""
	try:
		with open(fname, 'wb') as file:
			data = pickle.load(file)
		return data
	except:
		return None


def main():
	# Generate seed
	# seed = random.randint(0, 1e9)
	seed = 102649420
	print("Seed: ", seed)

	random.seed(seed)
	np.random.seed(seed)
	
	# Grab Data
	data, labels, columns = get_data(fname=FIX_DATAPATH)
	logger.info("Data collected.")
	
	# Split data into training and test sets
	train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=seed, test_size=0.20)
	
	# # Test Support Vector Machines
	# get_k_shot_perf(func=SVM, train_data=train_data, train_labels=train_labels, 
	# 	test_data=test_data, test_labels=test_labels, seed=seed, fname="../Plots/SVM_Regressor", label='SVM', step=STEP_SIZE, runs=RUNS)

	# # Test Gradient Boosted Decision Trees
	get_k_shot_perf(func=XGB, train_data=train_data, train_labels=train_labels, 
		test_data=test_data, test_labels=test_labels, seed=seed, fname="../Plots/XGB_Regressor", label='XGB', step=STEP_SIZE, runs=RUNS)

	# Test Random Forest
	# get_k_shot_perf(func=RF, train_data=train_data, train_labels=train_labels, 
		# test_data=test_data, test_labels=test_labels, seed=seed, fname="../Plots/RF_Regressor", label='RF', step=STEP_SIZE, runs=RUNS)
	
	# Test Multiple Linear Regression
	get_k_shot_perf(func=MLR, train_data=train_data, train_labels=train_labels, 
		test_data=test_data, test_labels=test_labels, seed=seed, fname="../Plots/MLR_Regressor", label='MLR', step=STEP_SIZE, runs=RUNS)
	
	# Test Huber Linear Regression
	# get_k_shot_perf(func=HR, train_data=train_data, train_labels=train_labels, 
		# test_data=test_data, test_labels=test_labels, seed=seed, fname="../Plots/HR_Regressor", label='HR', step=STEP_SIZE, runs=RUNS)

	save_plot(fname="./comb_plot")


if __name__ == "__main__":
	main()
	pass

