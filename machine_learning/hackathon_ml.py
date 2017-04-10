import numpy as np
import pandas as pd 
import sklearn
from sklearn import linear_model, pipeline, svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

def main():
	print("Reading data...")
	X,y = read_data("hackathon.csv")
	# scale_features(X)
	encode_categ_feats(X)

	X = X[:10000, :]
	y = y[:10000, 1]		# 0 = PREMIUM, 1 = TIV, 2 = DED
	# print(y)
	# print(y)
	print(X.shape)

	print("Training model...")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 13)


	# Plot data against long and lat
	# plt.scatter(X[:100, 0], y)
	# plt.ylabel('Premium')
	# plt.xlabel('Latitude')
	# plt.show()

	# Polynomial Regression Model
	# poly = preprocessing.PolynomialFeatures(degree=1)
	# X_train = poly.fit_transform(X_train)
	# X_test = poly.fit_transform(X_test)
	# clf = linear_model.LinearRegression()
	# clf.fit(X_train, y_train)

	# SVM Regression Model
	# clf = svm.SVR(degree=5)
	# clf.fit(X_train, y_train)

	# KNN Model
	clf = neighbors.KNeighborsRegressor(n_neighbors=3)
	clf.fit(X_train, y_train)

	print("Computing test accuracies...")
	predictions = clf.predict(X_test)

	# prediction = clf.predict([34.845617, -82.364463, 1, 1, 1950, 980.0300293, 194.0998688])
	# print(prediction)

	# neigh = clf.kneighbors(X_train, return_distance=False)
	# print(neigh)
	# print(X_train[neigh])
	# print(y_train[neigh])
	# predictions = np.mean(y_train[neigh], axis=1)
	r2 = sklearn.metrics.r2_score(y_test, predictions)

	fig1 = plt.figure()
	fig1.suptitle('Estimated TIVs against Location Attributes')

	ax1 = fig1.add_subplot(311)
	ax1.scatter(X_test[:, 0], predictions, c='r', label='Model')
	ax1.scatter(X_test[:, 0], y_test, c='g', label='Actual')
	ax1.set_ylim([0, 1000000])
	ax1.legend(loc='upper left')
	plt.xlabel('Latitude')
	plt.ylabel('TIV')

	ax2 = fig1.add_subplot(312)
	ax2.scatter(X_test[:, 1], predictions, c='r', label='Model')
	ax2.scatter(X_test[:, 1], y_test, c='g', label='Actual')
	ax2.set_ylim([0, 1000000])
	plt.xlabel('Longitude')
	plt.ylabel('TIV')

	ax3 = fig1.add_subplot(313)
	ax3.scatter(X_test[:, 6], predictions, c='r', label='Model')
	ax3.scatter(X_test[:, 6], y_test, c='g', label='Actual')
	ax3.set_ylim([0, 600000])
	plt.xlabel('Distance to Coast')
	plt.ylabel('TIV')
	
	plt.show()

	abs_error = sklearn.metrics.mean_absolute_error(y_test, predictions)

	# print("Predict:\n", predictions)
	# print("Actual:\n" , y_test)
	print("R2: ", r2)
	print("Absolute error: ", abs_error)


'''
Separates categorical and continuous data
Returns: 2-tuple (continuous feature indices, categorical feature indices) 
''' 
def cont_categ_split(feats):
	cont_vars = []
	categ_vars = []
	for col in range(feats.shape[1]):
		if isinstance(feats[0,col], str):
			categ_vars.append(col)
		else:
			cont_vars.append(col)

	return cont_vars, categ_vars

'''
Perform z-score standardization for scaling features
'''
def scale_features(feats):

	cont_vars, ret = cont_categ_split(feats)
	# perform z-score standarization on all continuous vars
	for col in cont_vars:
		mu = np.mean(feats[:, col])
		sd = np.std(feats[:, col])
		feats[:, col] = (feats[:, col] - mu)/sd

	return feats

'''
Transforms non-numeric features into integer categories
'''
def encode_categ_feats(feats):
	ret, categ_vars = cont_categ_split(feats)

	for col in categ_vars:
		lab_enc = preprocessing.LabelEncoder()
		# print("Before:\n", feats[:,col])
		feats[:, col] = lab_enc.fit_transform(feats[:, col])
		# print("After:\n", feats[:,col])
		
	# perform 1-hot encoding of categorical vars
	# might want to toggle 'sparse' arg
	one_enc = preprocessing.OneHotEncoder(categorical_features=categ_vars, sparse=False)
	# print("Before:\n", feats)
	feats = one_enc.fit_transform(feats)
	# print(one_enc.n_values)

	print(feats.shape)
	print("After:\n")
	print(feats.shape)


'''
Read and split data into training and test sets
'''
def read_data(file_name):
	df = pd.read_csv(file_name, low_memory=False, encoding = "ISO-8859-1")

	# eliminate % DED datapoints
	df = pd.concat([df[df.DED == 0], df[df.DED > 1]])

	# select features here
	# feats = ['PREMIUM', 'TIV', 'DED', 'LATITUDE', 'LONGITUDE', 'NUMBLDGS', 'BLDGSCHEME', 'NUMSTORIES', 'YEARBUILT', 'ELEVATION', 'DISTCOAST']
	df = df[df.COUNTRY == 'US']

	cols_needed = ['PREMIUM', 'TIV', 'DED', 'LATITUDE', 'LONGITUDE', 'NUMBLDGS', 'NUMSTORIES', 'YEARBUILT', 'ELEVATION', 'DISTCOAST']
	df['NUMBLDGS'] = pd.to_numeric(df['NUMBLDGS'])

	df = df[cols_needed]

	# replace invalid values by with NaN
	df['YEARBUILT'] = df['YEARBUILT'].replace([9999], np.nan)
	df.loc[df.ELEVATION <= 0, 'ELEVATION'] = np.nan
	df.loc[df.DISTCOAST <= 0, 'DISTCOAST'] = np.nan

	# # impute the means for these invalid values
	df['YEARBUILT'] = df['YEARBUILT'].fillna(value=np.nanmean(df['YEARBUILT']))
	df['ELEVATION'] = df['ELEVATION'].fillna(value=np.nanmean(df['ELEVATION']))
	df['DISTCOAST'] = df['DISTCOAST'].fillna(value=np.nanmean(df['DISTCOAST']))

	# print(df)
	df = df.values
	cols = df.shape[1]

	X = df[:, 3:]
	y = df[:, :3]
	return X,y



main()