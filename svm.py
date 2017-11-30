import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

## Jingwen Liao
## Julia Cosma

def read_csv(filename):
	file = open(filename, 'r')
	lines = file.readline().split('\r')
	X = []
	Y = []
	features = lines[0].split(',')[1:]
	for line in lines[1:-1]:
		values = line.split(',')[:-1]
		Y += [int(values[0])]
		expression = []
		for value in values[1:]:
			expression += [float(value)]
		X += [expression]
	Y = np.array(Y)
	X = np.array(X)
	return X, Y, features

def main():
	#X, Y, features = read_csv('Microarray_data.csv')
	#X, Y, features = read_csv('Microarray_data_centered.csv')
	X, Y, features = read_csv('Microarray_data_with_reduced_features_centered.csv')
	
	num_samples, num_features = X.shape

	# Maintaining class bias
	AD_indexs = [i for i in range(num_samples) if Y[1] == 0]
	SQ_indexs = [i for i in range(num_samples) if Y[1] == 1]
	AD_train ,AD_test = train_test_split(AD_indexs,test_size=0.4, random_state = 1118)
	SQ_train ,SQ_test = train_test_split(SQ_indexs,test_size=0.4, random_state = 527)
	train_indices = AD_train + SQ_train
	test_indices = AD_test + SQ_test

	X_train = X[train_indices]
	Y_train = Y[train_indices]
	X_test = X[test_indices]
	Y_test = Y[test_indices]

	# Create and fit linear SVM
	clf = svm.SVC(kernel='linear')
	clf.fit(X_train,Y_train)
	Ypred_train = clf.predict(X_train)
	print sum([Ypred_train[i] == Y_train[i] for i in range(len(Y_train))])*1.0 / len(Y_train)
	Ypred_test = clf.predict(X_test)
	print sum([Ypred_test[i] == Y_test[i] for i in range(len(Y_test))])*1.0 / len(Y_test)

	# Preliminary metric: confusion matrix
	tn, fp, fn, tp = confusion_matrix(Y_train, Ypred_train).ravel()
	print "True negatives: {}, false positives: {}, false negatives: {}, true positives: {}".format(tn, fp, fn, tp)

	tn, fp, fn, tp = confusion_matrix(Y_test, Ypred_test).ravel()
	print "True negatives: {}, false positives: {}, false negatives: {}, true positives: {}".format(tn, fp, fn, tp)

if __name__ == "__main__" :
    main()
