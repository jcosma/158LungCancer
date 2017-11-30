import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
	X, Y, features = read_csv('Microarray_data.csv')

	"""
	PCA_X = PCA(n_components=2).fit_transform(X)
	TSNE_X= TSNE(n_components=2).fit_transform(X)
	
	AD_X = [PCA_X[i][0] for i in range(len(X)) if Y[i] == 0]
	AD_Y = [PCA_X[i][1] for i in range(len(X)) if Y[i] == 0]
	SQ_X = [PCA_X[i][0] for i in range(len(X)) if Y[i] == 1]
	SQ_Y = [PCA_X[i][1] for i in range(len(X)) if Y[i] == 1]
	
	plt.plot(AD_X, AD_Y, 'go', label = 'AD')
	plt.plot(SQ_X, SQ_Y, 'r^', label = 'SQ')
	plt.title('Dimensionality reduction with PCA (2D)')
	plt.legend()
	plt.show()

	AD_X = [TSNE_X[i][0] for i in range(len(X)) if Y[i] == 0]
	AD_Y = [TSNE_X[i][1] for i in range(len(X)) if Y[i] == 0]
	SQ_X = [TSNE_X[i][0] for i in range(len(X)) if Y[i] == 1]
	SQ_Y = [TSNE_X[i][1] for i in range(len(X)) if Y[i] == 1]
	
	plt.plot(AD_X, AD_Y, 'go', label = 'AD')
	plt.plot(SQ_X, SQ_Y, 'r^', label = 'SQ')
	plt.title('Dimensionality reduction with t-SNE (2D)')
	plt.legend()
	plt.show()
	

	PCA_X = PCA(n_components=3).fit_transform(X)
	TSNE_X= TSNE(n_components=3).fit_transform(X)
	
	AD_X = [PCA_X[i][0] for i in range(len(X)) if Y[i] == 0]
	AD_Y = [PCA_X[i][1] for i in range(len(X)) if Y[i] == 0]
	AD_Z = [PCA_X[i][2] for i in range(len(X)) if Y[i] == 0]
	SQ_X = [PCA_X[i][0] for i in range(len(X)) if Y[i] == 1]
	SQ_Y = [PCA_X[i][1] for i in range(len(X)) if Y[i] == 1]
	SQ_Z = [PCA_X[i][2] for i in range(len(X)) if Y[i] == 1]
	

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(AD_X, AD_Y, AD_Z, c='g', marker='o')
	ax.scatter(SQ_X, SQ_Y, SQ_Z, c='r', marker='^')
	plt.title('Dimensionality reduction with PCA (3D)')
	plt.show()
	
	AD_X = [TSNE_X[i][0] for i in range(len(X)) if Y[i] == 0]
	AD_Y = [TSNE_X[i][1] for i in range(len(X)) if Y[i] == 0]
	AD_Z = [TSNE_X[i][2] for i in range(len(X)) if Y[i] == 0]
	SQ_X = [TSNE_X[i][0] for i in range(len(X)) if Y[i] == 1]
	SQ_Y = [TSNE_X[i][1] for i in range(len(X)) if Y[i] == 1]
	SQ_Z = [TSNE_X[i][2] for i in range(len(X)) if Y[i] == 1]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(AD_X, AD_Y, AD_Z, c='g', marker='o')
	ax.scatter(SQ_X, SQ_Y, SQ_Z, c='r', marker='^')
	plt.title('Dimensionality reduction with t-SNE (3D)')
	plt.show()
	"""
	index = []
	for i in range(len(features)):
		if 'let-7' in features[i]:
			index += [i]

	
	for i in index:
		AD = [X[j][i] for j in range(len(X)) if Y[j] == 0]
		SQ = [X[j][i] for j in range(len(X)) if Y[j] == 1]
		plt.hist([AD,SQ], label = ['AD', 'SQ'])
		plt.legend()
		plt.show()
	


if __name__ == "__main__" :
    main()