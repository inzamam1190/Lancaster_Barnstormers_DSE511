"""
Python script to run SVM model on the original and oversampled dataset 
generated by prepare_data.py script

Author: Inzamam Haque

Date: 11/25/2020
"""
#Importing necessary packages

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import prepare_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

#Getting original and oversampled data
X_origin, y_origin, X_over, y_over = prepare_data.get_data('financial.db')

def plot_decision_boundary(X,y,title:str):
	
	""" Plot decision boundary after dimensionality reduction using PCA

    Args:
        X: Features, a pandas DataFrame                   
        y: Target, a pandas Series
        title: 'Original'/'oversampled', a string
		
    Returns:
        decision boundary of PCA-reduced dataset 
    """
	pca = PCA(n_components=2)
	Xreduced = pca.fit_transform(X)

	def make_meshgrid(x, y, h=.02):
		x_min, x_max = x.min() - 1, x.max() + 1
		y_min, y_max = y.min() - 1, y.max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
		return xx, yy

	def plot_contours(ax, clf, xx, yy, **params):
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		out = ax.contourf(xx, yy, Z, **params)
		return out

	model = SVC(kernel='rbf', gamma='auto', C=3.0) #RBF kernel
	clf = model.fit(Xreduced, y)

	fig, ax = plt.subplots()

	# Set-up grid for plotting.
	X0, X1 = Xreduced[:, 0], Xreduced[:, 1]
	xx, yy = make_meshgrid(X0, X1)

	plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
	ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
	ax.set_ylabel('PC2')
	ax.set_xlabel('PC1')
	ax.set_xticks(())
	ax.set_yticks(())
	ax.set_title(f'Decison surface of {title} dataset using PCA')
	ax.legend()
	plt.show() 
	
	return None

def run_classifier(X_origin, y_origin, X_over, y_over):
    
	""" Run SVM classifier on the original and oversampled datasets

    Args:
        X_origin: original features, a pandas DataFrame                    
        y_origin: original target, a pandas Series
        X_over: oversampled features, a pandas DataFrame 
        y_over: oversampled target, a pandas Series
		
    Returns:
        y_pred : prediction on the original test dataset, ndarray
		y_pred1 : prediction on the oversampled test dataset, ndarray
    """
	
	#Normalize the dataset
	X_origin = StandardScaler().fit_transform(X_origin)
	X_over = StandardScaler().fit_transform(X_over)

	#Randomly split training and testing data from the original dataset
	X_train, X_test, y_train, y_test = train_test_split(X_origin, y_origin, test_size=0.33,random_state=21)

	#Create a svm Classifier
	clf = svm.SVC(kernel='rbf', gamma='auto', C=3.0) #RBF kernel

	#Train the model using the training sets
	clf.fit(X_train, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test)

	#Display classification summary report
	print('\nClassification report of SVM classifier for the original dataset\n')
	print(classification_report(y_test, y_pred))

	roc = roc_auc_score(y_test,y_pred)
	print(f'ROC-AUC score of the SVM classifier for the original dataset: {roc}')

	print('\nConfusion matrix of the SVM classifier for the original dataset:\n')
	print(confusion_matrix(y_test,y_pred))


	#Randomly split training and testing data from the oversampled dataset
	X_train1, X_test1, y_train1, y_test1 = train_test_split(X_over, y_over, test_size=0.33,random_state=21)

	#Train the model using the training sets
	clf.fit(X_train1, y_train1)

	#Predict the response for test dataset
	y_pred1 = clf.predict(X_test1)

	#Display classification summary report for the oversampled dataset
	print('\nClassification report of SVM classifier for the oversampled dataset\n')
	print(classification_report(y_test1, y_pred1))

	roc2 = roc_auc_score(y_test1,y_pred1)
	print(f'ROC-AUC score of the SVM classifier for the oversampled dataset: {roc2}')

	print('\nConfusion matrix of the SVM classifier for the oversampled dataset:\n')
	print(confusion_matrix(y_test1,y_pred1))
	
	_ = plot_decision_boundary(X_origin, y_origin, title='original')
	_ = plot_decision_boundary(X_over, y_over, title='oversampled')

	return y_pred, y_pred1

if __name__ == "__main__":
	y_pred, y_pred1 = run_classifier(X_origin, y_origin, X_over, y_over)




