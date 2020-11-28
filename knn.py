"""
File: knn.py
Author: verolero86
Date: November 28, 2020

Description:
This Python code will be used to apply the k-nearest neighbors (kNN) 
algorithm to the loan classification problem.
The code relies on scikit-learn (sklearn).
Matches svm.py workflow.
"""

# Import scikit-learn modules
from sklearn.neighbors import kNeighborsClassifier
from sklearn import metrics

# Import module to prepare data
import prepre_data

# Load data
X_origin, y_origin, X_over, y_over = prepare_data.get_data('financial.db')

def run_classifier(X_origin, y_origin, X_over, y_over, k):
  """
  
  Args:
    X_origin: original features, a pandas DataFrame                    
    y_origin: original target, a pandas Series
    X_over: oversampled features, a pandas DataFrame 
    y_over: oversampled target, a pandas Series
    k: number of neighbors to use for classification
    
  Returns:
    y_pred : prediction on the original test dataset, ndarray
    y_pred1 : prediction on the oversampled test dataset, ndarray
  
  """
  
  #Normalize the dataset
	X_origin = StandardScaler().fit_transform(X_origin)
	X_over = StandardScaler().fit_transform(X_over)
  
  #Randomly split training and testing data from the original dataset
	X_train, X_test, y_train, y_test = train_test_split(X_origin, y_origin, test_size=0.33,random_state=21)

  knn = kNeighborsClassifier(k)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  score = metrics.accuracy_score(y_test, y_pred)
  print(f"Accuracy on original data set using {k} neighbors is: {score}")
  
  #Randomly split training and testing data from the oversampled dataset
	X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.33,random_state=21)  
  
  knn1 = kNeighborsClassifier(k)
  knn1.fit(X_train, y_train)
  y_pred1 = knn1.predict(X_test)
  score1 = metrics.accuracy_score(y_test, y_pred1)
  print(f"Accuracy on original data set using {k} neighbors is: {score1}")
  
  return y_pred, y_pred1

if __name__ == "__main__":
	y_pred, y_pred1 = run_classifier(X_origin, y_origin, X_over, y_over,1)
  
