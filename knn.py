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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

def run_knn(X_origin, y_origin, X_over, y_over, k):
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

    knn = KNeighborsClassifier(k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy on original data set using {k} neighbor(s) is: {score}")
    
    print(f'\nClassification report of kNN(k={k}) classifier for the original dataset\n')
    print(classification_report(y_test, y_pred))

    roc = roc_auc_score(y_test,y_pred)
    print(f'ROC-AUC score of the kNN(k={k}) classifier for the original dataset: {roc}')

    print(f'\nConfusion matrix of the kNN(k={k}) classifier for the original dataset:\n')
    print(confusion_matrix(y_test,y_pred))
    
    #Randomly split training and testing data from the oversampled dataset
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_over, y_over, test_size=0.33,random_state=21)  

    knn1 = KNeighborsClassifier(k)
    knn1.fit(X_train1, y_train1)
    y_pred1 = knn1.predict(X_test1)
    score1 = metrics.accuracy_score(y_test1, y_pred1)
    print("================================================================================")
    print(f"Accuracy on oversampled data set using {k} neighbor(s) is: {score1}")
    
    print(f'\nClassification report of kNN(k={k}) classifier for the oversampled dataset\n')
    print(classification_report(y_test1, y_pred1))

    roc1 = roc_auc_score(y_test1,y_pred1)
    print(f'ROC-AUC score of the kNN(k={k}) classifier for the oversampled dataset: {roc1}')

    print(f'\nConfusion matrix of the kNN(k={k}) classifier for the oversampled dataset:\n')
    print(confusion_matrix(y_test1,y_pred1))
    
    # return these for easily plotting AUC curve
    return knn,X_test,y_test, knn1,X_test1,y_test1
    
if __name__ == "__main__":
    # Import module to prepare data
    import prepare_data

    # Load data
    X_origin, y_origin, X_over, y_over = prepare_data.get_data('financial.db')
    knn0,X_test0,y_test0, knn1,X_test1,y_test1 = run_knn(X_origin, y_origin, X_over, y_over,1)
  
