# README
 
This repository contains the final group project for DSE 511 for Team Lancaster Barnstormers.

## Team Members
- Tom Allemeier
- Inzamam Haque 
- Verónica G. Melesse Vergara
- Harry Zhang

## Project Goal
Deeper dive into the data from the individual projects by performing some statistics and machine learning to try and extract more patterns from the data.

## Project Plan
- Prepare both oversampled and original data
- Train kNN, SVM, and RF repectively on two datasets and compare
- Write final report

## Instructions of code
- Reading data by using

```python

import prepare_data
X_origin, y_origin, X_over, y_over = prepare_data.get_data('path/to/your/database/')

```

- X contains features and y is the target variable (1: good loan, 0: bad loan)

- Running SVM classifier

```python

import svm
clf0, X_test0, y_test0, clf1, X_test1, y_test1 = svm.run_svm(X_origin, y_origin, X_over, y_over)

```

- Running kNN classifier with the default k=1:
```python

python3 knn.py

```

- Running kNN classifier with a different value of k:
```python

import knn
clf0, X_test0, y_test0, clf1, X_test1, y_test1 = knn.run_knn(X_origin, y_origin, X_over, y_over, 1)

```

- Running RF classifier

```python
import RF
clf0, X_test0, y_test0, clf1, X_test1, y_test1 = RF.run_randomforest(X_origin, y_origin, X_over, y_over)

```

- Running the overall excecution script in terminal
  This prints out performance of all the classifiers and saves figures for ROC curve.
  
```python
python excecution.py
```