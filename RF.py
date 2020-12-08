#!/usr/bin/env python
# coding: utf-8

# In[11]:


import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

X_origin, y_origin, X_over, y_over = prepare_data.get_data('financial.db')

def run_randomforest(X_origin, y_origin, X_over, y_over):
    """Args:
        X_origin: original features, a pandas DataFrame                    
        y_origin: original target, a pandas Series
        X_over: oversampled features, a pandas DataFrame 
        y_over: oversampled target, a pandas Series

    Returns:
        y_pred : prediction on the original test dataset, ndarray
        y_pred1 : prediction on the oversampled test dataset, ndarray
    """
    X_train, X_test, y_train, y_test = train_test_split(X_origin, y_origin, test_size=0.33,random_state=21)

    clf=RandomForestClassifier(n_estimators=1000)

    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    print("Accuracy for original data:",metrics.accuracy_score(y_test, y_pred))
    print("Classification report for RF classifier on the original dataset:", classification_report(y_test, y_pred))

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_over, y_over, test_size=0.33,random_state=21)

    clf2=RandomForestClassifier(n_estimators=1000)

    clf2.fit(X_train2,y_train2)

    y_pred2=clf2.predict(X_test2)

    print("Accuracy for oversampled data:",metrics.accuracy_score(y_test2, y_pred2))
    print("Classification report for RF classifier on the oversampled dataset:", classification_report(y_test2, y_pred2))
    
    return y_pred, y_pred2

if __name__ == "__main__":
    y_pred, y_pred2 = run_randomforest(X_origin, y_origin, X_over, y_over)


# In[ ]:




