"""
File: excution.py
Author: Hairuilong Zhang
Date: 2020-12-08

Description:

This is the driver script for calling all algorithms and generating a plot
of ROC curves for six different models.
"""


import prepare_data
import svm
import RF
import knn
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

def main():
    X_origin, y_origin, X_over, y_over = prepare_data.get_data('financial.db')

    k = input("Input the value of k for kNN model: ")
    knn_ts = time.time()
    knn0,X_test0,y_test0, knn1,X_test1,y_test1 = knn.run_knn(X_origin, y_origin, X_over, y_over, int(k))
    knn_te = time.time()
    print('\n')

    rf_ts = time.time()
    rf0,X_test_rf0,y_test_rf0, rf1, X_test_rf1,y_test_rf1 = RF.run_randomforest(X_origin, y_origin, X_over, y_over)
    rf_te = time.time()
    print('\n')

    svm_ts = time.time()
    svm0,X_test_svm0,y_test_svm0, svm1, X_test_svm1,y_test_svm1 = svm.run_svm(X_origin, y_origin, X_over, y_over)
    svm_te = time.time()
    
    print("EXECUTION TIME")
    print(f"knn:\t{knn_te - knn_ts} seconds")
    print(f"RF:\t{rf_te - rf_ts} seconds")
    print(f"SVM:\t{svm_te - svm_ts} seconds")


    plt.figure(figsize=(9,6))
    knn0_disp = plot_roc_curve(knn0,X_test0,y_test0)
    rf0_disp = plot_roc_curve(rf0,X_test_rf0,y_test_rf0,ax=knn0_disp.ax_)
    svm0_disp = plot_roc_curve(svm0,X_test_svm0,y_test_svm0,ax=knn0_disp.ax_)
    plt.title("ROC curve for original dataset")
    plt.tight_layout()
    plt.savefig('fig0.png')

    plt.figure(figsize=(9,6))
    knn1_disp = plot_roc_curve(knn1,X_test1,y_test1)
    rf1_disp = plot_roc_curve(rf1,X_test_rf1,y_test_rf1,ax=knn1_disp.ax_)
    svm1_disp = plot_roc_curve(svm1,X_test_svm1,y_test_svm1,ax=knn1_disp.ax_)
    plt.title("ROC curve for oversampled dataset")
    plt.tight_layout()
    plt.savefig('fig1.png')

    #cm = confusion_matrix(y_origin, y_test0)
    #class_label = ["0", "1"]
    #df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    #sns.heatmap(df_cm, annot=True, fmt='d')
    #plt.title("Confusion Matrix for kNN - Original Data set")
    #plt.xlabel("Predicted Label")
    #plt.ylabel("True Label")
    #plt.savefig('confusion_knn_original.png')
    
    return None
    

if __name__ == "__main__":
    main()
