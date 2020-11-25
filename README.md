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

    import prepare_data
    X_origin, y_origin, X_over, y_over = prepare_data.get_data('path/to/your/database/')

- X contains features and y is the target variable (1: good loan, 0: bad loan)
