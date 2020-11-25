"""
This is the python script for preparing structured data for final group 
project of DSE 511.

Author: Hairuilong Zhang

Date: 2020-11-24
"""

# Import modules
import sqlite3
import pandas as pd
import numpy as np
import datetime as dt
from imblearn.over_sampling import SMOTE
from collections import Counter

# Define a function to prepare the data
def get_data(path, balance=True):
    """
    This function takes two arguments, and returns orginal data and 
    oversampled new data.
    
    Input:
        path: 
            path to your database
        balance:
             True: do oversampling the minority class using SMOTE
             False: only use original data set
             
    Return:
        X_origin: original features, a pandas DataFrame                    
        y_origin: original target, a pandas Series
        X_over: oversampled features, a pandas DataFrame 
        y_over: oversampled target, a pandas Series
    """
    conn = sqlite3.connect(path)
    c = conn.cursor()
    # order needs single quote 'order'
    order = pd.read_sql_query("SELECT * FROM 'order';", conn)
    # read other tables
    account = pd.read_sql_query("SELECT * FROM account;", 
                                conn,parse_dates=['date'])
    card = pd.read_sql_query("SELECT * FROM card;", conn)
    client = pd.read_sql_query("SELECT * FROM client;", conn,
                               parse_dates=['birth_date'])
    disp = pd.read_sql_query("SELECT * FROM disp;", conn)
    district = pd.read_sql_query("SELECT * FROM district;", conn)
    loan = pd.read_sql_query("SELECT * FROM loan;", conn,
                             parse_dates=['date'])
    trans = pd.read_sql_query("SELECT * FROM trans;", conn, 
                              parse_dates=['date'])
    
    # merge
    features = loan.merge(account, on='account_id', how='left',
                          suffixes=('_loan','_acct'))
    features = features.merge(disp, on='account_id',how='left')
    features = features.merge(client, on='client_id',how='left',
                              suffixes=('_bank','_client'))
    
    # only OWNER can ask for a loan
    features = features[features['type']=='OWNER']
    
    # create a funtion to add account balance statistics 
    # for the N previous months before loan
    def addfeats_mon_bal(features, trans, N):
        
        # merge loan date to transaction data
        trans_accdate = pd.merge(trans,features[['account_id','date_loan']],
                                 on='account_id',how='inner')
        
        # calculate time difference between loan date and transaction date
        trans_accdate['date_diff'] = trans_accdate['date_loan'] - trans_accdate['date'] 
        
        # keep transactions that took place before load date
        trans_accdate = trans_accdate[trans_accdate['date_diff'] >= dt.timedelta(0)]
        
        # only keep transactions occuring within N months (N*30 days) 
        trans_accdate = trans_accdate[trans_accdate['date_diff'] <= dt.timedelta(N*30)]
        
        # aggregate by account id
        mon_bal = trans_accdate.groupby('account_id')['balance'].agg(['min','max','mean']).reset_index()
        mon_bal.rename(columns={'min':f'min{N}','max':f'max{N}','mean':f'mean{N}'}, inplace=True)
        
        # merge features with account balance statistics
        features = pd.merge(features,mon_bal,on='account_id',how='left')
        
        return features
    
    # call the function to append previous 1 to 6 months 
    features = addfeats_mon_bal(features, trans, 1)
    features = addfeats_mon_bal(features, trans, 2)
    features = addfeats_mon_bal(features, trans, 3)
    features = addfeats_mon_bal(features, trans, 4)
    features = addfeats_mon_bal(features, trans, 5)
    features = addfeats_mon_bal(features, trans, 6)
    
    # create the target (response) based on 'status', A/C = 1(good), B/D = 0(bad)
    features['status'] = np.where(features['status'].isin(['A','C']), 1, 0)
    features.rename(columns = {'status':'target'}, inplace=True)
    
    # drop meaningless columns (ids, type being all OWNER)
    cols = ['account_id','disp_id','client_id','loan_id','district_id_bank','district_id_client','type']
    features.drop(cols, 1, inplace=True)
    
    # translate 'frequency' into English
    features.loc[features['frequency']=='POPLATEK MESICNE','frequency'] = 'monthly'
    features.loc[features['frequency']=='POPLATEK TYDNE','frequency'] = 'weekly'
    features.loc[features['frequency']=='POPLATEK PO OBRATU','frequency'] = 'after_trans'
    
    # transfer categorical into dummy columns
    features = pd.get_dummies(features)
    
    # transfer date into integer
    features.date_acct = features.date_acct.astype('int')
    features.date_loan = features.date_loan.astype('int')
    features.birth_date = features.birth_date.astype('int')
    
    # split X and y
    X_origin = features.drop('target', 1)
    y_origin = features['target']
    
    print(f'Shape of orgininal data: {X_origin.shape}')
    print(f'Class distribution: {Counter(y_origin)}\n')
    
    if balance:
        oversample = SMOTE(random_state=42)
        X_over, y_over = oversample.fit_resample(X=X_origin, y=y_origin)
        print(f'Shape of oversampled data: {X_over.shape}')
        print(f'Class distribution: {Counter(y_over)}')
    else:
        pass
    
    return X_origin, y_origin, X_over, y_over


if __name__ == "__main__":
    X_origin, y_origin, X_over, y_over = get_data('financial.db')

    
