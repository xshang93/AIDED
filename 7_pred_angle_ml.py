#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:39:03 2023

@author: xiao
"""
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib


# load data
data_ori = pd.read_csv('/home/xiao/projects/DED/BO_processing/Final_data/surfaces/2d_dataset_pix.csv')
# data_ori = pd.read_csv('/home/xiao/Downloads/ANN_Raw_Data.csv', index_col=0)

# data preprocessing
# data_ori=data_ori[data_ori['bad_data?']!=1]
# data_ori=data_ori[data_ori['index']!=259]

data = data_ori[[
    'Width',
                 'Power',
                 'Speed',
                  'hs',
                 'Angle',
                 'Height',
                 'Depth'
                 ]]
data = data.dropna()
corr = data.corr()

plt.figure(figsize=(10,10))
heatmap = sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns, annot=True)
plt.savefig("/home/xiao/Dropbox/UofT/Project_docs/DED/Process_opt/figures/supplementary_figures/heatmap_hs2angle.svg", format='svg')


# # X = data[['Power','Speed','rpm','Width']]
X = data[[
    'Width',
                 # 'Power',
                 'Speed',
                  'hs',
                 'Height',
                 # 'Depth'
    ]]
# corr_X = X.corr()
# sns.heatmap(corr_X,xticklabels=corr_X.columns,yticklabels=corr_X.columns)
# # Looks like Power and Width are colinear. Keep both first and see. If not working well drop one.

y = data[[
    'Angle',
          ]]
# corr_y_d = X.corrwith(y['Dilution'])
# corr_y_h = X.corrwith(y['Height'])

# poly = PolynomialFeatures(degree=3, include_bias=False)
# poly_X = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Create the parameter grid for hp tuning for rf
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [300],
#     'max_features': [1],
#     'min_samples_leaf': [1],
#     'min_samples_split': [2],
#     'n_estimators': [1200]
# }

cv = True
model = 'nn'
if cv==False:
    
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # fit with only training data
    model = MLPRegressor(
        hidden_layer_sizes=(20,5),
        alpha=0.1,
        learning_rate='adaptive',
        max_iter=100000)
    model.fit(X_train_std,y_train)
    y_pred = model.predict(X_test_std)
    r2 = r2_score(y_test,y_pred)
    r2_train = r2_score(y_train,model.predict(X_train_std))
    fig1,ax1 = plt.subplots()
    ax1.set_title("y_true vs y_pred")
    ax1.set_xlabel("y_true")
    ax1.set_ylabel("y_pred")
    ax1.scatter(y_test,y_pred)
    ax1.text(0.6,0.25,'r2 = {:.2f}'.format(r2))
    ax1.plot([10,50],[10,50])
    print('test r2 is {0}'.format(r2))
    print('training r2 is {0}'.format(r2_train))
    
elif model=='nn':
    sc = StandardScaler()
    model = MLPRegressor(random_state=42,max_iter=100000)
    pl = Pipeline(steps=[
        ('preprocessor',sc),
        ('estimater',model)
        ])
    # Create the parameter grid for hp tuning for ann
    param_grid = {
        'estimater__hidden_layer_sizes': [
            # (8),
            # (16),
                                          # (32),
                                          # (48),
                                          # (20,5),
                                           (16,32,16),
                                           # (16,32,48,16),
                                           # (16,32,48,32,16),
                                          # (16,128,16)
                                          ], 
        'estimater__activation': ['relu'],
        'estimater__solver': ['adam'],
        'estimater__alpha': [
            # 100,
            10,
            # 1,
            # 0.1
            ],
        # 'estimater__regressor__learning_rate_init': [
        #     0.0001,
        #     0.001,
        #     0.01,
        #     0.1
        #     ],
        'estimater__learning_rate': ['constant'],
    }
    grid_search = GridSearchCV(pl, param_grid = param_grid, 
                               cv = KFold(n_splits=5,shuffle=True, random_state=1), 
                              scoring = ['r2','neg_mean_absolute_percentage_error'], 
                              n_jobs = 1, verbose = 3, refit = 'r2')
    # fit with full dataset
    grid_search.fit(X_train,np.ravel(y_train))
    model = grid_search.best_estimator_
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    r2_val_mean = grid_search.cv_results_['mean_test_r2'][0]
    
    fig1,ax1 = plt.subplots()
    ax1.set_title("y_true vs y_pred")
    ax1.set_xlabel("y_true")
    ax1.set_ylabel("y_pred")
    ax1.scatter(y_test,y_pred)
    ax1.text(0.6,1.5,'r2 = {:.2f}'.format(r2))
    ax1.text(0.6,0.15,'r2_val_mean = {:.2f}'.format(r2_val_mean))
    ax1.plot([10,50],[10,50])
    print(grid_search.best_params_)

    print('test r2 is {0}'.format(r2))
    print('CV mean r2 is {0}'.format(r2_val_mean))
    
    # Convert the cv_results_ to a DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    # Save the DataFrame to a CSV file
    results_df.to_csv('./trained_models/grid_search_results_learningrate_angle.csv', index=False)
    
elif model=='lr':
    sc = StandardScaler()
    model = Ridge(random_state=42,max_iter=100000)
    pl = Pipeline(steps=[
        ('preprocessor',sc),
        ('estimater',model)
        ])
    # Create the parameter grid for hp tuning for ann
    param_grid = {
        'estimater__alpha': [0.0001,0.001,0.01,0.1,1,10],
    }
    grid_search = GridSearchCV(pl, param_grid = param_grid, 
                               cv = KFold(n_splits=5,shuffle=True, random_state=1), 
                              scoring = ['r2','neg_mean_absolute_percentage_error'], 
                              n_jobs = 1, verbose = 3, refit = 'r2')
    # fit with full dataset
    grid_search.fit(X_train,np.ravel(y_train))
    model = grid_search.best_estimator_
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    r2_val_mean = grid_search.cv_results_['mean_test_r2'][0]
    
    fig1,ax1 = plt.subplots()
    ax1.set_title("y_true vs y_pred")
    ax1.set_xlabel("y_true")
    ax1.set_ylabel("y_pred")
    ax1.scatter(y_test,y_pred)
    ax1.text(0.6,1.5,'r2 = {:.2f}'.format(r2))
    ax1.text(0.6,0.15,'r2_val_mean = {:.2f}'.format(r2_val_mean))
    ax1.plot([10,50],[10,50])
    print(grid_search.best_params_)

    print('test r2 is {0}'.format(r2))
    print('CV mean r2 is {0}'.format(r2_val_mean))
elif model=='rf':
    sc = StandardScaler()
    model = RandomForestRegressor(random_state=42)
    pl = Pipeline(steps=[
        ('preprocessor',sc),
        ('estimater',model)
        ])
    # Create the parameter grid for hp tuning for ann
    param_grid = {
        'estimater__n_estimators': [2000],
    }
    grid_search = GridSearchCV(pl, param_grid = param_grid, 
                               cv = KFold(n_splits=5,shuffle=True, random_state=1), 
                              scoring = ['r2','neg_mean_absolute_percentage_error'], 
                              n_jobs = 1, verbose = 3, refit = 'r2')
    # fit with full dataset
    grid_search.fit(X_train,np.ravel(y_train))
    model = grid_search.best_estimator_
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    r2_val_mean = grid_search.cv_results_['mean_test_r2'][0]
    
    fig1,ax1 = plt.subplots()
    ax1.set_title("y_true vs y_pred")
    ax1.set_xlabel("y_true")
    ax1.set_ylabel("y_pred")
    ax1.scatter(y_test,y_pred)
    ax1.text(0.6,1.5,'r2 = {:.2f}'.format(r2))
    ax1.text(0.6,0.15,'r2_val_mean = {:.2f}'.format(r2_val_mean))
    ax1.plot([10,50],[10,50])
    print(grid_search.best_params_)

    print('test r2 is {0}'.format(r2))
    print('CV mean r2 is {0}'.format(r2_val_mean))

# Save testing data for future use
X_test.to_csv('./trained_models/hs2angle_X_test.csv')
np.save('./trained_models/hs2angle_y_test',y_test)

# save trained model for use
joblib.dump(model,'./trained_models/hs2angle.pkl')