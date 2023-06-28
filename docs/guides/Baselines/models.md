# Baselines.models

Author: Shirley Zhang

Date: June 2023

[Source code](/src/Baselines/models.py)

## mean_baseline

A class to predict the mean of the training set for one y-variable. 
    
Allows the option of computing three different types of means: 

1. 'overall_mean': predicts the mean of the entire training set. 
2. 'latlon_mean': predicts the mean of each latitude and longitude point in the training set. 
3. 'latlonmonth_mean': predicts the mean of each latitude and longitude point for each month in the training set. 

The class handles training and predictions. 

### Parameters 

None. 

### Attributes 

- `y_var` : str - The y-variable to be predicted.
- `overall_mean` : numpy array - The overall mean for y_var (set during fit()).
- `latlon_mean` : numpy array - The latitude-longitude mean for y_var (set during fit()).
- `latlonmonth_mean` : numpy array - The latitude-longitude-month mean for y_var (set during fit()).

### Methods 

- `fit(train_ds)` - Fits the model by computing all three types of means using the training dataset.
- `predict(test_ds, method, return_ds=False)` - Returns/predicts the specified mean either as an array or as a new data variable in a copy of test_ds. 

## OLS_baseline 

A class to build a simple ordinary least squares (OLS) regression model for one y-variable. 
    
For a given x-variable and y-variable, the class models the following OLS formula: y ~ x. OLS is implemented using the 'LinearRegression' class from 'sklearn.linear_model'. This class fits a linear model through minimizing the residual sum of squares between the observed and predicted targets in  the given dataset. 

Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html 

### Parameters 

None. 

### Attributes 

- `y_var` : str - The y-variable to be predicted. 
- `overall_mean` : numpy array - The overall mean for y_var (set during fit()).
- `latlon_mean` : numpy array - The latitude-longitude mean for y_var (set during fit()).
- `latlonmonth_mean` : numpy array - The latitude-longitude-month mean for y_var (set during fit()).

### Methods 

- `fit(train_ds)` - Fits the model by computing all three types of means using the training dataset.
- `predict(test_ds, method, return_ds=False)` - Returns/predicts the specified mean either as an array or as a new data variable in a copy of test_ds.