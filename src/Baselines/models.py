# src/Baselines/models.py
# ########################
# Purpose: This script contains the code of two classes: 
#       1. mean_baseline
#       2. OLS_baseline 
#    These classes are used to generate baseline predictions through fitting on preprocessed training data, and 
#    then predicting on preprocessed validation or testing data. See the /notebooks/baselines_demo.ipynb notebook 
#    for an in-depth tutorial on how to use these classes. 
# Author: Shirley Zhang 
# Date: June 2023

# Imports
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression

class mean_baseline():
    """A class to predict the mean of the training set for one y-variable. 
    
    Allows the option of computing three different types of means: 
        1. 'overall_mean': predicts the mean of the entire training set. 
        2. 'latlon_mean': predicts the mean of each latitude and longitude point in the training set. 
        3. 'latlonmonth_mean': predicts the mean of each latitude and longitude point for each month in the training set. 

    Attributes
    ----------
    y_var : str
        The y-variable to be predicted 
    overall_mean : numpy array 
        The overall mean for y_var (set during fit())
    latlon_mean : numpy array 
        The latitude-longitude mean for y_var (set during fit())
    latlonmonth_mean : numpy array 
        The latitude-longitude-month mean for y_var (set during fit())

    Methods 
    -------
    fit(train_ds)
        Fits the model by computing all three types of means using the training dataset   
    predict(test_ds, method, return_ds=False)
        Returns/predicts the specified mean either as an array or as a new data variable in a copy of test_ds 
    """
    
    def __init__(self, y_var):
        """Initialize the mean_baselines() object by passing in the y-variable to predict. 
        
        Parameters
        ----------
        y_var : str
            The y-variable to be predicted 

        Raises
        ------
        TypeError
            If y_var passed is not a string 
        """

        # Exception catching 
        if not isinstance(y_var, str):
            raise TypeError(f"y_var must be a string, not {type(y_var)}") 
        
        # Set attributes
        self.y_var = y_var
        self.overall_mean = None
        self.latlon_mean = None
        self.latlonmonth_mean = None 

    def fit(self, train_ds): 
        """Fit the mean_baselines() object.
        
        Takes in the training dataset and computes the three types of means. Saves them as attributes
        (self.overall_mean, self.latlon_mean, self.latlonmonth_mean). 
        
        Parameters
        ----------
        train_ds : xarray dataset
            The training dataset used to compute the means 
   
        Raises
        ------
        TypeError 
            If train_ds passed is not an xarray dataset 
        NameError
            If train_ds passed does not contain self.y_var as one of its data variables; 
            If train_ds passed does not have 'latitude', 'longitude', and 'time' in its dimensions  
        """ 

        # Exception catching 
        if not isinstance(train_ds, xr.core.dataset.Dataset):
            raise TypeError(f'train_ds must be an xarray dataset, not {type(train_ds)}') 
        if self.y_var not in train_ds.data_vars:
            raise NameError(f'The train_ds passed does not contain the y_var "{self.y_var}"') 
        if 'latitude' not in train_ds.dims:
            raise NameError(f'The train_ds passed must contain the dimension "latitude"') 
        if 'longitude' not in train_ds.dims:
            raise NameError(f'The train_ds passed must contain the dimension "longitude"') 
        if 'time' not in train_ds.dims:
            raise NameError(f'The train_ds passed must contain the dimension "time"') 
        
        # Calculate the overall mean of self.y_var, skipping NaNs in the calculation 
        overall_mean_ds = train_ds[self.y_var].mean(skipna=True) 
        self.overall_mean = overall_mean_ds.values
        
        # Calculate the mean for each latitude longitude point, over all time points 
        latlon_mean_ds = train_ds[self.y_var].groupby('latitude').mean(dim='time') 
        self.latlon_mean = latlon_mean_ds.values

        # Calculate the mean for each latitude longitude point per month
        latlonmonth_mean_ds = train_ds[self.y_var].groupby('time.month').mean(dim='time')
        self.latlonmonth_mean = latlonmonth_mean_ds.values

    def predict(self, test_ds, method, return_ds=False): 
        """Use the model to predict y-variable outputs.  
        
        Takes in the validation/test dataset and creates an array of the training mean computed. The array 
        has the same shape of the y-variable in the validation/test dataset. 
        
        Parameters 
        ---------- 
        test_ds : xarray dataset
            The test dataset for which y-values will be predicted 
        method : str 
            The type of mean to be returned; could either be 'overall_mean', 'latlon_mean', or 'latlonmonth_mean'
        return_ds : bool
            Specifies whether the predicted values should be saved as a new data variable and returned as a dataset; 
            default False which will return just the predicted values as an array; if True, the method will return a 
            copy of test_ds with the predicted values as 'pred_{method}_{y_var}' instead of an array 
        
        Raises
        ------
        TypeError 
            If test_ds passed is not an xarray dataset;
            If method passed is not a string;
            If return_ds passed is not a boolean
        NameError
            If test_ds passed does not contain self.y_var as one of its data variables;
            If test_ds passed does not have 'latitude', 'longitude', and 'time' in its dimensions;
        ValueError
            If self.overall_mean, self.latlon_mean, or self.latlonmonth_mean are None 
        AssertionError
            If the number of values in the 'time' dimension in test_ds is not divisible by 12 
        """

        # Exception catching 
        # TypeErrors
        if not isinstance(test_ds, xr.core.dataset.Dataset):
            raise TypeError(f'test_ds must be an xarray dataset, not {type(train_ds)}') 
        if not isinstance(method, str):
            raise TypeError(f'method must be a string, not {type(method)}') 
        if not isinstance(return_ds, bool):
            raise TypeError(f'return_ds must be a boolean, not {type(method)}') 
        # NameErrors
        if self.y_var not in test_ds.data_vars:
            raise NameError(f'The test_ds passed does not contain the y_var "{self.y_var}"') 
        if 'latitude' not in test_ds.dims:
            raise NameError(f'The test_ds passed must contain the dimension "latitude"') 
        if 'longitude' not in test_ds.dims:
            raise NameError(f'The test_ds passed must contain the dimension "longitude"') 
        if 'time' not in test_ds.dims:
            raise NameError(f'The test_ds passed must contain the dimension "time"') 
        # ValueErrors
        if (not isinstance(self.overall_mean, np.ndarray)):
            if (self.overall_mean == None):
                raise ValueError(f'self.overall_mean has not been initialized, make sure fit() was called before predict()')
        if (not isinstance(self.latlon_mean, np.ndarray)):
            if (self.latlon_mean == None):
                raise ValueError(f'self.latlon_mean has not been initialized, make sure fit() was called before predict()')
        if (not isinstance(self.latlonmonth_mean, np.ndarray)):
            if (self.latlonmonth_mean == None):
                raise ValueError(f'self.latlonmonth_mean has not been initialized, make sure fit() was called before predict()')
        # AssertionError
        assert (test_ds.dims['time'] % 12 == 0), f'The number of values in the "time" dimension ({test_ds.dims["time"]}) in test_ds is not divisible by 12'

        # Get the dimensions for the test dataset 
        time_dim = test_ds.dims['time'] 
        lat_dim = test_ds.dims['latitude'] 
        lon_dim = test_ds.dims['longitude'] 
        
        # Tile the overall mean 
        if method == 'overall_mean': 
            pred_arr_tiled = np.tile(self.overall_mean, reps=(time_dim, lat_dim, lon_dim)) 

        # Return the latlon mean 
        elif method == 'latlon_mean': 
            pred_arr_tiled = np.tile(self.latlon_mean, reps=(time_dim, 1, 1))

        # Return the latlonmonth mean
        elif method == 'latlonmonth_mean': 
            # Find the number of years 
            nyears = int(time_dim / 12)
            # Tile array for number of years
            pred_arr_tiled = np.tile(self.latlonmonth_mean, reps=(nyears, 1, 1))

        # Return either as an array or as a data variable saved in a new dataset
        if return_ds == False:
            return pred_arr_tiled       
        elif return_ds == True:
            # Make a copy of the dataset 
            new_test_ds = test_ds.copy() 
            # Save into 'new_test_ds'
            new_test_ds = new_test_ds.assign(
                    pred_feat = (['time', 'latitude', 'longitude'], pred_arr_tiled)
                ).rename(
                    {'pred_feat': f"pred_{self.y_var}"})
            return new_test_ds 

class OLS_baseline():
    """A class to build a simple ordinary least squares (OLS) regression model for one y-variable. 
    
    For a given x-variable and y-variable, the class models the following OLS formula: y ~ x. OLS is 
    implemented using the 'LinearRegression' class from 'sklearn.linear_model'. This class fits a linear
    model through minimizing the residual sum of squares between the observed and predicted targets in 
    the given dataset. 
        Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html 

    Attributes
    ----------
    x_var : str 
        The x-variable used to predict y 
    y_var : str
        The y-variable to be predicted 
    ols_model : LinearRegression 
        An instance of the LinearRegression model from Sklearn fitted with the given training dataset 
    slope : float 
        The estimated slope/coefficient in the linear model for the x-variable 
    intercept : float 
        The estimated independent term/y-intercept in the linear model 
    r_sq : float 
        The R^2 score, or the coefficient of determination of the prediction 

    Methods
    -------
    fit(train_ds)
        Fits a linear model by using data from the training dataset   
    predict(test_ds, return_ds=False) 
        Returns the predictions of the y-variable either as an array or as a new data variable in a copy of test_ds   
    """
    
    def __init__(self, x_var, y_var): 
        """Initialize the mean_baselines() object by passing in the x- and y-variables used in the linear model. 
        
        Parameters
        ----------
        x_var : str 
            The x-variable used to predict y 
        y_var : str
            The y-variable to be predicted 
  
        Raises
        ------
        TypeError
            If x_var or y_var passed is not a string 
        """

        # Exception catching 
        if not isinstance(x_var, str):
            raise TypeError(f"x_var must be a string, not {type(x_var)}") 
        if not isinstance(y_var, str):
            raise TypeError(f"y_var must be a string, not {type(y_var)}") 

        # Set global attributes 
        self.x_var = x_var
        self.y_var = y_var
        
        self.ols_model = None
        self.slope = None 
        self.intercept = None
        self.r_sq = None 
        
    def fit(self, train_ds):
        """Fit the OLS_baselines() object.
        
        Takes in the training dataset and fits a linear model. Saves the fitted model, the 
        slope, the intercept, and r-squared score as attributes. 
        
        Parameters
        ----------
        train_ds : xarray dataset 
            The training dataset used to compute the means 
  
        Raises
        ------
        TypeError 
            If train_ds passed is not an xarray dataset 
        NameError
            If train_ds passed does not contain self.x_var or self.y_var as one of its data variables; 
            If train_ds passed does not have 'latitude', 'longitude', and 'time' in its dimensions  
        """

        # Exception catching 
        if not isinstance(train_ds, xr.core.dataset.Dataset):
            raise TypeError(f'train_ds must be an xarray dataset, not {type(train_ds)}') 
        if self.x_var not in train_ds.data_vars:
            raise NameError(f'The train_ds passed does not contain the x_var "{self.x_var}"') 
        if self.y_var not in train_ds.data_vars:
            raise NameError(f'The train_ds passed does not contain the y_var "{self.y_var}"') 
        if 'latitude' not in train_ds.dims:
            raise NameError(f'The train_ds passed must contain the dimension "latitude"') 
        if 'longitude' not in train_ds.dims:
            raise NameError(f'The train_ds passed must contain the dimension "longitude"') 
        if 'time' not in train_ds.dims:
            raise NameError(f'The train_ds passed must contain the dimension "time"') 

        # Get x and y for the model 
        x = train_ds[self.x_var].values.flatten().reshape((-1, 1)) # flatten into (n_examples, 1)
        y = train_ds[self.y_var].values.flatten() # flatten into (n_examples,) 

        # Create a model and fit it 
        ols_model = LinearRegression(fit_intercept=True)
        ols_model.fit(x, y)
        
        # Save model and parameters 
        self.ols_model = ols_model
        self.slope = ols_model.coef_[0] 
        self.intercept = ols_model.intercept_
        self.r_sq = ols_model.score(x, y)

    def predict(self, test_ds, return_ds=False): 
        """Use the model to predict y-variable outputs.  
        
        Takes in the validation/test dataset and computes an array of the y-variable values predicted for the given x-variable values. The array has the same shape of the y-variable in the validation/test dataset. 
        
        Parameters 
        ---------- 
        test_ds : xarray dataset
            The test dataset for which y-values will be predicted 
        return_ds : bool 
            Specifies whether the predicted values should be saved as a new data variable and returned as a dataset; 
            default False which will return just the predicted values as an array; if True, the method will return a 
            copy of test_ds with the predicted values as 'pred_ols_{y_var}' instead of an array 
        
        Raises
        ------
        TypeError 
            If test_ds passed is not an xarray dataset;
            If return_ds passed is not a boolean
        NameError
            If test_ds passed does not contain self.x_var or self.y_var as one of its data variables;
            If test_ds passed does not have 'latitude', 'longitude', and 'time' in its dimensions;
        ValueError
            If self.ols_model, self.slope, self.intercept, or self.r_sq are None 
        """

        # Exception catching 
        # TypeErrors
        if not isinstance(test_ds, xr.core.dataset.Dataset):
            raise TypeError(f'test_ds must be an xarray dataset, not {type(train_ds)}') 
        if not isinstance(return_ds, bool):
            raise TypeError(f'return_ds must be a boolean, not {type(method)}') 
        # NameErrors
        if self.x_var not in test_ds.data_vars:
            raise NameError(f'The test_ds passed does not contain the x_var "{self.x_var}"')
        if self.y_var not in test_ds.data_vars:
            raise NameError(f'The test_ds passed does not contain the y_var "{self.y_var}"') 
        if 'latitude' not in test_ds.dims:
            raise NameError(f'The test_ds passed must contain the dimension "latitude"') 
        if 'longitude' not in test_ds.dims:
            raise NameError(f'The test_ds passed must contain the dimension "longitude"') 
        if 'time' not in test_ds.dims:
            raise NameError(f'The test_ds passed must contain the dimension "time"') 
        if self.ols_model == None:
            raise ValueError(f'self.ols_model has not been initialized, make sure fit() was called before predict()')
        if self.slope == None:
            raise ValueError(f'self.slope has not been initialized, make sure fit() was called before predict()')
        if self.intercept == None:
            raise ValueError(f'self.intercept has not been initialized, make sure fit() was called before predict()')
        if self.r_sq == None:
            raise ValueError(f'self.r_sq has not been initialized, make sure fit() was called before predict()')
            
        # Get x from test_ds, and drop NaNs
        x_test_flattened = test_ds[self.x_var].values.flatten()
        x_test_nonans = x_test_flattened[~np.isnan(x_test_flattened)].reshape((-1, 1)) 
        
        # Predict on test_ds 
        y_pred = self.ols_model.predict(x_test_nonans)
        
        # Put NaNs back in their original positions 
        nan_indices = np.isnan(x_test_flattened) # Find the indices of NaN values in the flattened array 
        y_pred_nans = np.empty_like(x_test_flattened)
        y_pred_nans[~nan_indices] = y_pred 
        y_pred_nans[nan_indices] = np.nan
        
        # Reshape the array back to the original shape
        y_pred_nans = y_pred_nans.reshape(test_ds[self.x_var].shape) 

        # Return either as an array or as a data variable saved in a new dataset 
        if return_ds == False:
            return y_pred_nans 
            
        elif return_ds == True:
            # Make a copy of the dataset 
            new_test_ds = test_ds.copy() 
            # Save into 'new_test_ds'
            new_test_ds = new_test_ds.assign(
                    pred_feat = (['time', 'latitude', 'longitude'], y_pred_nans)
                ).rename(
                    {'pred_feat': f"pred_{self.y_var}"})
            return new_test_ds     
