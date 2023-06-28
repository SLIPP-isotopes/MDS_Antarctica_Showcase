# src/Baselines/tests/test_ols_baseline.py
#########################
# Purpose: ... 
# Author: Shirley Zhang 
# Date: June 2023

# Import statements 
import sys 
sys.path.insert(1, '../../../src/') 
from Baselines.models import * 
import pandas as pd
import statsmodels.api as sm

############################## Global variables 
toy_x_var = 'd18O_pr' 
toy_y_var = 'tmp2m' 
toy_train_ds = xr.open_dataset('toy_train_ds.nc')
toy_valid_ds = xr.open_dataset('toy_valid_ds.nc')

############################## Tests for __init__() 
def test_init():
    """Test that creating an OLS_baselines() object does not return None, and that self.x_vars and self.y_vars are initialized properly as strings"""

    # Create a model object 
    toy_model = OLS_baseline(x_var=toy_x_var, y_var=toy_y_var) 
    
    assert toy_model is not None, 'OLS_baseline() object not created correctly (it is None)' 
    assert isinstance(toy_model.x_var, str), f'model.x_var attribute is the wrong type (should be a string, not {type(toy_model.x_var)})' 
    assert isinstance(toy_model.y_var, str), f'model.y_var attribute is the wrong type (should be a string, not {type(toy_model.y_var)})' 
    assert toy_model.x_var == toy_x_var, f'model.x_var attribute was initialized incorrectly (it should be "{toy_x_var}", but is "{toy_model.x_var}")' 
    assert toy_model.y_var == toy_y_var, f'model.y_var attribute was initialized incorrectly (it should be "{toy_y_var}", but is "{toy_model.y_var}")' 

############################## Tests for fit() 
def test_fit():
    """Test that calling the fit() method will return None""" 

    # Create a model object and fit 
    toy_model = OLS_baseline(x_var=toy_x_var, y_var=toy_y_var) 
    fit_return = toy_model.fit(train_ds=toy_train_ds)

    assert fit_return is None, f'fit() should return None, instead a {type(fit_return)} is returned'

def test_fit_ols_model():
    """Test that calling the fit() method will initialize self.ols_model and that it's an sklearn.linear_model._base.LinearRegression object"""

    # Create a model object and fit 
    toy_model = OLS_baseline(x_var=toy_x_var, y_var=toy_y_var) 
    toy_model.fit(train_ds=toy_train_ds)
    
    assert toy_model.ols_model is not None, f'model.ols_model attribute should not be None after calling fit()'
    assert isinstance(toy_model.ols_model, LinearRegression), f'model.ols_model should be of type sklearn.linear_model._base.LinearRegression, not {type(toy_model.ols_model)}'
    
def test_fit_ols_params():
    """Test that calling the fit() method will initialize self.slope, self.intercept, and self.r_sq; also test that these parameters were calculated correctly by cross-checking using the statsmodels.api and pandas libraries"""
    
    # Create a model object and fit 
    toy_model = OLS_baseline(x_var=toy_x_var, y_var=toy_y_var) 
    toy_model.fit(train_ds=toy_train_ds)

    # Create an OLS model with statsmodels.api and pandas
    toy_train_df = toy_train_ds.to_dataframe()
    X = toy_train_df[toy_x_var]
    Y = toy_train_df[toy_y_var]
    X = sm.add_constant(X)
    stats_model = sm.OLS(Y,X)
    stats_model_results = stats_model.fit()
    stats_model_slope=stats_model_results.params[toy_x_var]
    stats_model_intercept=stats_model_results.params.const
    stats_model_r_sq=stats_model_results.rsquared

    # Slope 
    assert toy_model.slope is not None, f'model.slope attribute should not be None after calling fit()'
    assert isinstance(toy_model.slope, float), f'model.slope should be of type float, not {type(toy_model.slope)}'
    assert round(toy_model.slope, 5) == round(stats_model_slope, 5), f'model.slope does not seem to be calculated correctly \
        \n\nmodel.slope: {round(toy_model.slope, 5)} \
        \nSlope with statsmodels.api: {round(stats_model_slope, 5)}'
    # Intercept 
    assert toy_model.intercept is not None, f'model.intercept attribute should not be None after calling fit()'
    assert isinstance(toy_model.intercept, float), f'model.intercept should be of type float, not {type(toy_model.intercept)}'
    assert round(toy_model.intercept, 5) == round(stats_model_intercept, 5), f'model.intercept does not seem to be calculated correctly \
        \n\nmodel.intercept: {round(toy_model.intercept, 5)} \
        \nSlope with statsmodels.api: {round(stats_model_intercept, 5)}'
    # R-squared 
    assert toy_model.r_sq is not None, f'model.r_sq attribute should not be None after calling fit()'
    assert isinstance(toy_model.r_sq, float), f'model.r_sq should be of type float, not {type(toy_model.r_sq)}'
    assert round(toy_model.r_sq, 5) == round(stats_model_r_sq, 5), f'model.r_sq does not seem to be calculated correctly \
        \n\nmodel.r_sq: {round(toy_model.r_sq, 5)} \
        \nSlope with statsmodels.api: {round(stats_model_r_sq, 5)}'

############################## Tests for predict() 
def test_predict():
    """Test that calling predict(return_ds=False) will return an np.ndarray, and that calling predict(return_ds=True) will return an xarray dataset with a new data variable""" 

    # Create a model object, fit, and predict  
    toy_model = OLS_baseline(x_var=toy_x_var, y_var=toy_y_var)  
    toy_model.fit(train_ds=toy_train_ds)
    nparray_predict_return = toy_model.predict(test_ds=toy_valid_ds, return_ds=False)
    ds_predict_return = toy_model.predict(test_ds=toy_valid_ds, return_ds=True) 
    new_data_var = 'pred_' + toy_y_var

    assert isinstance(nparray_predict_return, np.ndarray), f'Calling model.predict(return_ds=False) should return an np.ndarray, not {type(nparray_predict_return)}'
    assert isinstance(ds_predict_return, xr.core.dataset.Dataset), f'Calling model.predict(return_ds=True) should return an xarray dataset, not {type(ds_predict_return)}' 
    assert new_data_var in ds_predict_return.data_vars, f'The new data variable "{new_data_var}" does not exist in the new dataset when model.predict(return_ds=True) is called'
    
def test_predict_predictions(): 
    """Test that predictions were calculated correctly by cross-checking using the statsmodels.api and pandas libraries"""

    # Create a model object, fit, and predict  
    toy_model = OLS_baseline(x_var=toy_x_var, y_var=toy_y_var)  
    toy_model.fit(train_ds=toy_train_ds)
    pred_ols = toy_model.predict(test_ds=toy_valid_ds, return_ds=False)
    pred_ols_arr = pred_ols.flatten()

    # Create an OLS model and predict with statsmodels.api and pandas
        # Reference: https://stackoverflow.com/questions/58882719/how-to-predict-data-using-linearregression-using-linear-model-ols-from-statsmode
    toy_train_df = toy_train_ds.to_dataframe()
    toy_valid_df = toy_valid_ds.to_dataframe()
    X_train = toy_train_df[toy_x_var]
    y_train = toy_train_df[toy_y_var]
    X_test = toy_valid_df[toy_x_var]
    X2 = sm.add_constant(X_train)
    est = sm.OLS(y_train, X2).fit()
    X_test = sm.add_constant(X_test) 
    y_test_predicted = est.predict(X_test) 
    y_test_predicted_arr = np.array(y_test_predicted)

    assert np.array_equal(np.round(pred_ols_arr, 5), np.round(y_test_predicted_arr, 5)), f'The predicted "{toy_y_var}" values do not seem to be calculated correctly; they differ from predictions using statsmodels.api'
    
############################## TEST THE TESTS 
test_init() 
test_fit()
test_fit_ols_model()
test_fit_ols_params()
test_predict()
test_predict_predictions()
print('All tests passed! 🌈')