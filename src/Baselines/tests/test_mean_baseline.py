# src/Baselines/tests/test_mean_baseline.py
#########################
# Purpose: ... 
# Author: Shirley Zhang 
# Date: June 2023

# Import statements 
import sys 
sys.path.insert(1, '../../../src/') 
from Baselines.models import * 
import pandas as pd

############################## Global variables 
toy_y_var = 'tmp2m' 
toy_train_ds = xr.open_dataset('toy_train_ds.nc')
toy_valid_ds = xr.open_dataset('toy_valid_ds.nc')

############################## Tests for __init__() 
def test_init():
    """Test that creating a mean_baselines() object does not return None, and that self.y_vars is initialized properly as a string"""

    # Create a model object 
    toy_model = mean_baseline(y_var=toy_y_var) 
    
    assert toy_model is not None, 'mean_baseline() object not created correctly (it is None)'
    assert isinstance(toy_model.y_var, str), f'model.y_var attribute is the wrong type (should be a string, not {type(toy_model.y_var)})'
    assert toy_model.y_var == toy_y_var, f'model.y_var attribute was initialized incorrectly (it should be "{toy_y_var}", but is "{toy_model.y_var}")' 

############################## Tests for fit() 
def test_fit():
    """Test that calling the fit() method will return None""" 

    # Create a model object and fit 
    toy_model = mean_baseline(y_var=toy_y_var) 
    fit_return = toy_model.fit(train_ds=toy_train_ds)

    assert fit_return is None, f'fit() should return None, instead a {type(fit_return)} is returned'

def test_fit_calc_overall_mean(): 
    """Test that fitting a model will initialize self.overall_mean, that it's an np.ndarray, and that it was calculated correctly"""

    # Create a model object and fit 
    toy_model = mean_baseline(y_var=toy_y_var) 
    toy_model.fit(train_ds=toy_train_ds)

    # Calculate the overall mean using pandas
    toy_train_df = toy_train_ds.to_dataframe()
    pandas_overall_mean = toy_train_df[toy_y_var].mean()
    
    assert toy_model.overall_mean is not None, f'model.overall_mean attribute should not be None after calling fit()'
    assert isinstance(toy_model.overall_mean, np.ndarray), f'model.overall_mean should be of type np.ndarray, not {type(toy_model.overall_mean)}'
    assert toy_model.overall_mean.item() == pandas_overall_mean, f'model.overall_mean was not calculated correctly'

def test_fit_calc_latlon_mean():
    """Test that fitting a model will initialize self.latlon_mean, that it's an np.ndarray, and that it was calculated correctly"""

    # Create a model object and fit 
    toy_model = mean_baseline(y_var=toy_y_var) 
    toy_model.fit(train_ds=toy_train_ds)

    # Calculate the latlon mean using pandas
    toy_train_df = toy_train_ds.to_dataframe()
    pandas_latlon_mean = np.array(toy_train_df[toy_y_var].groupby(['latitude', 'longitude']).mean())
    
    assert toy_model.latlon_mean is not None, f'model.latlon_mean attribute should not be None after calling fit()'
    assert isinstance(toy_model.latlon_mean, np.ndarray), f'model.latlon_mean should be of type np.ndarray, not {type(toy_model.latlon_mean)}'
    assert np.array_equal(toy_model.latlon_mean.flatten(), pandas_latlon_mean), f'model.latlon_mean was not calculated correctly'

def test_fit_calc_latlonmonth_mean():
    """Test that fitting a model will initialize self.latlonmonth_mean, that it's an np.ndarray, and that it was calculated correctly"""

    # Create a model object and fit 
    toy_model = mean_baseline(y_var=toy_y_var) 
    toy_model.fit(train_ds=toy_train_ds)

    # Calculate the latlonmonth mean using pandas
    toy_train_df = toy_train_ds[toy_y_var].to_dataframe()
    toy_train_df.reset_index()['month'] = toy_train_df.reset_index()['time'].dt.month
    toy_train_df['month'] = list(toy_train_df.reset_index()['time'].dt.month)
    toy_train_df = toy_train_df.reset_index().set_index(['time', 'latitude', 'longitude', 'month']).groupby(['month', 'latitude', 'longitude']).mean()
    pandas_latlonmonth_mean = np.array(toy_train_df[toy_y_var])
        
    assert toy_model.latlonmonth_mean is not None, f'model.latlonmonth_mean attribute should not be None after calling fit()'
    assert isinstance(toy_model.latlonmonth_mean, np.ndarray), f'model.latlonmonth_mean should be of type np.ndarray, not {type(toy_model.latlonmonth_mean)}'
    assert np.array_equal(toy_model.latlonmonth_mean.flatten(), pandas_latlonmonth_mean), f'model.latlonmonth_mean was not calculated correctly'
        
############################## Tests for predict() 
def test_predict():
    """Test that calling predict(return_ds=False) will return an np.ndarray, and that calling predict(return_ds=True) will return an xarray dataset with a new data variable""" 

    # Create a model object, fit, and predict  
    toy_model = mean_baseline(y_var=toy_y_var) 
    toy_model.fit(train_ds=toy_train_ds)
    nparray_predict_return = toy_model.predict(test_ds=toy_valid_ds, method='overall_mean', return_ds=False)
    ds_predict_return = toy_model.predict(test_ds=toy_valid_ds, method='overall_mean', return_ds=True) 
    new_data_var = 'pred_' + toy_y_var

    assert isinstance(nparray_predict_return, np.ndarray), f'Calling model.predict(return_ds=False) should return an np.ndarray, not {type(nparray_predict_return)}'
    assert isinstance(ds_predict_return, xr.core.dataset.Dataset), f'Calling model.predict(return_ds=True) should return an xarray dataset, not {type(ds_predict_return)}'
    assert new_data_var in ds_predict_return.data_vars, f'The new data variable "{new_data_var}" does not exist in the new dataset when model.predict(return_ds=True) is called'

def test_predict_overall_mean():
    """Test that the predicted overall_mean array is the right shape, and that it was tiled correctly"""

    # Create a model object, fit, and predict  
    toy_model = mean_baseline(y_var=toy_y_var) 
    toy_model.fit(train_ds=toy_train_ds)
    pred_overall_mean = toy_model.predict(test_ds=toy_valid_ds, method='overall_mean', return_ds=False)

    # Check tiling 
    unique_pred_overall_mean = np.unique(pred_overall_mean)[0]

    assert pred_overall_mean.shape == toy_valid_ds[toy_y_var].shape, f'The shape of predicted overall_mean should be {toy_valid_ds[toy_y_var].shape}, not {pred_overall_mean.shape}'
    assert unique_pred_overall_mean == toy_model.overall_mean.item(), f'The predicted overall_mean does not seem to be tiled correctly'
    
def test_predict_latlon_mean():
    """Test that the predicted latlon_mean array is the right shape, and that it was tiled correctly"""

    # Create a model object, fit, and predict  
    toy_model = mean_baseline(y_var=toy_y_var) 
    toy_model.fit(train_ds=toy_train_ds)
    pred_latlon_mean = toy_model.predict(test_ds=toy_valid_ds, method='latlon_mean', return_ds=False)

    # Check tiling with pandas 
    pandas_latlon_mean = toy_train_ds[toy_y_var].to_dataframe().groupby(['latitude', 'longitude']).mean()
    pandas_latlon_mean_arr = np.array(pandas_latlon_mean).flatten()
        
    assert pred_latlon_mean.shape == toy_valid_ds[toy_y_var].shape, f'The shape of predicted latlon_mean should be {toy_valid_ds[toy_y_var].shape}, not {pred_latlon_mean.shape}'
    
    # latlon means should be the same for every random time-point - choose a random time point and check that it's equal 
    assert np.array_equal(pred_latlon_mean[2].flatten(), pandas_latlon_mean_arr), f'The predicted latlon_mean does not seem to be tiled correctly' 

def test_predict_latlonmonth_mean(): 
    """Test that the predicted latlonmonth_mean array is the right shape, and that it was tiled correctly"""

    # Create a model object, fit, and predict  
    toy_model = mean_baseline(y_var=toy_y_var) 
    toy_model.fit(train_ds=toy_train_ds)
    pred_latlonmonth_mean = toy_model.predict(test_ds=toy_valid_ds, method='latlonmonth_mean', return_ds=False) 

    # Check tiling with pandas 
    toy_train_df = toy_train_ds[toy_y_var].to_dataframe()
    toy_train_df.reset_index()['month'] = toy_train_df.reset_index()['time'].dt.month
    toy_train_df['month'] = list(toy_train_df.reset_index()['time'].dt.month)
    toy_train_df = toy_train_df.reset_index().set_index(['time', 'latitude', 'longitude', 'month']).groupby(['month', 'latitude', 'longitude']).mean()
    pandas_latlonmonth_mean = np.array(toy_train_df[toy_y_var]) 

    assert pred_latlonmonth_mean.shape == toy_valid_ds[toy_y_var].shape, f'The shape of predicted latlonmonth_mean should be {toy_valid_ds[toy_y_var].shape}, not {pred_latlonmonth_mean.shape}'
    
    # latlonmonth means should be the same for every random year - choose a random year and check that it's equal 
    assert np.array_equal(pred_latlonmonth_mean[12:24,:,:].flatten(), pandas_latlonmonth_mean), f'The predicted pred_latlonmonth_mean does not seem to be tiled correctly' 

############################## TEST THE TESTS 
test_init()
test_fit()
test_fit_calc_overall_mean()
test_fit_calc_latlon_mean()
test_fit_calc_latlonmonth_mean()
test_predict()
test_predict_overall_mean()
test_predict_latlon_mean()
test_predict_latlonmonth_mean()
print('All tests passed! 🌈')