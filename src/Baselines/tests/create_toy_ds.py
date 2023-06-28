# src/Baselines/tests/create_toy_ds.py
#########################
# Purpose: Script that creates small datasets (toy_train_ds.nc and toy_valid_ds.nc) used for the Baseline testing scripts 
# Usage: 
#    $ python3 create_toy_ds.py 
# Author: Shirley Zhang 
# Date: June 2023

# Imports
import numpy as np
import xarray as xr

############################## Path to preprocessed training data <TO BE UPDATED>
train_path = '/Users/shirley/Desktop/z_june_2/preprocessed_2023-06-02/preprocessed_train_ds.nc'
############################## 

# Check if the given path contains the preprocessed training data 
try:
    with open(train_path) as f:
        pass
except FileNotFoundError:
    print(f'Error: The preprocessed training data does not exist at the path provided. \
    \nPlease update the variable "train_path" within the script. \
    \n\ntrain_path: {train_path}')
    exit()

# Load preprocessed training data
train_ds = xr.open_dataset(train_path)
    
# Create and save toy_train_ds.nc
toy_train_ds = train_ds.sel(
    time=slice('2017-01-16T09:00:00.000000000', '2020-12-16T09:00:00.000000000'),
    latitude=slice(-75.235001, -71.426003),
    longitude=slice(350.625, 352.5)
)
toy_train_ds.to_netcdf(path='toy_train_ds.nc') 
print(f'toy_train_ds.nc created.')

# Create and save toy_valid_ds.nc
toy_valid_ds = train_ds.sel(
    time=slice('2014-01-16T09:00:00.000000000', '2016-12-16T09:00:00.000000000'),
    latitude=slice(-75.235001, -71.426003),
    longitude=slice(350.625, 352.5)
)
toy_valid_ds.to_netcdf(path='toy_valid_ds.nc') 
print(f'toy_valid_ds.nc created.')