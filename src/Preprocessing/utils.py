import sys
sys.path.append('./src/')
sys.path.append('../src/')
import math
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import xarray as xr
import netCDF4 as nc
import gpytorch as gp
from torch import nn, tensor
from matplotlib import pyplot as plt
from polar_convert import polar_lonlat_to_xy
from polar_convert.constants import (
    EARTH_ECCENTRICITY, EARTH_RADIUS_KM
)
LAT_RANGE = slice(-90, -60) # Antarctica

def combine_latlon_grids(ds):
    """
    Combine latitude_2 and longitude_2 dimensions of the dataset to latitude and longitude dimensions.
    Uses linear interpolation and properly accounts for the circular nature of longitude (0 to 360 degrees).
    
    Note: A more correct approach would be to remap the variable using polar coordinates, but this has not been implemented yet.
    
    Parameters:
    ----------
    ds : xarray DataSet
        Input dataset with latitude_2 and longitude_2 dimensions.
        
    Returns:
    -------
    xarray DataSet
        Dataset with latitude and longitude dimensions after combining latitude_2 and longitude_2 dimensions.
    """
    
    # We copy into ds_north and ds_south to handle the circle
    ds_south = ds
    ds_north = ds_south.copy()
    
    # ds north changes longitude scale from 0-360 to -180-180
    ds_north = ds_north.assign_coords(longitude = (((ds_north.longitude + 180) % 360) - 180),
                                      longitude_2 = (((ds_north.longitude_2 + 180) % 360) - 180)
                                     )
    ds_north = ds_north.sortby("longitude")
    ds_north = ds_north.sortby("longitude_2")
    
    # Interpolate ds_south and ds_north separately
    ds_south = ds_south.interp(longitude_2 = ds_south['longitude'], method="linear", kwargs={"fill_value": "extrapolate"})
    ds_south = ds_south.interp(latitude_2 = ds_south['latitude'], method="linear", kwargs={"fill_value": "extrapolate"})
    ds_south = ds_south.drop(["latitude_2", "longitude_2"])
    
    ds_north = ds_north.interp(longitude_2 = ds_north['longitude'], method="linear", kwargs={"fill_value": "extrapolate"})
    ds_north = ds_north.interp(latitude_2 = ds_north['latitude'], method="linear", kwargs={"fill_value": "extrapolate"})
    ds_north = ds_north.drop(["latitude_2", "longitude_2"])
    
    # Take the north half of north and the south half of south
    ds_north = ds_north.sel(longitude = slice(-90, 90))
    ds_south = ds_south.sel(longitude = slice(90, 270))
    
    # Revert ds north longitude scale back to 0-360
    ds_north = ds_north.assign_coords(longitude = (ds_north.longitude + 360) % 360)
    
    # Merge ds_south and ds_north
    ds_merged = xr.merge([ds_south, ds_north])
    
    return ds_merged

def check_valid_xarray_input(dataset):
    """
    Check the validity of the input dataset.
    It should be an xarray DataSet or xarray DataArray with dimensions ['time', 'latitude', 'longitude'].
    
    Parameters:
    ----------
    dataset : xarray DataSet or xarray DataArray
        Input dataset to be checked.
        
    Returns:
    -------
    bool or str
        - True if the dataset is valid and has the correct dimensions.
        - "transpose lat/lon" if the dataset had dimensions ['time', 'longitude', 'latitude'] and requires transposing.
        - False if the dataset is invalid or has incorrect dimensions.
    
    """
    if isinstance(dataset, xr.core.dataarray.DataArray):
        dim_names = [*dataset.dims]
        if dim_names == ['variable', 'time', 'longitude', 'latitude']:
            dataset = dataset.to_dataset(dim='variable').transpose('time', 'latitude', 'longitude').to_array().to_dataset(dim='variable') # Forces the dimensions to be (time, lat, lon)
            return "transpose lat/lon"
        if not dim_names == ['variable', 'time', 'latitude', 'longitude']:
            print("Error: The input DataArray should have dimensions=['variable', 'time', 'latitude', 'longitude']") # TODO: Proper error msgs
            return False
        return True
    elif isinstance(dataset, xr.core.dataset.Dataset):
        dim_names = [*dataset.dims.keys()]
        if dim_names == ['time', 'longitude', 'latitude']:
            dataset = dataset.transpose('time', 'latitude', 'longitude').to_array().to_dataset(dim='variable') # Forces the dimensions to be (time, lat, lon)
            return "transpose lat/lon"
        if not dim_names == ['time', 'latitude', 'longitude']:
            print("Error: The input DataSet should have dimensions=['time', 'latitude', 'longitude']") # TODO: Proper error msgs
            return False
        return True
    else:
        print("Error: The input should be an XArray dataset") # TODO: Proper error msgs
        return False

def get_lon_lat_numpy_grids(dataset):
    """
    Given an XArray dataset, return 3D grids (numpy arrays) of longitude and lattitude values.
    The input can be an XArray DataArray or an XArray Dataset.

    This function transforms an input XArray Dataset or DataArray into 3D grids (numpy arrays) of longitude and latitude values. 
    It retrieves the latitude and longitude values from the input and forms 3D numpy arrays with dimensions representing time, latitude, and longitude.

    Parameters
    ----------
    dataset : xr.Dataset or xr.DataArray
        The input XArray Dataset or DataArray. In the case of DataArray, the input should have dimensions 
        of (variable, time, latitude, longitude). In the case of Dataset, the input should have dimensions 
        of (time, latitude, longitude).

    Returns
    -------
    dict
        A dictionary with two 3D numpy arrays representing grids of longitude and latitude values. 
        The keys of the dictionary are 'longitude' and 'latitude'. The 3D numpy arrays have dimensions 
        representing time, latitude, and longitude. 

        For example:

        {
            'longitude': lon_grid,
            'latitude': lat_grid
        }

    Notes
    -----
    This function assumes that the input Dataset or DataArray has dimensions for time, latitude, and longitude. 
    If these are not present in the input, the function may not behave as expected. Also, the function does not 
    check whether the values in the longitude and latitude dimensions are actually valid longitude and latitude 
    values.
    """
    check_valid_xarray_input(dataset)
    if isinstance(dataset, xr.core.dataset.Dataset):
        dataset = dataset.to_array()

    # Get the dimension names and sizes
    dim_names = [*dataset.dims]
    dims = {
        dim_name: dim_size 
        for dim_name, dim_size in zip(dim_names, dataset.shape)
    }

    lats, lons = dataset.latitude.values, dataset.longitude.values

    # These are 2D numpy arrays w/ dimensions=(lat, lon): 
    lons_per_timeslice = np.tile(
        lons[:, np.newaxis],    # Values are longitudes.
        dims['latitude']
    ).T           
    lats_per_timeslice = np.repeat(
        lats[:, np.newaxis], 
        dims['longitude'], # Values are latitudes.
        axis=1
    )

    # These are 3D numpy arrays w/ dimensions=(lat, lon, time):
    lon_grid = np.tile(
        lons_per_timeslice[:, :, np.newaxis], # Values are longitudes.
        dims['time']
    )   
    lat_grid = np.tile(
        lats_per_timeslice[:, :, np.newaxis], # Values are latitudes.
        dims['time']
    )

    # Swap 1st and last dimensions so that the 3D arrays have dimensions=(time, lat, lon):
    lon_grid = np.moveaxis(lon_grid, -1, 0)   
    lat_grid = np.moveaxis(lat_grid, -1, 0)

    return {
        'longitude': lon_grid,
        'latitude': lat_grid
    }

def get_time_numpy_grid(dataset):
    """
    Given an XArray dataset, return 3D grids (numpy arrays) of datetime objects.

    Parameters
    ----------
    dataset : xr.Dataset or xr.DataArray
        The input XArray Dataset or DataArray.

    Returns
    -------
    numpy.ndarray
        A 3D numpy array with dimensions representing time, latitude, and longitude. 
        The values in the array are datetime objects.

    Notes
    -----
    The input can be an XArray DataArray or an XArray Dataset:
        - In the case of DataArray, the input should have dimensions of (variable, time, latitude, longitude).
        - In the case of Dataset, the input should have dimensions of (time, latitude, longitude).
    """
    check_valid_xarray_input(dataset)
    if isinstance(dataset, xr.core.dataset.Dataset):
        dataset = dataset.to_array()

    # Get the dimension names and sizes
    dim_names = [*dataset.dims]
    dims = {
        dim_name: dim_size 
        for dim_name, dim_size in zip(dim_names, dataset.shape)
    }
    
    times = dataset.time.values # This is a 1D numpy array of the *unique* timeslices in the dataset. (Values are datetime objects).
    time_grid_flat = np.repeat(times, dims['latitude'] * dims['longitude']) # Still a 1D numpy array, but now each timeslice is repeated once for each lat/lon location. 
    time_grid = time_grid_flat.reshape((dims['time'], dims['latitude'], dims['longitude'])) # A 3D numpy array w/ dimensions=(time, lat, lon) and values = datatime objects.

    return time_grid

def get_polar_coords_numpy_grids(dataset):
    """
    Given an XArray dataset, return 3D grids (numpy arrays) of projected E/N values (using UPS projection).

    Parameters
    ----------
    dataset : xr.Dataset or xr.DataArray
        The input XArray Dataset or DataArray.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Two 3D numpy arrays representing grids of projected E and N values. 
        The arrays have dimensions representing time, latitude, and longitude.

    Notes
    -----
    The input can be an XArray DataArray or an XArray Dataset:
        - In the case of DataArray, the input should have dimensions of (variable, time, latitude, longitude).
        - In the case of Dataset, the input should have dimensions of (time, latitude, longitude).
    """
    lon_lat_grids = get_lon_lat_numpy_grids(dataset)
    E_grid, N_grid = lon_lat_to_polar(
        lon_lat_grids['longitude'], lon_lat_grids['latitude']
    )
    return E_grid, N_grid

def lon_lat_to_polar(lon, lat, hemisphere = "south", true_scale_lat = 80):
    """
    Given 3D grids (numpy arrays) of longitude and latitude values, create arrays of projected E/N values (using UPS projection). 
    This is a helper function for `get_polar_coords_numpy_grids`.

    Parameters
    ----------
    lon : numpy.ndarray
        A 3D numpy array representing the grid of longitude values.
    lat : numpy.ndarray
        A 3D numpy array representing the grid of latitude values.
    hemisphere : str, optional
        The hemisphere for projection. Valid values are 'north' and 'south'. Default is 'south'.
    true_scale_lat : float, optional
        The true scale latitude for projection. Default is 80.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Two 3D numpy arrays representing grids of projected E and N values. 
        The arrays have dimensions representing time, latitude, and longitude.

    """
    return polar_lonlat_to_xy(
        lon, lat, 
        true_scale_lat=true_scale_lat, hemisphere=hemisphere,
        re=EARTH_RADIUS_KM, e=EARTH_ECCENTRICITY
    )

def datetime_to_months(dates):
    """
    Convert datetime objects to integer months.

    This function converts datetime objects to integer months with January as 1 and December as 12.

    Parameters
    ----------
    dates : np.array
        Numpy array of datetime objects.

    Returns
    -------
    np.array
        Numpy array of integer months corresponding to the input datetime objects.
    """
    return dates.astype('datetime64[M]').astype(int) % 12 + 1

def datetime_to_years(dates):
    """
    Convert datetime objects to integer years.

    This function converts datetime objects to integer years with AD1 (1CE) as 1.

    Parameters
    ----------
    dates : np.array
        Numpy array of datetime objects.

    Returns
    -------
    np.array
        Numpy array of integer years corresponding to the input datetime objects.
    """
    return dates.astype('datetime64[Y]').astype(int) + 1970

def array_to_flat_tensor(array):
    """
    Convert a numpy array to a flat PyTorch tensor.

    This function converts a numpy array (of any shape) into a flat (1D) PyTorch tensor with dtype=float32.

    Parameters
    ----------
    array : np.array
        Numpy array to be converted.

    Returns
    -------
    torch.Tensor
        A 1D (flattened) PyTorch tensor derived from the input numpy array.

    Notes
    -----
    The function converts the array data to float32 type before making the conversion to a PyTorch tensor. This is done to save memory when training.
    """
    return tensor(array.astype(np.float32)).flatten()
    
def add_polar_coords(dataset):
    """
    Add projected E/N values to an XArray dataset (using UPS projection).

    This function calculates the projected easting (E) and northing (N) values using the Universal Polar Stereographic (UPS) projection.
    The resulting E and N grids are added as variables 'E' and 'N' to the input dataset.

    Parameters
    ----------
    dataset : xr.Dataset or xr.DataArray
        The input XArray Dataset or DataArray.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The modified XArray Dataset or DataArray with added 'E' and 'N' variables.

    Notes
    -----
    The input can be an XArray DataArray or an XArray Dataset:
        - In the case of DataArray, the input should have dimensions of (variable, time, latitude, longitude).
        - In the case of Dataset, the input should have dimensions of (time, latitude, longitude).
    """
    check_valid_xarray_input(dataset)
    if isinstance(dataset, xr.core.dataarray.DataArray):
        out = 'DataArray'
        dataset = dataset.to_dataset(dim='variable')
    elif isinstance(dataset, xr.core.dataset.Dataset):
        out = 'DataSet'

    E_grid, N_grid = get_polar_coords_numpy_grids(dataset)
    # TESTING
    E_grid = np.round(E_grid, decimals=15)
    N_grid = np.round(N_grid, decimals=15)
    # TESTING
    dataset = dataset.assign(E = (['time', 'latitude', 'longitude'], E_grid))
    dataset = dataset.assign(N = (['time', 'latitude', 'longitude'], N_grid))

    if out=='DataArray':
        return dataset.to_array()
    return dataset

def add_months(dataset):
    """
    Add month values (integers) to an XArray dataset (Jan=1, ..., Dec=12).
    
    This function calculates the month values from the datetime objects in the input dataset.
    The resulting month grid is added as a variable 'month' to the input dataset.

    Parameters
    ----------
    dataset : xr.Dataset or xr.DataArray
        The input XArray Dataset or DataArray.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The modified XArray Dataset or DataArray with added 'month' variable.

    Notes
    -----
    The input can be an XArray DataArray or an XArray Dataset:
        - In the case of DataArray, the input should have dimensions of (variable, time, latitude, longitude).
        - In the case of Dataset, the input should have dimensions of (time, latitude, longitude).
    - The month values are calculated from the datetime objects in the 'time' dimension of the dataset.
    """
    check_valid_xarray_input(dataset)
    if isinstance(dataset, xr.core.dataarray.DataArray):
        out = 'DataArray'
        dataset = dataset.to_dataset(dim='variable')
    elif isinstance(dataset, xr.core.dataset.Dataset):
        out = 'DataSet'

    time_grid = get_time_numpy_grid(dataset)
    month_grid = datetime_to_months(time_grid)
    dataset = dataset.assign(month = (['time', 'latitude', 'longitude'], month_grid))

    if out=='DataArray':
        return dataset.to_array()
    return dataset

def add_years(dataset):
    """
    Add year values (integers) to an XArray dataset (1=AD1=1CE).
    
    This function calculates the year values from the datetime objects in the input dataset.
    The resulting year grid is added as a variable 'year' to the input dataset.

    Parameters
    ----------
    dataset : xr.Dataset or xr.DataArray
        The input XArray Dataset or DataArray.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The modified XArray Dataset or DataArray with added 'year' variable.

    Notes
    -----
    The input can be an XArray DataArray or an XArray Dataset:
        - In the case of DataArray, the input should have dimensions of (variable, time, latitude, longitude).
        - In the case of Dataset, the input should have dimensions of (time, latitude, longitude).
    - The year values are calculated from the datetime objects in the 'time' dimension of the dataset.
    """
    check_valid_xarray_input(dataset)
    if isinstance(dataset, xr.core.dataarray.DataArray):
        out = 'DataArray'
        dataset = dataset.to_dataset(dim='variable')
    elif isinstance(dataset, xr.core.dataset.Dataset):
        out = 'DataSet'

    time_grid = get_time_numpy_grid(dataset)
    year_grid = datetime_to_years(time_grid)
    dataset = dataset.assign(year = (['time', 'latitude', 'longitude'], year_grid))

    if out=='DataArray':
        return dataset.to_array()
    return dataset

def add_land_sea_mask(dataset, filepath):
    """
    Join land sea mask to xArray dataset from file.

    This function reads a land sea mask dataset from a specified file and joins it with the input dataset.
    The land sea mask is added as a variable 'landsfc' to the input dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The input XArray Dataset
    filepath : str
        The file path to the land sea mask dataset.

    Returns
    -------
    xr.Dataset
        The modified XArray Dataset with added land sea mask variable ('landsfc').

    Notes
    -----
    - The input dataset should have dimensions of (time, latitude, longitude).
    - The land sea mask dataset is expected to have dimensions of latitude and longitude.
    - The 'landsfc' variable represents the land-sea mask values as 0s and 1s.
    """
    landsea_ds = xr.open_dataset(filepath)
    landsea_ds = landsea_ds.squeeze('time') # Remove time dimensions
    landsea_ds = landsea_ds.sel(latitude = LAT_RANGE)
    dataset['landsfc'] = (['latitude', 'longitude'], landsea_ds['landsfc'].data) 
    dataset = dataset.to_array().to_dataset(dim='variable') # This broadcasts time dimension onto mask variable
    
    return dataset

def add_dist_to_coast(dataset):
    """
    Given an XArray dataset with polar coordinates and a land sea mask,
    calculate the distance to coast feature for each point.

    Parameters
    ----------
    dataset : xr.Dataset
        The input XArray dataset with polar coordinates and a land sea mask.

    Returns
    -------
    xr.Dataset
        The modified XArray dataset with an additional 'dist_to_coast' variable.

    Notes
    -----
    - The input dataset must contain variables 'E', 'N', and 'landsfc' representing the polar coordinates and land sea mask.
    - The 'dist_to_coast' variable represents the calculated distance to the coast feature for each point in the dataset.
    - The calculation is performed by iterating over the points and finding the minimum distance to the opposite masked point.
    - The distance is adjusted for sea points by setting them to have negative distance.
    """
    assert 'E' in dataset, "Easting missing from dataset, cannot calculate distances"
    assert 'N' in dataset, "Northing missing from dataset, cannot calculate distances"
    assert 'landsfc' in dataset, "Land sea mask missing, cannot calculate distances"
    
    # Original dimensions so we can re-broadcast generically
    n_time = dataset.sizes['time']
    n_lats = dataset.sizes['latitude']
    n_lons = dataset.sizes['longitude']
    
    # Convert to 2d-numpy matrix for faster math
    mat = dataset[['E', 'N', 'landsfc']].isel(time=0).to_array().to_numpy()
    mat = mat.reshape(3, -1).swapaxes(0, 1) # Flip to row-wise
    mat = np.c_[mat, np.full(len(mat), np.inf)] # Insert empty column
    
    # Calculate minimum distance to opposite masked point
    for i in range(len(mat)):
        for j in range(len(mat)):     # Yes this is n**2; it's fine we're only doing it once
            if mat[i][2] != mat[j][2]: # Calculate only if land/sea is different
                d = ( (mat[i][0]-mat[j][0])**2 + (mat[i][1]-mat[j][1])**2 ) ** 0.5
                mat[i][3] = min(mat[i][3], d)
    
    # Adjust distances of sea points to be negative
    sea_mask = mat[:,2] == 0
    mat[sea_mask, 3] = -mat[sea_mask, 3]
    
    # Reshape, broadcast for all time, and insert into original dataset
    dist_to_coast = mat[:, 3].reshape(1, n_lats, n_lons) * np.ones((n_time, 1, 1))
    dataset['dist_to_coast'] = (['time', 'latitude', 'longitude'], dist_to_coast)
    
    return dataset
    
def add_orogrd(dataset, filepath):
    """
    Join the orography variable to an XArray dataset from a file.

    This function reads an orographpy dataset from a specified file and joins it with the input dataset.
    The orography is added as a variable 'oro' to the input dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The input XArray dataset.
    filepath : str
        The file path to the orography dataset.

    Returns
    -------
    xr.Dataset
        The modified XArray Dataset with added orographpy variable ('oro').

    Notes
    -----
    - The input dataset should have dimensions of (time, latitude, longitude).
    - The orography dataset is expected to have dimensions of latitude and longitude.
    - The 'oro' variable represents the surface altitude of a given point.
    """
    oro_ds = xr.open_dataset(filepath)
    oro_ds = oro_ds.squeeze('time') # Remove time dimensions
    oro_ds = oro_ds.sel(latitude = LAT_RANGE)
    dataset['oro'] = (['latitude', 'longitude'], oro_ds['oro'].data) 
    dataset = dataset.to_array().to_dataset(dim='variable') # This broadcasts time dimension onto oro variable
    
    return dataset

def xarray_to_numpy2D(dataset, features):
    """
    Convert an XArray dataset to a 2D numpy array with shape (num_examples, num_features).

    The features included in the output numpy array are specified with `features`, which should be a list of strings representing the feature names.
    The index of each feature's column in the output numpy array will match the index of the feature's name in the input feature list.

    Parameters
    ----------
    dataset : xr.Dataset or xr.DataArray
        The input XArray Dataset or DataArray.
    features : list of str
        A list of strings representing the feature names to be included in the output numpy array.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array representing the converted dataset. The array has shape (num_examples, num_features).

    Notes
    -----
    - The input dataset should have dimensions of (variable, time, latitude, longitude) for DataArray or (time, latitude, longitude) for Dataset.
    - The features specified in the `features` list will be extracted from the dataset and included as columns in the output numpy array.
    - The `features` list can include dimensions (e.g. 'latitude' and 'longitude').
    - The dimension 'time' is not recommended as a feature since it contains datetime objects. Instead, use the features 'month' or 'year'.
    """
    check_valid_xarray_input(dataset)
    if isinstance(dataset, xr.core.dataarray.DataArray):
        dataset = dataset.to_dataset(dim='variable')
    
    # Get lat/lon grids (if needed).
    ##=====## # This code covers the case of `features` including a dimension of the dataset. (rather than a variable) (e.g. 'latitude').
    dim_inds = []
    lon_lat_grids = None
    if 'longitude' in features:
        dim_inds.append(features.index('longitude'))
        lon_lat_grids = get_lon_lat_numpy_grids(dataset)
        lon_grid = lon_lat_grids['longitude']
    if 'latitude' in features:
        dim_inds.append(features.index('latitude'))
        if lon_lat_grids is not None:
            lat_grid = lon_lat_grids['latitude']
        else:
            lat_grid = get_lon_lat_numpy_grids(dataset)['latitude']
    if 'time' in features:  # Nope.
        print("Using 'time' as a feature column is probably not a great idea. Are you sure you want a column of datetime objects?")
        print("Try using 'year' or 'month' instead of 'time'.")
        # dim_inds.append(features.index('time'))
        # time_grid = get_time_numpy_grid(dataset)
        return
    dim_features = get_elements(features, dim_inds)
    dim_feature_numpy_grids = []
    for dim in dim_features:
        if dim == 'longitude':
            dim_feature_numpy_grids.append(lon_grid)
        elif dim == 'latitude':
            dim_feature_numpy_grids.append(lat_grid)
        # elif dim == 'time':
        #     dim_feature_numpy_grids.append(time_grid)
    ##=====##

    # Get feature grids
    features_no_dims = delete_elements(features, dim_inds)
    feature_numpy_grids = [
        get_feat_numpy_grid(dataset, feat_name=feat)
        for feat in features_no_dims
    ]

    # Insert the lat/lon grids into the appropriate positions (if needed).
    # This ensures that the indices in the input feature list match the column indices in the output numpy array.
    for dim_ind, dim_grid in zip(dim_inds, dim_feature_numpy_grids):
        feature_numpy_grids.insert(dim_ind, dim_grid)

    # Flatten all feature grids (3D) into 1D arrays.
    feature_numpy_flat_arrays = [
        feat_grid.flatten()
        for feat_grid in feature_numpy_grids
    ]
    
    # Stack feature arrays into a 2D array, transpose to make the features the columns.
    return np.stack(feature_numpy_flat_arrays).T  # Shape = (num_examples, num_features) 

def get_feat_numpy_grid(dataset, feat_name):
    """
    Given an XArray dataset, return a 3D grid (numpy array) of a variable in that dataset.

    This function extracts the grid of the specified variable from the dataset and returns it as a 3D numpy array with dimensions ('time', 'latitude', 'longitude').

    Parameters
    ----------
    dataset : xr.Dataset or xr.DataArray
        The input XArray Dataset or DataArray.
    feat_name : str
        The name of the variable to extract from the dataset.

    Returns
    -------
    numpy.ndarray
        A 3D numpy array representing the grid of the specified variable in the dataset. The array has shape (num_time_slices, num_lat_slices, num_lon_slices). 

    Notes
    -----
    - The input dataset should have dimensions of (variable, time, latitude, longitude) for DataArray or (time, latitude, longitude) for Dataset.
    """
    check_valid_xarray_input(dataset)
    if isinstance(dataset, xr.core.dataarray.DataArray):
        dataset = dataset.to_dataset(dim='variable')
    
    if not isinstance(feat_name, str):
        print("Error: 'feat_name' should be a string.") # TODO: Proper error msgs
        return
    
    return getattr(dataset, feat_name).values

def flat_to_grid(dataset, flat_array):
    """
    Given an XArray dataset and a flat array of values, reshape the array into a 3D grid (numpy array) with a shape that is compatible with the XArray dataset.
    
    This function reshapes the input 1D array into a 3D numpy array with dimensions matching the XArray dataset.

    Parameters
    ----------
    dataset : xr.Dataset or xr.DataArray
        The input XArray Dataset or DataArray.
    flat_array : numpy.ndarray
        A 1D numpy array of values.

    Returns
    -------
    numpy.ndarray
        A 3D numpy array representing the reshaped grid of values compatible with the XArray dataset.

    Notes
    -----
    - The input dataset should have dimensions of (variable, time, latitude, longitude) for DataArray or (time, latitude, longitude) for Dataset.
    """
    check_valid_xarray_input(dataset)
    if isinstance(dataset, xr.core.dataarray.DataArray):
        dataset = dataset.to_dataset(dim='variable')
        
    grid_shape = get_time_numpy_grid(dataset).shape

    return np.reshape(flat_array, grid_shape)

def delete_elements(list, inds):
    """
    Return a copy of the input list with the elements at indices `inds` deleted.

    Parameters
    ----------
    list : list
        The input list.
    inds : list of int
        A list of indices indicating the elements to be deleted.

    Returns
    -------
    list
        A new list with the elements at the specified indices removed.
    """
    return np.delete(np.array(list), inds).tolist()

def get_elements(list, inds):
    """
    Return a copy of the input list with only the elements at indices `inds` included.

    Parameters
    ----------
    list : list
        The input list.
    inds : list of int
        A list of indices indicating the elements to be included.

    Returns
    -------
    list
        A new list with only the elements at the specified indices included.
    """
    return [list[i] for i in inds]
        
class XArrayStandardScaler():
    """
    StandardScaler for XArray.

    Similar to StandScaler for scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
    """
    def __init__(self, scale_feats=['tmp2m']):
        """
        Initialize the XArrayStandardScaler.

        Parameters:
        -----------
        scale_feats : list
            List of feature names to be scaled. Default is ['tmp2m'].

        Attributes:
        -----------
        scale_feats : list
            List of feature names to be scaled. 
        mean : numpy.ndarray or None
            Array of feature means for scaling.
        stdev : numpy.ndarray or None
            Array of feature standard deviations for scaling.
        """
        self.scale_feats = scale_feats
        self.mean = None
        self.stdev = None

    def _is_initialized(self):
        """
        Check if the XArrayStandardScaler object is initialized.

        Returns:
        --------
        bool
            True if scaler is initialized, False otherwise.
        """
        if self.mean is None or self.stdev is None:
            return False
        if (
            len(self.mean) != len(self.scale_feats) 
            or 
            len(self.stdev) != len(self.scale_feats)
        ):
            print("Error during initilization.")
            return False
        return True

    def initialize(self, dataset, from_attrs=False, attrs=None):
        """
        Initialize the scaler with mean and standard deviation values.

        Parameters:
        -----------
        dataset : xr.Dataset or xr.DataArray
            The input XArray dataset or data array.
        from_attrs : bool, optional
            Whether to initialize from attributes. Default is False.
        attrs : dict or None, optional
            Dictionary of attributes for initializing from attributes. Default is None.
            If None, the function will retrieve the dictionary from the input dataset.

        Notes:
        ------
        - If `from_attrs` is True, the mean and standard deviation values are retrieved from the dataset attributes.
        - If `from_attrs` is False, the mean and standard deviation values are calculated from the dataset.
        """
        assert not self._is_initialized(), "Scaler has already been initialized."
        scale_feats = self.scale_feats

        if from_attrs:
            if attrs is None:
                check_valid_xarray_input(dataset)
                if isinstance(dataset, xr.core.dataarray.DataArray):
                    dataset = dataset.to_dataset(dim='variable')
                attrs = dataset.attrs
            assert isinstance(attrs, dict), "Incorrect attributes format. Cannot initialize scaler."

            means, stdevs = [], []
            for scale_feat_name in scale_feats:
                mean_key = f'scaler_mean_{scale_feat_name}'
                stdev_key = f'scaler_stdev_{scale_feat_name}'
                means.append(attrs[mean_key])
                stdevs.append(attrs[stdev_key])
            self.mean = np.array(means)
            self.stdev = np.array(stdevs)
            assert self._is_initialized(), "Error. (XArrayStandardScaler initialization)"

            return
        
        check_valid_xarray_input(dataset)
        if isinstance(dataset, xr.core.dataarray.DataArray):
            dataset = dataset.to_dataset(dim='variable')
        
        input_feat_names = [*dataset.keys()]
        assert set(scale_feats).issubset(input_feat_names), \
            "The input dataset must contain all of 'scale_feats' as variables."
        
        feat_data_numpy_2D = xarray_to_numpy2D(dataset, scale_feats)
        n_examples, n_feats = feat_data_numpy_2D.shape
        assert n_examples > 1, "Cannot calculate standard deviation using n=1 examples." # For calculating standard deviation we divide by n-1.

        self.mean = np.nanmean(feat_data_numpy_2D, axis=0)
        self.stdev = np.nanstd(feat_data_numpy_2D, axis=0)
        assert self._is_initialized(), "Error. (XArrayStandardScaler initialization)"

    def fit(self, dataset, from_attrs=False, attrs=None):
        """
        Alias for the `initialize` method.
        """
        self.initialize(dataset, from_attrs, attrs)

    def transform(self, dataset):
        """
        Transform the dataset by scaling the specified features.

        Parameters:
        -----------
        dataset : xr.Dataset or xr.DataArray
            The input XArray dataset or data array to be transformed.

        Returns:
        --------
        xr.Dataset
            A new XArray dataset with the scaled features.

        Raises:
        -------
        RuntimeError
            If the scaler is not initialized.
        """
        if not self._is_initialized():
            raise RuntimeError("The scaler must be initialized first!")
        
        check_valid_xarray_input(dataset)
        if isinstance(dataset, xr.core.dataarray.DataArray):
            dataset = dataset.to_dataset(dim='variable')
    
        scale_feats = self.scale_feats
        input_feat_names = [*dataset.keys()]
        assert set(scale_feats).issubset(input_feat_names), \
            "The input dataset must contain all of 'scale_feats' as variables."
        
        feat_data_numpy_2D = xarray_to_numpy2D(dataset, scale_feats)

        scaled_data_numpy_2D = (feat_data_numpy_2D - self.mean) / self.stdev
        scaled_columns = [
            scaled_data_numpy_2D[:, i] 
            for i in range(len(scale_feats))
        ]
        scaled_grids = [
            flat_to_grid(dataset, scaled_column)
            for scaled_column, scale_feat_name in zip(scaled_columns, scale_feats)
        ]

        new_dataset = dataset.copy()
        for i, (scaled_grid, scale_feat_name) in enumerate(zip(scaled_grids, scale_feats)):
            new_dataset = new_dataset.assign(
                scaled_feat = (['time', 'latitude', 'longitude'], scaled_grid)
            ).rename(
                {'scaled_feat': f"scaled_{scale_feat_name}"}
            ).assign_attrs({
                f'scaler_mean_{scale_feat_name}': self.mean[i],
                f'scaler_stdev_{scale_feat_name}': self.stdev[i],
            })

        return new_dataset
      
    def fit_transform(self, dataset):
        """
        Fit and transform the dataset in a single step.

        Parameters:
        -----------
        dataset : xr.Dataset or xr.DataArray
            The input XArray dataset or data array to be transformed.

        Returns:
        --------
        xr.Dataset
            A new XArray dataset with the scaled features.
        """
        self.fit(dataset)
        return self.transform(dataset)
    
    def inverse_transform(self, scaled_dataset):
        """
        Inverse transform the scaled dataset to obtain the original values.

        Parameters:
        -----------
        scaled_dataset : xr.Dataset or xr.DataArray
            The input XArray dataset or data array with scaled features.

        Returns:
        --------
        xr.Dataset
            A new XArray dataset with the original unscaled features.

        Raises:
        -------
        RuntimeError
            If the scaler is not initialized.
        """
        if not self._is_initialized():
            raise RuntimeError("The scaler must be intitialized first!")
        
        check_valid_xarray_input(scaled_dataset)
        dataset = scaled_dataset
        if isinstance(dataset, xr.core.dataarray.DataArray):
            dataset = dataset.to_dataset(dim='variable')
    
        scaled_feats = [f"scaled_{feat_name}" for feat_name in self.scale_feats]
        input_feat_names = [*dataset.keys()]
        assert set(scaled_feats).issubset(input_feat_names), \
            "The input dataset must contain all of 'scale_feats' as variables."
        
        scaled_data_numpy_2D = xarray_to_numpy2D(dataset, scaled_feats)

        original_data_numpy_2D = (scaled_data_numpy_2D *self.stdev) +  self.mean

        original_columns = [
            original_data_numpy_2D[:, i] 
            for i in range(len(scaled_feats))
        ]
        original_grids = [
            flat_to_grid(dataset, original_column)
            for original_column, scaled_feat_name in zip(original_columns, scaled_feats)
        ]

        new_dataset = dataset.copy()
        for i, (original_grid, original_feat_name) in enumerate(zip(original_grids, self.scale_feats)):
            new_dataset = new_dataset.assign(
                orignal_feat = (['time', 'latitude', 'longitude'], original_grid)
            ).rename( # This part will throw an error if the original feature is already in the input dataset. That's fine tho. 
                {'orignal_feat': original_feat_name} # The whole point of inverse_transform is to add the original feature to the dataset, assuming we only have the scaled feature.
            ).assign_attrs({
                f'scaler_data_{original_feat_name}': {
                    'mean': self.mean[i],
                    'stdev': self.stdev[i]
                    }
            })

        return new_dataset


def deseasonalize(ds, years, train_ds=None): 
    """
    Given an XArray dataset, return a new dataset with monthly means and deseasonalized variables as new variables in the dataset.

    Assumes that the input `ds` starts perfectly at January of one year and ends in December of another (i.e., full years).

    Parameters:
    -----------
    ds : xr.Dataset
        The input XArray dataset.
    years : int
        The number of years covered by the dataset. This parameter should match the actual number of years in the dataset.
    train_ds : xr.Dataset or None, optional
        The training dataset used to compute seasonal means. If provided, the function uses the seasonal means from the training dataset.
        If not provided, the function computes the seasonal means from the input `ds` itself.

    Returns:
    --------
    xr.Dataset
        A new XArray dataset with monthly means and deseasonalized variables.
    Notes:
    ------
    - The function computes deseasonalized variables by subtracting the seasonal means from the original variables.
    - The deseasonalized variables are added as new variables with names 'deseas_{var}' in the new dataset, where '{var}' is the original variable name.
    - The monthly means (seasonal means) are also added as new attributes to the new dataset with names 'seasonal_means_{var}'.
    - If `train_ds` is provided, the function uses the seasonal means from the training dataset to compute deseasonalized variables.
    - If `train_ds` is not provided, the function computes the seasonal means from the input `ds` itself.
    """
    
    # get number of years
    num_years = years # TODO: get num_year automatically from the input `ds` instead of making it a function parameter
    # get list of variables 
    variables_list = [*ds.keys()]

    new_ds = ds.copy()

    # if not training data
    if train_ds is not None:
        for var in variables_list: 
            # retrieve seasonal means from training data
            attr_name = 'seasonal_means_' + var
            seasonal_means = train_ds[attr_name].values[:12, :, :]
            expanded_seasonal_means_array = np.tile(seasonal_means, reps=(1, num_years, 1, 1))
            # subtract means to get anomalies 
            deseas_array = ds[var].values - expanded_seasonal_means_array
            # save the deseasonalized array and monthly means 
            new_ds = new_ds.assign(
                deseas = (['time', 'latitude', 'longitude'], deseas_array[0])
            ).rename(
                {'deseas': f"deseas_{var}"}
            ).assign(
                seasonal_means = (['time', 'latitude', 'longitude'], expanded_seasonal_means_array[0])
            ).rename(
                {'seasonal_means': f'seasonal_means_{var}'}
            )

    # if training data
    else:
        for var in variables_list: 
            # compute seasonal means from training data (to be saved later)
            seasonal_means = ds[var].groupby('time.month').mean(dim='time').to_numpy()
            # reshape/reformat array of seasonal means
            seasonal_means = np.expand_dims(seasonal_means.astype(np.float64), axis=0)
            # tile array for number of years
            expanded_seasonal_means_array = np.tile(seasonal_means, reps=(1, num_years, 1, 1))
            # subtract means to get anomalies 
            deseas_array = ds[var].values - expanded_seasonal_means_array
            # save the deseasonalized array and monthly means 
            new_ds = new_ds.assign(
                deseas = (['time', 'latitude', 'longitude'], deseas_array[0])
            ).rename(
                {'deseas': f"deseas_{var}"}
            ).assign(
                seasonal_means = (['time', 'latitude', 'longitude'], expanded_seasonal_means_array[0])
            ).rename(
                {'seasonal_means': f'seasonal_means_{var}'}
            )
        
    return new_ds 

    
def percent_nans(flat_tensor):
    """
    Takes in a flattened tensor of values and returns the percentage of NaNs in the tensor as a float.

    This is a helper function for interpolate_nans().

    Parameters:
    -----------
    flat_tensor : torch.Tensor
        A flattened tensor representing a variable.

    Returns:
    --------
    float
        The percentage of NaN values in the flattened tensor.
    """
    return int(flat_tensor.isnan().sum()) / flat_tensor.shape[0] * 100

def interpolate_nans_smart(ds, dims_list, var):
    """
    Takes in a dataset and a variable to interpolate, and does advanced interpolation along 3 dimensions to fill missing values.

    This is a helper function for interpolate_nans().

    Parameters:
    -----------
    ds : xr.Dataset
        The input XArray dataset.
    dims_list : list
        A list of dimension names in the dataset, in the order [time, latitude, longitude].
    var : str
        The name of the variable to interpolate.

    Returns:
    --------
    xr.Dataset
        A new XArray dataset with NaNs filled for the specified variable.

    Notes:
    ------
    - The function performs advanced interpolation for the specified variable along the time, latitude, and longitude dimensions.
    - It identifies the indices of NaN values in the variable and performs interpolation only for those indices.
    - The interpolated values are combined with the original dataset, replacing the NaN values.
    """
    time = dims_list[0]
    lat = dims_list[1]
    lon = dims_list[2] 
    
    # Get the time, latitude, and longitude dimensions that correspond to the NaN values:
    # first, flatten variables and dimensions 
    var_flat_array = (ds[var].values.astype(np.float32)).flatten()
    time_flat_array = (np.repeat(ds[time].values.astype(np.float64), ds.dims[lat]*ds.dims[lon]))
    lat_flat_array = (np.tile(np.repeat(ds[lat].values, ds.dims[lon]), ds.dims[time]))
    lon_flat_array = (np.tile(ds[lon].values, ds.dims[time]*ds.dims[lat]))
    
    # stack as (n, 4) array --> (var, time, lat, lon)
    stacked_array = np.stack([var_flat_array, time_flat_array, lat_flat_array, lon_flat_array,], axis=1)

    # get only rows with NaNs 
    stacked_array_nans = stacked_array[np.isnan(stacked_array).any(axis=1)]

    # get time, lat, and lon corresponding to NaNs
    time_nans = list(stacked_array_nans[:,1].astype('datetime64[ns]')) # convert back to datetime
    lat_nans = stacked_array_nans[:,2].tolist()
    lon_nans = stacked_array_nans[:,3].tolist() 

    # INTERPOLATE 
    # drop NaNs for var (you have to choose a dimension, so we'll do it along the longitude dimension) so we can interpolate 
    # (or else interpolation values will be NaN)
    var_drop_nans_ds = ds[var].dropna(dim=lat, how="any")
    var_interp_ds = var_drop_nans_ds.interp(time=('z', time_nans), 
                                            latitude=('z', lat_nans),
                                            longitude=('z', lon_nans),
                                            kwargs={'fill_value': None},
                                            )

    # COMBINE interpolated values with the original ds 
    # [PANDAS STEP] reset the index to convert to xarray 
    var_interp_ds = var_interp_ds.to_dataframe().set_index([time, lat, lon]).to_xarray()
    final_var_interp_ds = ds[var].combine_first(var_interp_ds) 
    
    return final_var_interp_ds 

def interpolate_nans(dataset, method='smart'):
    """
    MAIN INTERPOLATION FUNCTION.
    Takes in a dataset and the method of interpolation.
    The method can be either to drop all NaNs or 'smartly' interpolate missing NaNs.

    Parameters:
    -----------
    ds : xr.Dataset
        The input XArray dataset.
    method : str, optional
        The method of interpolation. Default is 'smart'.
        - 'smart': Performs advanced interpolation for missing NaNs.
        - 'drop': Drops all NaN values. (TODO)

    Returns:
    --------
    xr.Dataset
        A new XArray dataset with NaN values filled according to the specified interpolation method.

    Notes:
    ------
    - The 'drop' method drops all NaN values from the dataset.
    - The 'smart' method performs advanced interpolation for missing NaN values, using the `interpolate_nans_smart` helper function.
    - The function checks the percentage of NaN values for each variable in the dataset.
    - It only performs interpolation for variables that have a percentage of NaN values greater than 0.
    - After interpolation, the function checks the percentage of NaN values again for each variable and prints a warning if any NaN values remain.
    """

    # Make a copy of the dataset passed 
    ds = dataset.copy()

    # Get the names of the coordinates and variables as lists of strings:
    all_vars_list, dims_list, variables_list = [], [], []
    for k, v, in ds.variables.items():
        all_vars_list.append(k)
    for k, v, in ds.dims.items():
        dims_list.append(k)
    for el in all_vars_list:
        if el not in dims_list:
            variables_list.append(el)

    # Get percentages of NaNs for variables, only use ones that have percentage > 0%
    interp_vars_list = []
    for i, var in enumerate(variables_list):
        flat_tensor = tensor(ds[var].values.astype(np.float32).flatten())
        perc = percent_nans(flat_tensor)
        if perc > 0:
            interp_vars_list.append(var)
        
    # Interpolate with advanced interpolation 
    if method == 'smart':
        for var in interp_vars_list:
            final_var_interp_ds = interpolate_nans_smart(ds, dims_list, var)
            ds[var].values = final_var_interp_ds[var].values # replace variable values in original ds with interpolated ones 

    elif method == 'drop':
        for var in interp_vars_list:
           # drop missing variable values in original ds (TODO) 
            print("Not yet implemented.")

    # sanity check that all NaNs are interpolated:
    for i, var in enumerate(variables_list):
        flat_tensor = tensor(ds[var].values.astype(np.float32).flatten())
        perc = percent_nans(flat_tensor)
        if perc > 0.0 and method is not None:
            print(f'Warning: {var} has not been properly interpolated')
    
    return ds 
