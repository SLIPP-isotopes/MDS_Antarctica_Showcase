# Preprocessing.utils

Authors: Jakob Thoms, Daniel Cairns, Shirley Zhang

Date: June 2023

[Source code](/src/Preprocessing/utils.py)

## `combine_latlon_grids(ds)`

Combine latitude_2 and longitude_2 dimensions of the dataset to latitude and longitude dimensions.
Uses linear interpolation and properly accounts for the circular nature of longitude (0 to 360 degrees).

Note: A more correct approach would be to remap the variable using polar coordinates, but this has not been implemented yet.

### Parameters

- `ds` : xarray DataSet
  - Input dataset with latitude_2 and longitude_2 dimensions.

### Returns

- `xarray.DataSet`
  - Dataset with latitude and longitude dimensions after combining latitude_2 and longitude_2 dimensions.

## `check_valid_xarray_input(dataset)`

Check the validity of the input dataset.
It should be an xarray DataSet or xarray DataArray with dimensions ['time', 'latitude', 'longitude'].

### Parameters

- `dataset` : xarray DataSet or xarray DataArray
  - Input dataset to be checked.

### Returns

- `bool` or `str`
  - True if the dataset is valid and has the correct dimensions.
  - "transpose lat/lon" if the dataset had dimensions ['time', 'longitude', 'latitude'] and requires transposing.
  - False if the dataset is invalid or has incorrect dimensions.

## `get_lon_lat_numpy_grids(dataset)`

Given an XArray dataset, return 3D grids (numpy arrays) of longitude and lattitude values.
The input can be an XArray DataArray or an XArray Dataset.
This function transforms an input XArray Dataset or DataArray into 3D grids (numpy arrays) of longitude and latitude values. It retrieves the latitude and longitude values from the input and forms 3D numpy arrays with dimensions representing time, latitude, and longitude.

### Parameters

- `dataset` : xr.Dataset or xr.DataArray
  - The input XArray Dataset or DataArray. In the case of DataArray, the input should have dimensions
        of (variable, time, latitude, longitude). In the case of Dataset, the input should have dimensions
        of (time, latitude, longitude).

### Returns

- `dict`
  - A dictionary with two 3D numpy arrays representing grids of longitude and latitude values.
        The keys of the dictionary are 'longitude' and 'latitude'. The 3D numpy arrays have dimensions
        representing time, latitude, and longitude.

        For example:

        {
            'longitude': lon_grid,
            'latitude': lat_grid
        }

### Notes

This function assumes that the input Dataset or DataArray has dimensions for time, latitude, and longitude.
If these are not present in the input, the function may not behave as expected. Also, the function does not
check whether the values in the longitude and latitude dimensions are actually valid longitude and latitude
values.

## `get_time_numpy_grid(dataset)`

Given an XArray dataset, return 3D grids (numpy arrays) of datetime objects.

### Parameters

- `dataset` : xr.Dataset or xr.DataArray
  - The input XArray Dataset or DataArray.

### Returns

- `numpy.ndarray`
  - A 3D numpy array with dimensions representing time, latitude, and longitude.
        The values in the array are datetime objects.

### Notes

The input can be an XArray DataArray or an XArray Dataset:
    - In the case of DataArray, the input should have dimensions of (variable, time, latitude, longitude).
    - In the case of Dataset, the input should have dimensions of (time, latitude, longitude).

## `get_polar_coords_numpy_grids(dataset)`

Given an XArray dataset, return 3D grids (numpy arrays) of projected E/N values (using UPS projection).

### Parameters

- `dataset` : xr.Dataset or xr.DataArray
  - The input XArray Dataset or DataArray.

### Returns

- `numpy.ndarray, numpy.ndarray`
  - Two 3D numpy arrays representing grids of projected E and N values.
  - The arrays have dimensions representing time, latitude, and longitude.

### Notes

- The input can be an XArray DataArray or an XArray Dataset:
  - In the case of DataArray, the input should have dimensions of (variable, time, latitude, longitude).
  - In the case of Dataset, the input should have dimensions of (time, latitude, longitude).

## `lon_lat_to_polar(lon, lat, hemisphere="south", true_scale_lat=80)`

Given 3D grids (numpy arrays) of longitude and latitude values, create arrays of projected E/N values (using UPS projection).
This is a helper function for `get_polar_coords_numpy_grids`.

### Parameters**

- `lon` : numpy.ndarray
  - A 3D numpy array representing the grid of longitude values.
- `lat` : numpy.ndarray
  - A 3D numpy array representing the grid of latitude values.
- `hemisphere` : str, optional
  - The hemisphere for projection. Valid values are 'north' and 'south'. Default is 'south'.
- `true_scale_lat` : float, optional
  - The true scale latitude for projection. Default is 80.

### Returns

- `numpy.ndarray, numpy.ndarray`
  - Two 3D numpy arrays representing grids of projected E and N values.
  - The arrays have dimensions representing time, latitude, and longitude.


## `datetime_to_months(dates)`

Convert datetime objects to integer months.

This function converts datetime objects to integer months with January as 1 and December as 12.

### Parameters

- `dates` : np.array
  - Numpy array of datetime objects.

### Returns

- `np.array`
  - Numpy array of integer months corresponding to the input datetime objects.


## `datetime_to_years(dates)`

Convert datetime objects to integer years.

This function converts datetime objects to integer years with AD1 (1CE) as 1.

### Parameters

- `dates` : np.array
  - Numpy array of datetime objects.

### Returns

- `np.array`
  - Numpy array of integer years corresponding to the input datetime objects.


## `array_to_flat_tensor(array)`

Convert a numpy array to a flat PyTorch tensor.

This function converts a numpy array (of any shape) into a flat (1D) PyTorch tensor with dtype=float32.

### Parameters
- `array` : np.array
  - Numpy array to be converted.

### Returns

- `torch.Tensor`
  - A 1D (flattened) PyTorch tensor derived from the input numpy array.

### Notes

- The function converts the array data to float32 type before making the conversion to a PyTorch tensor. This is done to save memory when training.

## `add_polar_coords(dataset)`

Add projected E/N values to an XArray dataset (using UPS projection).

This function calculates the projected easting (E) and northing (N) values using the Universal Polar Stereographic (UPS) projection.
The resulting E and N grids are added as variables 'E' and 'N' to the input dataset.

### Parameters

- `dataset` : xr.Dataset or xr.DataArray
  - The input XArray Dataset or DataArray.

### Returns

- `xr.Dataset or xr.DataArray`
  - The modified XArray Dataset or DataArray with added 'E' and 'N' variables.

### Notes

- The input can be an XArray DataArray or an XArray Dataset:
  - In the case of DataArray, the input should have dimensions of (variable, time, latitude, longitude).
  - In the case of Dataset, the input should have dimensions of (time, latitude, longitude).


## `add_months(dataset)`

Add month values (integers) to an XArray dataset (Jan=1, ..., Dec=12).

This function calculates the month values from the datetime objects in the input dataset.
The resulting month grid is added as a variable 'month' to the input dataset.

### Parameters

- `dataset` : xr.Dataset or xr.DataArray
  - The input XArray Dataset or DataArray.

### Returns

- `xr.Dataset or xr.DataArray`
  - The modified XArray Dataset or DataArray with added 'month' variable.

### Notes

- The input can be an XArray DataArray or an XArray Dataset:
  - In the case of DataArray, the input should have dimensions of (variable, time, latitude, longitude).
  - In the case of Dataset, the input should have dimensions of (time, latitude, longitude).
- The month values are calculated from the datetime objects in the 'time' dimension of the dataset.


## `add_years(dataset)`

Add year values (integers) to an XArray dataset (1=AD1=1CE).

This function calculates the year values from the datetime objects in the input dataset.
The resulting year grid is added as a variable 'year' to the input dataset.

### Parameters

- `dataset` : xr.Dataset or xr.DataArray
  - The input XArray Dataset or DataArray.

### Returns

- `xr.Dataset or xr.DataArray`
  - The modified XArray Dataset or DataArray with added 'year' variable.

### Notes

- The input can be an XArray DataArray or an XArray Dataset:
  - In the case of DataArray, the input should have dimensions of (variable, time, latitude, longitude).
  - In the case of Dataset, the input should have dimensions of (time, latitude, longitude).
- The year values are calculated from the datetime objects in the 'time' dimension of the dataset.


## `add_land_sea_mask(dataset, filepath)`

Join land sea mask to xArray dataset from file.

This function reads a land sea mask dataset from a specified file and joins it with the input dataset.
The land sea mask is added as a variable 'landsfc' to the input dataset.

### Parameters

- `dataset` : xr.Dataset
  - The input XArray Dataset.
- `filepath` : str
  - The file path to the land sea mask dataset.

### Returns

- `xr.Dataset`
  - The modified XArray Dataset with added land sea mask variable ('landsfc').

### Notes

- The input dataset should have dimensions of (time, latitude, longitude).
- The land sea mask dataset is expected to have dimensions of latitude and longitude.
- The 'landsfc' variable represents the land-sea mask values as 0s and 1s.


## `add_dist_to_coast(dataset)`

Given an XArray dataset with polar coordinates and a land sea mask, calculate the distance to coast feature for each point.

### Parameters

- `dataset` : xr.Dataset
  - The input XArray dataset with polar coordinates and a land sea mask.

### Returns

- `xr.Dataset`
  - The modified XArray dataset with an additional 'dist_to_coast' variable.

### Notes

- The input dataset must contain variables 'E', 'N', and 'landsfc' representing the polar coordinates and land sea mask.
- The 'dist_to_coast' variable represents the calculated distance to the coast feature for each point in the dataset.
- The calculation is performed by iterating over the points and finding the minimum distance to the opposite masked point.
- The distance is adjusted for sea points by setting them to have negative distance.

## `add_orogrd(dataset, filepath)`

Join the orography variable to an XArray dataset from a file.

This function reads an orography dataset from a specified file and joins it with the input dataset.
The orography is added as a variable 'oro' to the input dataset.

### Parameters

- `dataset` : xr.Dataset
  - The input XArray dataset.
- `filepath` : str
  - The file path to the orography dataset.

### Returns

- `xr.Dataset`
  - The modified XArray Dataset with added orography variable ('oro').

### Notes

- The input dataset should have dimensions of (time, latitude, longitude).
- The orography dataset is expected to have dimensions of latitude and longitude.
- The 'oro' variable represents the surface altitude of a given point.


## `xarray_to_numpy2D(dataset, features)`

Convert an XArray dataset to a 2D numpy array with shape (num_examples, num_features).

The features included in the output numpy array are specified with `features`, which should be a list of strings representing the feature names.
The index of each feature's column in the output numpy array will match the index of the feature's name in the input feature list.

### Parameters

- `dataset` : xr.Dataset or xr.DataArray
  - The input XArray Dataset or DataArray.
- `features` : list of str
  - A list of strings representing the feature names to be included in the output numpy array.

### Returns

- `numpy.ndarray`
  - A 2D numpy array representing the converted dataset. The array has shape (num_examples, num_features).

### Notes

- The input dataset should have dimensions of (variable, time, latitude, longitude) for DataArray or (time, latitude, longitude) for Dataset.
- The features specified in the `features` list will be extracted from the dataset and included as columns in the output numpy array.
- The `features` list can include dimensions (e.g. 'latitude' and 'longitude').
- The dimension 'time' is not recommended as a feature since it contains datetime objects. Instead, use the features 'month' or 'year'.


## `get_feat_numpy_grid(dataset, feat_name)`

Given an XArray dataset, return a 3D grid (numpy array) of a variable in that dataset.

This function extracts the grid of the specified variable from the dataset and returns it as a 3D numpy array with dimensions ('time', 'latitude', 'longitude').

### Parameters

- `dataset` : xr.Dataset or xr.DataArray
  - The input XArray Dataset or DataArray.
- `feat_name` : str
  - The name of the variable to extract from the dataset.

### Returns

- `numpy.ndarray`
  - A 3D numpy array representing the grid of the specified variable in the dataset. The array has shape (num_time_slices, num_lat_slices, num_lon_slices).

### Notes

- The input dataset should have dimensions of (variable, time, latitude, longitude) for DataArray or (time, latitude, longitude) for Dataset.


## `flat_to_grid(dataset, flat_array)`

Given an XArray dataset and a flat array of values, reshape the array into a 3D grid (numpy array) with a shape that is compatible with the XArray dataset.

### Parameters

- `dataset` : xr.Dataset or xr.DataArray
  - The input XArray Dataset or DataArray.
- `flat_array` : numpy.ndarray
  - A 1D numpy array of values.

### Returns

- `numpy.ndarray`
  - A 3D numpy array representing the reshaped grid of values compatible with the XArray dataset.

### Notes

- The input dataset should have dimensions of (variable, time, latitude, longitude) for DataArray or (time, latitude, longitude) for Dataset.

---

## `XArrayStandardScaler`

StandardScaler for XArray.

Similar to StandScaler for scikit-learn ([source](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)).

#### `__init__(self, scale_feats=['tmp2m'])`

Initialize the XArrayStandardScaler.

**Parameters**

- `scale_feats` : list
  - List of feature names to be scaled. Default is `['tmp2m']`.

**Attributes**

- `scale_feats` : list
  - List of feature names to be scaled.
- `mean` : numpy.ndarray or None
  - Array of feature means for scaling.
- `stdev` : numpy.ndarray or None
  - Array of feature standard deviations for scaling.

#### `_is_initialized(self)`

Check if the XArrayStandardScaler object is initialized.

**Returns**

- `bool`
  - True if scaler is initialized, False otherwise.

#### `initialize(self, dataset, from_attrs=False, attrs=None)`

Initialize the scaler with mean and standard deviation values.

**Parameters**

- `dataset` : xr.Dataset or xr.DataArray
  - The input XArray dataset or data array.
- `from_attrs` : bool, optional
  - Whether to initialize from attributes. Default is False.
- `attrs` : dict or None, optional
  - Dictionary of attributes for initializing from attributes. Default is None.
  - If None, the function will retrieve the dictionary from the input dataset.

**Notes**

- If `from_attrs` is True, the mean and standard deviation values are retrieved from the dataset attributes.
- If `from_attrs` is False, the mean and standard deviation values are calculated from the dataset.


#### `fit(self, dataset, from_attrs=False, attrs=None)`

Alias for the `initialize` method.


#### `transform(self, dataset)`

Transform the dataset by scaling the specified features.

**Parameters**

- `dataset` : xr.Dataset or xr.DataArray
  - The input XArray dataset or data array to be transformed.

**Returns**

- `xr.Dataset`
  - A new XArray dataset with the scaled features.


#### `fit_transform(self, dataset)`

Fit and transform the dataset in a single step.

**Parameters**

- `dataset` : xr.Dataset or xr.DataArray
  - The input XArray dataset or data array to be transformed.

**Returns**

- `xr.Dataset`
  - A new XArray dataset with the scaled features.

#### `inverse_transform(self, scaled_dataset)`

Inverse transform the scaled dataset to obtain the original values.

**Parameters**

- `scaled_dataset` : xr.Dataset or xr.DataArray
  - The input XArray dataset or data array with scaled features.

**Returns**

- `xr.Dataset`
  - A new XArray dataset with the original unscaled features.


---

## `deseasonalize(ds, years, train_ds=None)`

Given an XArray dataset, return a new dataset with monthly means and deseasonalized variables as new variables in the dataset.

Assumes that the input `ds` starts perfectly at January of one year and ends in December of another (i.e., full years).

### Parameters

- `ds` : xr.Dataset
  - The input XArray dataset.
- `years` : int
  - The number of years covered by the dataset. This parameter should match the actual number of years in the dataset.
- `train_ds` : xr.Dataset or None, optional
  - The training dataset used to compute seasonal means. If provided, the function uses the seasonal means from the training dataset.
  - If not provided, the function computes the seasonal means from the input `ds` itself.

### Returns

- `xr.Dataset`
  - A new XArray dataset with monthly means and deseasonalized variables.

### Notes

- The function computes deseasonalized variables by subtracting the seasonal means from the original variables.
- The deseasonalized variables are added as new variables with names 'deseas_{var}' in the new dataset, where '{var}' is the original variable name.
- The monthly means (seasonal means) are also added as new attributes to the new dataset with names 'seasonal_means_{var}'.
- If `train_ds` is provided, the function uses the seasonal means from the training dataset to compute deseasonalized variables.
- If `train_ds` is not provided, the function computes the seasonal means from the input `ds` itself.

## `percent_nans(flat_tensor)`

Takes in a flattened tensor of values and returns the percentage of NaNs in the tensor as a float.

This is a helper function for interpolate_nans().

### Parameters

- `flat_tensor` : torch.Tensor
  - A flattened tensor representing a variable.

### Returns

- `float`
  - The percentage of NaN values in the flattened tensor.

## `interpolate_nans_smart(ds, dims_list, var)`

Takes in a dataset and a variable to interpolate, and does advanced interpolation along 3 dimensions to fill missing values.

This is a helper function for interpolate_nans().

### Parameters

- `ds` : xr.Dataset
  - The input XArray dataset.
- `dims_list` : list
  - A list of dimension names in the dataset, in the order [time, latitude, longitude].
- `var` : str
  - The name of the variable to interpolate.

### Returns

- `xr.Dataset`
  - A new XArray dataset with NaNs filled for the specified variable.

### Notes

- The function performs advanced interpolation for the specified variable along the time, latitude, and longitude dimensions.
- It identifies the indices of NaN values in the variable and performs interpolation only for those indices.
- The interpolated values are combined with the original dataset, replacing the NaN values.

## `interpolate_nans(ds, method='smart')`
The main interpolation function. Takes in a dataset and the method of interpolation.
The method can be either to drop all NaNs or 'smartly' interpolate missing NaNs.

### Parameters

- `ds` : xr.Dataset
  - The input XArray dataset.
- `method` : str, optional
  - The method of interpolation. Default is 'smart'.
  - 'smart': Performs advanced interpolation for missing NaNs.
  - 'drop': Drops all NaN values. (TODO)

### Returns

- `xr.Dataset`
  - A new XArray dataset with NaN values filled according to the specified interpolation method.

**Notes**
- The 'drop' method drops all NaN values from the dataset.
- The 'smart' method performs advanced interpolation for missing NaN values, using the `interpolate_nans_smart` helper function.
- The function checks the percentage of NaN values for each variable in the dataset.
- It only performs interpolation for variables that have a percentage of NaN values greater than 0.
- After interpolation, the function checks the percentage of NaN values again for each variable and prints a warning if any NaN values remain.
