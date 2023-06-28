# GaussianProcesses.utils

Author: Jakob Thoms

Date: June 2023

[Source code](/src/GaussianProcesses/utils.py)

## `wrangle`

This function loads and preprocesses data for Gaussian Process (GP) models. It prepares the input features and output variable from the given preprocessed datasets. The data is wrangled into a format suitable for training and validating GP models.

### Parameters

- `data_dir` : (str, optional) Directory path for the preprocessed data files, by default "/arc/project/st-aorsi-1/data/preprocessed/"
- `output_var` : (int, optional) Index of the output variable, by default 2. The available options are:
  - 0: hgtprs (Geopotential height)
  - 1: pratesfc (surface Precipitation rate)
  - 2: tmp2m (Temp. 2m above surface)
- `kernel_config_id` : (int, optional) ID of the kernel configuration, by default 0. The kernel configuration determines whether to deasonalize the data and include certain variables. Even config IDs use month as a variable and don't deseasonalize, while odd config IDs deasonalize the data and don't include month as a variable.

### Returns

- `tuple`: Tuple containing the wrangled data and related information.

### Note

The function assumes that the preprocessed training and validation datasets are stored in NetCDF format (.nc) files. The naming convention for the files is "preprocessed_train_ds.nc" and "preprocessed_valid_ds.nc" respectively.

### Example Usage

```python
train_X, train_Y, valid_X, valid_Y, _, _, _, output_feature = wrangle(data_dir="data/", output_var=1, kernel_config_id=3)
```

## `reduce_training_size`

Return a split consisting of 1/train_split_nth of the training dataset. Non-randomized (deterministic) splits. (reproducible + maintains temporal ordering).

### Parameters

- `train_Y_no_nans_flat` : (numpy.ndarray) The flattened training target array.
- `train_X_no_nans` : (numpy.ndarray) The training feature matrix.
- `train_split_n` : (int, optional) The number of splits to create. Default is 9.
- `train_split_id` : (int, optional) The ID of the split to return. Default is 0.
- `downscale_data` : (bool, optional) Whether to downscale the data. Default is True.
- `quick_test_run` : (bool, optional) Whether to perform a quick test run. Only use for running quick local tests. Default is False.

### Returns

- `numpy.ndarray` - The split of the training feature matrix.
- `numpy.ndarray` - The split of the flattened training target array.

## `flat_validation_predictions_to_xarray`

Convert flat validation predictions back to XArray format. This function takes the flat predictions generated during validation and converts them back to the original XArray format. It also predicts NaN for locations where the input data was also NaN (indicating missing values in the validation set) (if it's missing, it's missing for validation).

### Parameters

- `valid_pred` : (torch.Tensor) Flat validation predictions.
- `rand_uniform_pred` : (torch.Tensor) Flat random uniform predictions.
- `rand_normal_pred` : (torch.Tensor) Flat random normal predictions.
- `valid_Y` : (numpy.ndarray) Original validation data.
- `valid_ds` : (xarray.Dataset) XArray dataset containing the validation data.
- `valid_nan_mask` : (numpy.ndarray) Mask indicating the locations of NaN values in the validation data.
- `output_feat_name` : (str) Name of the output feature.
- `kernel_config_id` : (int) ID of the kernel configuration.
- `train_split_id` : (int) ID of the training split.

### Returns

- `xarray.Dataset` - XArray dataset containing the converted validation predictions.

## `downscale`

Downscale the training data for a GP model by removing some of the examples 
which are close to the centre of Antarctica or far from the centre of Antarctica.

Given a 2D numpy array of GP training inputs and a 1D numpy array of GP training outputs, 
downscale the training data by removing two thirds of the examples which are 'close'
to the centre of Antarctica and one half of the examples which are 'far' 
from the centre of Antarctica. 

In this context, 'close' means that the example's UPS Easting and UPS Northing
coordinates are both less than half a standard deviation away from the centre of Antarctica.
In this context, 'far' means that at least one of the example's UPS Easting or UPS Northing
coordinates is more than two standard deviations away from the centre of Antarctica.

We remove some points near the centre because the data was sampled on a homogenous
latitude/longitude grid, which means that there is a much greater concentration of 
sampled points near the south pole (i.e. near the centre of Antactica). 
The concentration of examples near the south pole is so high that we can discard 
two thirds of these examples and still have sufficient data to train a GP model.

We remove some points far from the centre because these points are located 
at the edge of the Southern Ocean and as such are not as useful/important for
training a GP model to make predictions in the continent of Antarctica. 

### Parameters

- `train_X_no_nans` : (2D numpy array) The GP training inputs (with NaN entries removed)
- `train_Y_no_nans_flat` : (1D numpy array) The GP training outputs (with NaN entries removed)
- `verbose` : (bool) Whether or not to print the number of examples dropped during downscaling.

### Returns

 - `tuple` of (2D numpy array, 1D numpy array) - The GP training inputs and outputs after downscaling.

## `RMSE`

Calculate a model's Root Mean Square Error (RMSE). Given the actual and predicted values, the function calculates the mean square error (MSE) using the nn.MSELoss function from PyTorch and then returns the square root of the MSE, which represents the RMSE.

### Parameters

- `actual` : (numpy array or PyTorch tensor) The actual values.
- `pred` : (numpy array or PyTorch tensor) The predicted values.

### Returns

 - `float` - The Root Mean Square Error (RMSE) between the actual and predicted values.
