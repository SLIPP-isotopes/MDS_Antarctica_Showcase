import xarray as xr
import numpy as np
import torch
from torch import tensor, nn

from Preprocessing.utils import xarray_to_numpy2D, flat_to_grid

def wrangle(data_dir="/arc/project/st-aorsi-1/data/preprocessed/", output_var=2, kernel_config_id=0):
    """
    Wrangle data for GP models.

    This function loads and preprocesses data for Gaussian Process (GP) models. 
    It prepares the input features and output variable from the given preprocessed datasets. 
    The data is wrangled into a format suitable for training and validating GP models.

    Parameters
    ----------
    data_dir : str, optional
        Directory path for the preprocessed data files, by default "/arc/project/st-aorsi-1/data/preprocessed/"
    output_var : int, optional
        Index of the output variable, by default 2. The available options are:
        - 0: hgtprs (Geopotential height)
        - 1: pratesfc (surface Precipitation rate)
        - 2: tmp2m (Temp. 2m above surface)
    kernel_config_id : int, optional
        ID of the kernel configuration, by default 0. 
        The kernel configuration determines whether to deasonalize the data and include certain variables. 
        Even config IDs use month as a variable and don't deseasonalize, while odd config IDs deasonalize the data and don't include month as a variable.

    Returns
    -------
    tuple
        Tuple containing the wrangled data and related information:
        - train_X_no_nans: 2D tensor of GP training inputs after preprocessing, with NaN entries removed.
        - train_Y_no_nans_flat: 1D tensor of GP training outputs after preprocessing, with NaN entries removed and flattened.
        - valid_X_no_nans: 2D tensor of validation inputs after preprocessing, with NaN entries removed.
        - valid_Y_no_nans_flat: 1D tensor of validation outputs after preprocessing, with NaN entries removed and flattened.
        - valid_Y: 2D tensor of original validation outputs.
        - valid_ds: xarray Dataset of the preprocessed validation data.
        - valid_nan_mask: Boolean tensor indicating NaN entries in the validation inputs.
        - output_feat: Name of the output feature used for the GP model.

    Note
    ----
    The function assumes that the preprocessed training and validation datasets are stored in NetCDF format (.nc) files. 
    The naming convention for the files is "preprocessed_train_ds.nc" and "preprocessed_valid_ds.nc" respectively. 
    The datasets should contain the necessary input features and output variables according to the kernel configuration and output variable index.

    Example usage:
    train_X, train_Y, valid_X, valid_Y, _, _, _, output_feature = wrangle(data_dir="data/", output_var=1, kernel_config_id=3)
    """
    
    # LOAD AND WRANGLE DATA:
    train_ds = xr.open_dataset(data_dir + "preprocessed_train_ds.nc")
    valid_ds = xr.open_dataset(data_dir + "preprocessed_valid_ds.nc")

    # The GP model's output features (the columns of Y):
    if kernel_config_id % 2 == 0: # Don't deasonalize the data; use a periodic kernel with the month variable.
        output_feats = ['scaled_hgtprs', 'scaled_pratesfc', 'scaled_tmp2m']
    elif kernel_config_id % 2 == 1: # Deasonalize the data; don't include month as a variable.
        output_feats = ['scaled_deseas_hgtprs', 'scaled_deseas_pratesfc', 'scaled_deseas_tmp2m']
    # For now, the GP model can only handle one output variable at a time.
    # # OUTPUT_VAR = 0  ## hgtprs (Geopotential height)
    # # OUTPUT_VAR = 1  ## pratesfc (surface Precipitation rate)
    # OUTPUT_VAR = 2  ## tmp2m (Temp. 2m above surface)
    output_feat = output_feats[output_var]
    output_feat_name = output_feat.split("_")[-1]

    # The GP model's input features (the columns of X):
    # Kernels with even config_id use month as a variable and don't deseasonalize.
    # Kernels with odd config_id deasonalize and don't use month as a variable.
    if kernel_config_id % 2 == 0: # Don't deasonalize the data; use a periodic kernel with the month variable.
        input_feats = ['scaled_d18O_pr', 'scaled_E', 'scaled_N', 'scaled_oro', 'scaled_dist_to_coast', 'month']
    elif kernel_config_id % 2 == 1: # Deasonalize the data; don't include month as a variable.
        input_feats = ['scaled_deseas_d18O_pr', 'scaled_E', 'scaled_N', 'scaled_oro', 'scaled_dist_to_coast']

    # Convert to a 2D numpy array. Rows are observations, columns are features. 
    train_X = xarray_to_numpy2D(train_ds, features=input_feats)
    valid_X = xarray_to_numpy2D(valid_ds, features=input_feats)

    # Convert numpy array to a pytorch tensor
    train_X = tensor(train_X.astype(np.float32))
    valid_X = tensor(valid_X.astype(np.float32))


    # Convert to a 2D numpy array. Rows are observations, columns are features. 
    train_Y = xarray_to_numpy2D(train_ds, features=output_feat)
    valid_Y = xarray_to_numpy2D(valid_ds, features=output_feat)

    # Convert numpy array to a pytorch tensor
    train_Y = tensor(train_Y.astype(np.float32))
    valid_Y = tensor(valid_Y.astype(np.float32))

    # Drop NAs (if any)
    train_nan_mask = torch.any(train_X.isnan(), dim=1) | torch.any(np.isnan(train_Y), dim=1)
    train_nan_mask = tensor(train_nan_mask.numpy().astype(bool))
    valid_nan_mask = torch.any(valid_X.isnan(), dim=1) | torch.any(np.isnan(valid_Y), dim=1)
    valid_nan_mask = tensor(valid_nan_mask.numpy().astype(bool))
    train_X_no_nans = train_X[~train_nan_mask]
    train_Y_no_nans = train_Y[~train_nan_mask]
    valid_X_no_nans = valid_X[~valid_nan_mask]
    valid_Y_no_nans = valid_Y[~valid_nan_mask]

    # Flatten tensors (univariate output)
    train_Y_no_nans_flat = train_Y_no_nans.flatten()
    valid_Y_no_nans_flat = valid_Y_no_nans.flatten()
    
    return train_X_no_nans, train_Y_no_nans_flat, \
        valid_X_no_nans, valid_Y_no_nans_flat, \
        valid_Y, valid_ds, valid_nan_mask, output_feat

def reduce_training_size(train_Y_no_nans_flat, train_X_no_nans, train_split_n=9, train_split_id=0, downscale_data=True, quick_test_run=False):
    """
    Return a split consisting of 1/train_split_nth of the training dataset.
    Non-randomized (deterministic) splits. (reproducible + maintains temporal ordering).

    Parameters
    ----------
    train_Y_no_nans_flat : numpy.ndarray
        The flattened training target array.
    train_X_no_nans : numpy.ndarray
        The training feature matrix.
    train_split_n : int, optional
        The number of splits to create. Default is 9.
    train_split_id : int, optional
        The ID of the split to return. Default is 0.
    downscale_data : bool, optional
        Whether to downscale the data. Default is True.
    quick_test_run : bool, optional
        Whether to perform a quick test run. Only use for running quick local tests. Default is False.

    Returns
    -------
    numpy.ndarray
        The split of the training feature matrix.
    numpy.ndarray
        The split of the flattened training target array.
    """
    if downscale_data:
        train_X_no_nans, train_Y_no_nans_flat = downscale(train_X_no_nans, train_Y_no_nans_flat)
    
    n_train_examples_total = train_Y_no_nans_flat.shape[0]
    n_train_examples_per_split = n_train_examples_total // train_split_n

    i = train_split_id
    first_index = i * n_train_examples_per_split
    last_index = (i + 1) * n_train_examples_per_split

    if quick_test_run:
        first_index, last_index = first_index // 100, last_index // 100 # For running locally
        print("!!WARNING!!\n\t quick_test_run=True\n!!WARNING!!")

    train_Y_no_nans_flat_split = train_Y_no_nans_flat[first_index:last_index]
    train_X_no_nans_split = train_X_no_nans[first_index:last_index, :]

    print(f'Training using examples with indices between {first_index} (inclusive) and {last_index} (exclusive)')

    return train_X_no_nans_split, train_Y_no_nans_flat_split

def flat_validation_predictions_to_xarray(valid_pred, rand_uniform_pred, rand_normal_pred, valid_Y, valid_ds, valid_nan_mask, output_feat_name, kernel_config_id, train_split_id):
    """
    Convert flat validation predictions back to XArray format.
    This function takes the flat predictions generated during validation and converts them back to the original XArray format. 
    It also predicts NaN for locations where the input data was also NaN (indicating missing values in the validation set) (if it's missing, it's missing for validation).

    Parameters
    ----------
    valid_pred : torch.Tensor
        Flat validation predictions.
    rand_uniform_pred : torch.Tensor
        Flat random uniform predictions.
    rand_normal_pred : torch.Tensor
        Flat random normal predictions.
    valid_Y : numpy.ndarray
        Original validation data.
    valid_ds : xarray.Dataset
        XArray dataset containing the validation data.
    valid_nan_mask : numpy.ndarray
        Mask indicating the locations of NaN values in the validation data.
    output_feat_name : str
        Name of the output feature.
    kernel_config_id : int
        ID of the kernel configuration.
    train_split_id : int
        ID of the training split.

    Returns
    -------
    xarray.Dataset
        XArray dataset containing the converted validation predictions.
    """
    valid_pred_numpy = valid_pred.numpy()
    valid_pred_flat_grid = np.zeros_like(valid_Y)
    valid_pred_flat_grid[~valid_nan_mask] = valid_pred_numpy[:, np.newaxis]
    valid_pred_flat_grid[valid_nan_mask] = np.nan
    valid_pred_grid = flat_to_grid(valid_ds, valid_pred_flat_grid)

    rand_uniform_pred_numpy = rand_uniform_pred.numpy()
    rand_uniform_pred_flat_grid = np.zeros_like(valid_Y)
    rand_uniform_pred_flat_grid[~valid_nan_mask] = rand_uniform_pred_numpy[:, np.newaxis]
    rand_uniform_pred_flat_grid[valid_nan_mask] = np.nan
    rand_uniform_pred_grid = flat_to_grid(valid_ds, rand_uniform_pred_flat_grid)

    rand_normal_pred_numpy = rand_normal_pred.numpy()
    rand_normal_pred_flat_grid = np.zeros_like(valid_Y)
    rand_normal_pred_flat_grid[~valid_nan_mask] = rand_normal_pred_numpy[:, np.newaxis]
    rand_normal_pred_flat_grid[valid_nan_mask] = np.nan
    rand_normal_pred_grid = flat_to_grid(valid_ds, rand_normal_pred_flat_grid)

    if kernel_config_id % 2 == 0:
        output_feat_name = 'scaled_' + output_feat_name
    elif kernel_config_id % 2 == 1:
        output_feat_name = 'scaled_deseas_' + output_feat_name
    valid_preds_ds = valid_ds.assign(pred = (['time', 'latitude', 'longitude'], valid_pred_grid)).rename({
        'pred': f'pred_{output_feat_name}'
    }).assign(pred = (['time', 'latitude', 'longitude'], rand_uniform_pred_grid)).rename({
        'pred': f'pred_{output_feat_name}__uniform_random_deltas'
    }).assign(pred = (['time', 'latitude', 'longitude'], rand_normal_pred_grid)).rename({
        'pred': f'pred_{output_feat_name}__normal_random_deltas'
    })

    valid_preds_ds = valid_preds_ds.assign_attrs({
                'kernel_config_id': kernel_config_id,
                'train_split_id': train_split_id,
    })


    return valid_preds_ds

        
def RMSE(actual, pred):
    """
    Calculate a model's Root Mean Square Error (RMSE).

    Given the actual and predicted values, the function calculates the mean square error (MSE)
    using the nn.MSELoss function from PyTorch and then returns the square root of the MSE,
    which represents the RMSE.

    Parameters
    ----------
    actual : numpy array or PyTorch tensor
        The actual values.
    pred : numpy array or PyTorch tensor
        The predicted values.

    Returns
    -------
    float
        The Root Mean Square Error (RMSE) between the actual and predicted values.
    """
    MSE = nn.MSELoss()
    return np.sqrt(MSE(actual, pred))

def downscale(train_X_no_nans, train_Y_no_nans_flat, verbose=True):
    """
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

    Parameters
    ----------
    train_X_no_nans : 2D numpy array
        The GP training inputs (with NaN entries removed)
    train_Y_no_nans_flat : 1D numpy array
        The GP training outputs (with NaN entries removed)
    verbose : bool
        Whether or not to print the number of examples dropped during downscaling.
        
    Returns
    -------
    tuple of (2D numpy array, 1D numpy array)
        The GP training inputs and outputs after downscaling.  
    """
    np.random.seed(0) # Set the RNG seed for reproducibility 

    # True if the example is 'close' to the centre of Antarctica 
    close_to_center_mask = (
        (abs(train_X_no_nans.numpy()[:, 1]) < 0.5) # Scaled UPS Easting < 0.5 stdevs
        & 
        (abs(train_X_no_nans.numpy()[:, 2]) < 0.5) # Scaled UPS Northing < 0.5 stdevs
    ).astype(bool)

    # True for two thirds of all examples (at random w/ set seed)
    random_mask_inner = (np.random.rand(train_X_no_nans.shape[0]) > 0.3).astype(bool)

    # True for 'close' to centre examples which will be discarded.
    downscale_mask_inner = (close_to_center_mask & random_mask_inner)


    # True if the example is 'far' from the centre of Antarctica 
    far_from_center_mask = (
        (abs(train_X_no_nans.numpy()[:, 1]) > 2.0)
        | 
        (abs(train_X_no_nans.numpy()[:, 2]) > 2.0)
    ).astype(bool)

    # True for one half of all examples (at random w/ set seed)
    random_mask_outer = (np.random.rand(train_X_no_nans.shape[0]) > 0.5).astype(bool)

    # True for 'far' from centre examples which will be discarded.
    downscale_mask_outer = (far_from_center_mask & random_mask_outer)

    # True for examples to keep, False for examples to discard
    downscale_mask = ~(downscale_mask_inner | downscale_mask_outer)

    # Apply the boolean mask to downscale the data
    train_X_no_nans_downscaled = train_X_no_nans[downscale_mask]
    train_Y_no_nans_flat_downscaled = train_Y_no_nans_flat[downscale_mask]

    if verbose:
        n_examples_dropped = (~downscale_mask).sum()
        print(f'\n\t {n_examples_dropped} examples dropped during downscaling.\n')

    return train_X_no_nans_downscaled, train_Y_no_nans_flat_downscaled
