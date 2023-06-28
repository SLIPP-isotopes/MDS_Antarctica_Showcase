import xarray as xr
import argparse
import sys
sys.path.append('./src/')
from Preprocessing.utils import *

def preprocess(raw_ds, is_training_data, train_ds=None, interp_method='smart'):
    """
    Preprocess a raw XArray dataset of IsoGSM climate data.
    Works for training and validation datasets.
        - Deseasonalizes the temporal variables ('d18O_pr', 'hgtprs', 'pratesfc', 'tmp2m'). Deseasonalization is done using the *training* seasonal means.
        - Additional spatial variables are added through feature engineering (polar coords, month, land-sea mask, distance-to-coast, orography).
        - Interpolation of missing data is performed for the training set. 
        - All variables are scaled to have *global* mean=0 and stdev=1. Scaling is done using the *training* mean and stdev for each variable.
    
    Parameters:
    ----------
    raw_ds : xarray DataSet
        Raw dataset to be processed.
    is_training_data : bool
        Whether or not the input `raw_ds` is training data.
    train_ds : xarray DataSet, optional
        For initializing the scaler/deseasonalization objects when using validation/test data.
        Must be provided if is_training_data is False.
    interp_method : str, optional
        Specify interpolation method for handling missing data.
        Possible values: 'smart' or 'drop'
        
    Returns:
    -------
    xarray DataSet
        Fully preprocessed dataset.
    
    Notes:
    ------
    - If the user supplies an xarray DataArray as `raw_ds`, it is converted to an xarray DataSet.
    - The relevant variables ('d18O_pr', 'hgtprs', 'pratesfc', 'tmp2m') are selected from `raw_ds`.
    - Deseasonalization is performed based on the value of is_training_data:
        - If True, deseasonalization is fit using the dataset (26 years of training data).
        - If False, deseasonalization is fit using attributes from the corresponding preprocessed training dataset.
    - Additional variables are added through feature engineering:
        - Polar coordinates are added.
        - Month and year information is added.
        - Land-sea mask and distance to coast are added.
        - Orography (surface altitude) is added.
    - Interpolation of missing data is performed based on the value of is_training_data. We do not interpolate missing data for the validation/test sets.
    - Variables are scaled using XArrayStandardScaler, either fit with the dataset (is_training_data=True) or initialized with attributes from the preprocessed training dataset (is_training_data=False).
    """
    # CHECK INPUT TYPE
    if isinstance(raw_ds, xr.core.dataarray.DataArray):
        raw_ds = raw_ds.to_dataset(dim='variable')  # If the user supplies an XArray DataArray, convert it to an XArray DataSet.
    if not (isinstance(raw_ds, xr.core.dataset.Dataset)):
        raise TypeError("the input raw_ds should be an XArray DataSet.")
    if not isinstance(is_training_data, bool):
        raise TypeError("is_training_data must be specified as a boolean value (True or False).")
    
    # FILTER FOR RELEVANT VARIABLES, DROP IRRELEVANT DATA
    preprocessed_ds = raw_ds.copy() 
    preprocessed_ds = preprocessed_ds[["d18O_pr", "hgtprs", "pratesfc", "tmp2m"]] # Select key variables (delta-18O, geopotential height, precipitation, & temperature)
    preprocessed_ds = preprocessed_ds.squeeze("levels").drop("levels") # Drop levels dimension
    
    # SHIFT GEOPOTENTIAL HEIGHT
    if not (
        'latitude_2' in [*preprocessed_ds.dims] 
        and 
        'longitude_2' in [*preprocessed_ds.dims]
    ):
        raise ValueError("Invalid input dataset.")
    preprocessed_ds = combine_latlon_grids(preprocessed_ds) # shift geopotential height

    # CHECK DIMENSIONS
    input_check = check_valid_xarray_input(preprocessed_ds)
    if input_check == "transpose lat/lon":
        preprocessed_ds = preprocessed_ds.transpose(
            'time', 'latitude', 'longitude'
        ).to_array().to_dataset(dim='variable') # Forces the dimensions to be (time, lat, lon)
        input_check = check_valid_xarray_input(preprocessed_ds)
    if input_check is not True:
        raise ValueError("Invalid input dataset.")
    
    if is_training_data is False:
        # CHECK INPUT TRAINING DATASET TYPE AND DIMENSIONS
        if train_ds is None:
            raise ValueError("To preprocesses validation/testing data you must supply the corresponding preprocessed training dataset.")
        input_check = check_valid_xarray_input(train_ds)
        if input_check == "tranpose lat/lon":
            train_ds = train_ds.transpose(
                'time', 'latitude', 'longitude'
            ).to_array().to_dataset(dim='variable') # Forces the dimensions to be (time, lat, lon)
            input_check = check_valid_xarray_input(train_ds)
        if input_check is not True:
            raise ValueError("Invalid train dataset. To preprocesses validation/testing data you must supply the corresponding **preprocessed** training dataset.") 
        
    # DESEASONALIZE (delta-18O, geopotential height, precipitation, & temperature) (i.e. the spatial-temporal variables)
    if is_training_data:
        # Calculate the number of years in the training dataset, save as integer
        train_years = int(len(preprocessed_ds.time) / 12)
        # Deseasonalize; Fit using the dataset
        preprocessed_ds = deseasonalize(preprocessed_ds, years=train_years)
    else:
        # Calculate the number of years in the valid/test dataset, save as integer
        valtest_years = int(len(preprocessed_ds.time) / 12)
        # Deseasonalize; Fit using attrs from corresponding preprocessed training dataset
        preprocessed_ds = deseasonalize(preprocessed_ds, years=valtest_years, train_ds=train_ds) 
    
    # ADD MORE VARIABLES (FEATURE ENGINGEERING)
    preprocessed_ds = add_polar_coords(preprocessed_ds)
    preprocessed_ds = add_months(preprocessed_ds)
    preprocessed_ds = add_years(preprocessed_ds)
    
    land_sea_mask_file = 'data/IsoGSM/IsoGSM_land_sea_mask.nc'
    preprocessed_ds = add_land_sea_mask(preprocessed_ds, land_sea_mask_file)
    preprocessed_ds = add_dist_to_coast(preprocessed_ds)
    
    orogrd_file = 'data/IsoGSM/IsoGSM_orogrd.nc'
    preprocessed_ds = add_orogrd(preprocessed_ds, orogrd_file)    
    
    # INTERPOLATE
    if is_training_data: # for test or validation data if the data is missing then it's just missing
        preprocessed_ds = interpolate_nans(preprocessed_ds, interp_method)
        pass
        
    # SCALE VARIABLES (NORMALIZATION)
    scale_feats_list = [
        "d18O_pr", "hgtprs", "pratesfc", "tmp2m", 
        "deseas_d18O_pr", "deseas_hgtprs", "deseas_pratesfc", "deseas_tmp2m", 
        "E", "N", "dist_to_coast", "oro"
    ]
    scaler = XArrayStandardScaler(scale_feats = scale_feats_list)
    if is_training_data:
        scaler.fit(preprocessed_ds)
    else:
        scaler.fit(
            preprocessed_ds, 
            from_attrs=True, 
            attrs=train_ds.attrs # for validation/test data, initialize the scaler w/ parameters extracted from the corresponding train_ds
        ) 
    preprocessed_ds = scaler.transform(preprocessed_ds)
    
    return preprocessed_ds
