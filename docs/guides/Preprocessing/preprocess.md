# Preprocessing.preprocess

Authors: Jakob Thoms, Daniel Cairns, Shirley Zhang

Date: June 2023

[Source code](/src/Preprocessing/preprocess.py)

## `preprocess`

This function preprocesses a raw XArray dataset of IsoGSM climate data, for both training and validation datasets. It deseasonalizes temporal variables, adds spatial variables through feature engineering, interpolates missing data for training sets, and scales all variables to have a global mean of 0 and a standard deviation of 1.

### Parameters

- `raw_ds` : (xarray DataSet) Raw dataset to be processed.
- `is_training_data` : (bool) Whether or not the input `raw_ds` is training data.
- `train_ds` : (xarray DataSet, optional) For initializing the scaler/deseasonalization objects when using validation/test data. Must be provided if is_training_data is False.
- `interp_method` : (str, optional) Specify interpolation method for handling missing data. Possible values: 'smart' or 'drop'

### Returns

- (xarray DataSet) Fully preprocessed dataset.

### Notes

The preprocessing process includes several steps, such as deseasonalization, feature engineering, missing data interpolation, and variable scaling.
