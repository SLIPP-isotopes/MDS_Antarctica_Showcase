# Preprocessing.main

Authors: Jakob Thoms

Date: June 2023

[Source code](/src/Preprocessing/main.py)

This script is designed to preprocess the raw IsoGSM climate data for use in Gaussian Process (GP) and Neural Network (NN) models.
It performs train/valid/test split, feature engineering, deseasonalization, scaling, and saves the resulting preprocessed datasets to disk.

## Command line flags

When run via the terminal, the script supports the following command line flags:

- `-o` or `--out_dir`: Specifies the output directory to save the preprocessed data files. Default is 'data/'.
- `-i` or `--input_dir`: Specifies the input directory to load the raw dataset from. Default is 'data/'.
- `-f` or `--input_file`: Specifies the filename of the raw dataset file. Default is 'IsoGSM/Total.IsoGSM.ERA5.monmean.nc'.
- `-r` or `--return_data`: Specifies whether or not to return the preprocessed datasets in addition to saving them to disk. Default is False.

## `main` function

This is the main driver function for preprocessing. It handles the train/valid/test split, feature engineering, deseasonalization, scaling, and I/O for preprocessing.

### Parameters

- `out_dir` (str, optional): The directory to save the preprocessed data files to. Default is 'data/'.
- `input_dir` (str, optional): The directory to load the raw dataset from. Default is 'data/'.
- `input_file` (str, optional): The filename of the raw dataset file. Default is 'IsoGSM/Total.IsoGSM.ERA5.monmean.nc'.
- `return_data` (bool, optional): Whether or not to return the preprocessed datasets in addition to saving them to disk. Default is False.

### Notes

- Loads the raw dataset from the specified input file.
- Performs train/valid/test split on the raw dataset.
- Calls the `preprocess` function to preprocess each dataset:
  - Deseasonalizes the temporal variables ('d18O_pr', 'hgtprs', 'pratesfc', 'tmp2m').
  - Additional spatial variables are added through feature engineering (polar coords, month, land-sea mask, distance-to-coast, orography).
  - Interpolation of missing data is performed for the training set.
  - All variables are scaled to have *global* mean=0 and stdev=1.
- Saves the preprocessed datasets as NetCDF files in the specified output directory.
- If `return_data` is True, the preprocessed datasets are returned as a tuple.

## `parse_arguments`

This function parses the command line arguments provided when running the script via the terminal.

### Returns

- A tuple of parsed arguments as follows:

  - `out_dir` (str): The output directory to save the preprocessed data files.
  - `input_dir` (str): The input directory to load the raw dataset from.
  - `input_file` (str): The filename of the raw dataset file.
  - `return_data` (bool): Whether or not to return the preprocessed datasets (in addition to saving them to disk).

### Notes

- The default values are set according to the `main()` function's default arguments.
