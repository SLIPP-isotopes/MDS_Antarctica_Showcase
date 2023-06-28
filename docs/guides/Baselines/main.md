# Baselines.main

Authors: Shirley Zhang

Date: June 2023

[Source code](/src/Baselines/scripts/bl_main.py)

This script is designed to train Baseline models.

**NOTES:**
- This script will only use scaled_deseas variables, such as `scaled_deseas_tmp2m` and not `tmp2m`.
- The x-variable will always be d18O_pr (`scaled_deseas_d18O_pr`).
- The script will create predictions for one y-variable and one baseline method at a time.
- It assumes/hardcodes the names of the preprocessed training, validation, and test dataset names.

## Command line flags

When run via the terminal, the script supports the following command line flags:

- `-o` or `--out_dir`: Specifies the output path to save the predictions.
- `-d` or `--data_dir`: Specifies the input path containing the preprocessed data.
- `-v` or `--OUTPUT_VAR_ID`: Specifies the output variable to be predicted (0=hgtprs, 1=pratesfc, 2=tmp2m).
- `-b` or `--BASELINE_METHOD`: Specify the type of baseline to use ("overall_mean", "latlon_mean", "latlonmonth_mean", "ols").
- `-vt` or `--VALTEST`: Specify whether to predict on the validation or test set ("valid", "test").
- `-verbose` or `--VERBOSE`: Specify whether to display an informative message to the console (0=False, 1=True).

## `main` function

This is the main driver function for training Baseline models. 

### Parameters

- `out_dir` (str): The output path where the predictions are saved.
- `data_dir` (str): The input path containing the preprocessed data.
- `OUTPUT_VAR_ID` (int): The output variable to be predicted (0=hgtprs, 1=pratesfc, 2=tmp2m).
- `BASELINE_METHOD` (str): The type of baseline to use ("overall_mean", "latlon_mean", "latlonmonth_mean", "ols").
- `VALTEST` (str): Whether to predict on the validation or test set ("valid", "test").
- `VERBOSE` (int): Whether to display an informative message to the console (0=False, 1=True).

### Returns

- None. 

### Notes

- The available options for `OUTPUT_VAR_ID` are:
  - 0: hgtprs (Geopotential height)
  - 1: pratesfc (surface Precipitation rate)
  - 2: tmp2m (Temp. 2m above surface)

- The available options for `BASELINE_METHOD` are:
  - "overall_mean"
  - "latlon_mean"
  - "latlonmonth_mean"
  - "ols"

- The available options for `VALTEST` are:
    - "valid"
    - "test"

- The available options for `VERBOSE` are:
    - 0: False
    - 1: True

- The output file prefix is generated using the following format (where output_var is one of the three climate variables):

    ```
    bl_{BASELINE_METHOD}_{output_var}_{VALTEST}_preds
    ```

- Additional Information:
  - The validation/test predictions are saved in a NetCDF file named
`bl_{BASELINE_METHOD}_{output_var}_{VALTEST}_preds.nc`.

## `parse_arguments`

This function parses the command line arguments provided when running the script via terminal.

### Returns

- A tuple of parsed arguments as follows:
    - `out_dir` (str): The output path where the predictions are saved.
    - `data_dir` (str): The input path containing the preprocessed data.
    - `OUTPUT_VAR_ID` (int): The output variable to be predicted (0=hgtprs, 1=pratesfc, 2=tmp2m).
    - `BASELINE_METHOD` (str): The type of baseline to use ("overall_mean", "latlon_mean", "latlonmonth_mean", "ols").
    - `VALTEST` (str): Whether to predict on the validation or test set ("valid", "test").
    - `VERBOSE` (int): Whether to display an informative message to the console (0=False, 1=True).