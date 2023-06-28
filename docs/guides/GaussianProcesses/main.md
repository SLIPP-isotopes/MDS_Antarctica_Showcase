# GaussianProcesses.main

Authors: Jakob Thoms, Shirley Zhang

Date: June 2023

[Source code](/src/GaussianProcesses/scripts/gp_main.py)

This script is designed to train Gaussian Process (GP) models.

## Command line flags

When run via the terminal, the script supports the following command line flags:

- `-o` or `--out_dir`: Specifies the output file name/path.
- `-d` or `--data_dir`: Specifies the input file name/path.
- `-v` or `--OUTPUT_VAR_ID`: Specifies the output variable to be predicted (0=hgtprs, 1=pratesfc, 2=tmp2m).
- `-k` or `--KERNEL_CONFIG_ID`: Specifies the kernel configuration (integer between 0 and 7).
- `-n` or `--TRAIN_SPLIT_N`: Specifies the number of splits in the training set. Default is 13.
- `-s` or `--TRAIN_SPLIT_ID`: Specifies the split of the training set (integer between 0 and 12).
- `-e` or `--NUM_EPOCHS`: Specifies the total number of epochs (integer greater than 1). Default is 10.
- `-l` or `--LEARNING_RATE`: Specifies the learning rate (float greater than 0). Default is 0.15.
- `-r` or `--NUM_RESTARTS`: Specifies the number of restarts for warmup (integer greater than 0). Default is 4.
- `-er` or `--NUM_EPOCHS_PER_RESTART`: Specifies the number of epochs per restart during warmup (integer greater than 0). Default is 5.
- `-m` or `--MEAN_COMPONENT`: Specifies the mean component for the GP model.
- `-pu` or `--PROCESSING_UNIT`: Specifies the device (CPU or GPU) for training the GP model.

## `main` function

This is the main driver function for training GP models. 

### Parameters

- `out_dir` (str): The output file name/path.
- `data_dir` (str): The input file name/path.
- `OUTPUT_VAR_ID` (int): The output variable to be predicted (0=hgtprs, 1=pratesfc, 2=tmp2m).
- `TRAIN_SPLIT_N` (int): The number of splits in the training set.
- `TRAIN_SPLIT_ID` (int): The split of the training set to be used (integer between 0 and TRAIN_SPLIT_N - 1).
- `KERNEL_CONFIG_ID` (int): The kernel configuration (integer between 0 and 7).
- `NUM_EPOCHS` (int, optional): The total number of epochs (integer greater than 1). Default is 10.
- `LEARNING_RATE` (float, optional): The learning rate (float greater than 0). Default is 0.15.
- `NUM_RESTARTS` (int, optional): The number of restarts for warmup (integer greater than 0). Default is 4.
- `NUM_EPOCHS_PER_RESTART` (int, optional): The number of epochs per restart during warmup (integer greater than 0). Default is 5.
- `MEAN_COMPONENT` (str, optional): The mean component for the GP model.
- `PROCESSING_UNIT` (str, optional): The device (CPU or GPU) for training the GP model.

### Returns

- None or dict: If `return_saved_data` is True, returns a dictionary of saved data. Otherwise, returns None.

### Notes

- The available options for `OUTPUT_VAR_ID` are:
  - 0: hgtprs (Geopotential height)
  - 1: pratesfc (surface Precipitation rate)
  - 2: tmp2m (Temp. 2m above surface)

- The available options for `KERNEL_CONFIG_ID` are:
  - 0: RBF + PeriodicKernel for month
  - 1: RBF (deseasonalized; no month)
  - 2: PiecewisePolynomialKernel + PeriodicKernel for month
  - 3: PiecewisePolynomialKernel (deseasonalized; no month)
  - 4: RQKernel + PeriodicKernel for month
  - 5: RQKernel (deseasonalized; no month)
  - 7: SpectralMixture (deseasonalized; no month)


- The output file prefix is generated using the following format:

    ```
    gp_{output_feat_name}_k{KERNEL_CONFIG_ID}_m{MEAN_ID}_n{TRAIN_SPLIT_N}_s{TRAIN_SPLIT_ID}_e{NUM_EPOCHS}_l{LEARNING_RATE}_{PROCESSING_UNIT}
    ```

- Additional Information:
  - The model state is saved in a file named `gp_{output_feat_name}_k{KERNEL_CONFIG_ID}_m{MEAN_ID}_n{TRAIN_SPLIT_N}_s{TRAIN_SPLIT_ID}_e{NUM_EPOCHS}_l{LEARNING_RATE}_{PROCESSING_UNIT}_model_state.pth`.
  - The validation predictions are saved in a NetCDF file named `gp_{output_feat_name}_k{KERNEL_CONFIG_ID}_m{MEAN_ID}_n{TRAIN_SPLIT_N}_s{TRAIN_SPLIT_ID}_e{NUM_EPOCHS}_l{LEARNING_RATE}_{PROCESSING_UNIT}_valid_preds.nc`.

## `parse_arguments`

This function parses the command line arguments provided when running the script via terminal.

### Returns

- A tuple of parsed arguments as follows:
  - `out_dir` (str): The output file name/path.
  - `data_dir` (str): The input file name/path.
  - `OUTPUT_VAR_ID` (int): The output variable to be predicted (0=hgtprs, 1=pratesfc, 2=tmp2m).
  - `TRAIN_SPLIT_N` (int): The number of splits in the training set.
  - `TRAIN_SPLIT_ID` (int): The split of the training set to be used (integer between 0 and TRAIN_SPLIT_N - 1).
  - `KERNEL_CONFIG_ID` (int): The kernel configuration (integer between 0 and 7).
  - `NUM_EPOCHS` (int): The total number of epochs (integer greater than 1).
  - `LEARNING_RATE` (float): The learning rate (float greater than 0).
  - `NUM_RESTARTS` (int): The number of restarts for warmup (integer greater than 0).
  - `NUM_EPOCHS_PER_RESTART` (int): The number of epochs per restart during warmup (integer greater than 0).
  - `MEAN_COMPONENT` (str): The mean component for the GP model.
  - `PROCESSING_UNIT` (str): The device (CPU or GPU) for training the GP model.
