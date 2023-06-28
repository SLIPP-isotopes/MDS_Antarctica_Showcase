# Postprocessing.main

Authors: Jakob Thoms

Date: June 2023

[Source code](/src/Postprocessing/main.py)

This script is designed to evaluate models and calculate metrics for Gaussian Process (GP) and Neural Network (NN) models.
It performs calculations, generates plots, and saves results for both types of models.

## Command line flags

When run via the terminal, the script supports the following command line flags:

- `-d` or `--data_dir`: Specifies the directory to load the prediction data from. This flag is required.
- `-o` or `--out_dir`: Specifies the directory to save the residual diagnostic results to. This flag is required.
- `-v` or `--OUTPUT_VAR_IDS`: Specifies the output variable(s) to be evaluated. Accepts integer values representing the variable IDs. Default is `[0, 1, 2]` which corresponds to `hgtprs`, `pratesfc`, and `tmp2m` respectively.
- `-k` or `--KERNEL_CONFIG_IDS`: Specifies the GP kernel ID(s) to evaluate. Accepts integer values. Default is `[0, 1]`.
- `-n` or `--TRAIN_SPLIT_N`: Specifies the number of splits in the training set for GP models. Accepts an integer value. Default is `6`.
- `-sgp` or `--TRAIN_SPLIT_ID`: Specifies the split ID of the training set to process for GP models. Accepts an integer value. Default is `3`.
- `-egp` or `--NUM_EPOCHS_GP`: Specifies the number of epochs for training GP models. Accepts an integer value. Default is `10`.
- `-lgp` or `--LEARNING_RATE_GP`: Specifies the learning rate for training GP models. Accepts a floating-point value. Default is `0.001`.
- `-m` or `--MEAN_ID`: Specifies the ID of the mean component for GP models. Accepts an integer value. Default is `1`.
- `-pu` or `--PROCESSING_UNIT`: Specifies the device for training the GP models. Accepts a string value. Default is `"cpu"`.
- `-a` or `--ARCHITECTURES`: Specifies the NN architecture(s) to evaluate. Accepts a list of strings. Default is `['foo', 'bar']`.
- `-enn` or `--NUM_EPOCHS_NN`: Specifies the number of epochs for training NN models. Accepts an integer value. Default is `10`.
- `-lnn` or `--LEARNING_RATE_NN`: Specifies the learning rate for training NN models. Accepts a floating-point value. Default is `0.001`.
- `-snn` or `--SEED`: Specifies the seed used for random initialization of weights for NN models. Accepts an integer value. Default is `123`.

Note:

- The available options for `OUTPUT_VAR_IDS` are:
  - `0`: hgtprs (Geopotential height)
  - `1`: pratesfc (surface Precipitation rate)
  - `2`: tmp2m (Temp. 2m above surface)


## `main` function

This is the main driver function for postprocessing. It performs the evaluation and metric calculations for both GP and NN models.

### Parameters

- `data_dir` (str): Directory to load the data from.
- `out_dir` (str): Directory to save the results to.
- `OUTPUT_VAR_IDS` (list of int): The output variable(s) to be evaluated (0=hgtprs, 1=pratesfc, 2=tmp2m).
- `KERNEL_CONFIG_IDS` (list of int): List of GP kernel ID(s) to evaluate.
- `TRAIN_SPLIT_N` (int): The number of splits in the training set for GP models.
- `TRAIN_SPLIT_ID` (int): The split ID of the training set to process for GP models.
- `NUM_EPOCHS_GP` (int): The number of epochs for training GP models.
- `LEARNING_RATE_GP` (float): The learning rate for training GP models.
- `MEAN_ID` (int): The ID of the mean component for GP models.
- `PROCESSING_UNIT` (str): The device for training the GP models.
- `ARCHITECTURES` (list of str): List of NN model architecture(s) to evaluate.
- `NUM_EPOCHS_NN` (int): The number of epochs for training NN models.
- `LEARNING_RATE_NN` (float): The learning rate for training NN models.
- `SEED` (int): The seed used for random initialization of weights for NN models.

### Notes

- The function iterates over the provided `KERNEL_CONFIG_IDS` and `OUTPUT_VAR_IDS` to evaluate different combinations of kernel configurations and output variables.
- The function also iterates over the provided `ARCHITECTURES` to evaluate different neural network architectures.
- For each model, it loads the relevant data, calculates residuals, and calculates the root mean square error (RMSE) for the models.
- The results are printed and saved to a metrics file.
- Residual plots and time series plots are also generated to visualize the model performance.

## `parse_arguments`

This function parses the command line arguments provided when running the script via terminal.

### Returns

- A tuple of parsed arguments as follows:
  - `data_dir` (str): Directory to load the data from.
  - `out_dir` (str): Directory to save the results to.
  - `OUTPUT_VAR_IDS` (int): The output variable(s) to be evaluated (0=hgtprs, 1=pratesfc, 2=tmp2m).
  - `KERNEL_CONFIG_IDS` (list): List of GP kernel ID(s) to evaluate.
  - `TRAIN_SPLIT_N` (int): The number of splits in the training set for GP models.
  - `TRAIN_SPLIT_ID` (int): The split ID of the training set to process for GP models.
  - `NUM_EPOCHS_GP` (int): The number of epochs for training GP models.
  - `LEARNING_RATE_GP` (float): The learning rate for training GP models.
  - `MEAN_ID` (int): The ID of the mean component for GP models.
  - `PROCESSING_UNIT` (str): The device for training the GP models.
  - `ARCHITECTURES` (list): List of NN architecture(s) to evaluate.
  - `NUM_EPOCHS_NN` (int): The number of epochs for training NN models.
  - `LEARNING_RATE_NN` (float): The learning rate for training NN models.
  - `SEED` (int): The seed used for random initialization of weights for NN models.
