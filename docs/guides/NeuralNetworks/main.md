# NeuralNetwork.main

Authors: Jakob Thoms, Daniel Cairns

Date: June 2023

[Source code](/src/NeuralNetwork/scripts/nn_main.py)

This script is designed to train Neural Network (NN) models.

## Command line flags

When run via the terminal, the script supports the following command line flags:

- `-o, --out_dir`: Specifies the output file name/path.
- `-d, --data_dir`: Specifies the input file name/path.
- `-a, --ARCHITECTURE`: Specifies the architecture for the NN model.
- `-e, --NUM_EPOCHS`: Specifies the total number of epochs (integer greater than 1).
- `-l, --LEARNING_RATE`: Specifies the learning rate (float greater than 0).
- `-s, --SEED`: Specifies the seed used for random initialization of weights.

## `main` function

This is the main driver function for training NN models.

### Parameters

- `out_dir` (str): The output file name/path.
- `data_dir` (str): The input file name/path.
- `ARCHITECTURE` (str): The architecture for the NN model.
- `NUM_EPOCHS` (int, optional): The total number of epochs (integer greater than 1). Default is 10.
- `LEARNING_RATE` (float, optional): The learning rate (float greater than 0). Default is 0.15.
- `SEED` (int, optional): The seed used for random initialization of weights. Default is 123.
- `show_training_progress_plot` (bool, optional): Whether or not to show a plot of the training progress. Default is False.
- `return_saved_data` (bool, optional): Whether or not to return the saved data. Default is False.

### Returns

- None or dict: If `return_saved_data` is True, returns a dictionary of saved data. Otherwise, returns None.

### Notes

- The output file prefix is generated using the following format:

    ```
    nn_{ARCHITECTURE}_e{NUM_EPOCHS}_l{LEARNING_RATE}_s{SEED}
    ```

- Additional Information:
  - The training progress plot is saved in a file named `nn_{ARCHITECTURE}_e{NUM_EPOCHS}_l{LEARNING_RATE}_s{SEED}_training_progress_plot.png`.
  - The model state is saved in a file named `nn_{ARCHITECTURE}_e{NUM_EPOCHS}_l{LEARNING_RATE}_s{SEED}_model_state.pt`.
  - The validation predictions are saved in a NetCDF file named `nn_{ARCHITECTURE}_e{NUM_EPOCHS}_l{LEARNING_RATE}_s{SEED}_valid_preds.nc`.


## `parse_arguments`

This function parses the command line arguments provided when running the script via terminal.

### Returns

- A tuple of parsed arguments as follows:
  - `out_dir` (str): The output file name/path.
  - `data_dir` (str): The input file name/path.
  - `ARCHITECTURE` (str): The architecture for the NN model. There are 9 pre-defined architectures; see documentation.
  - `NUM_EPOCHS` (int): The total number of epochs (integer greater than 1).
  - `LEARNING_RATE` (float): The learning rate (float greater than 0).
  - `SEED` (int): The seed used for random initialization of weights. Default is 123.
