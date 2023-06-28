import sys
sys.path.append('./src/')
import torch
import argparse
from pprint import pprint
from torch import tensor
from NeuralNetwork.nn import NeuralNetworkModel, xArray_to_tensor
from Postprocessing.utils import *

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from torch import manual_seed
from torchsummary import summary

def main(out_dir, data_dir, ARCHITECTURE, NUM_EPOCHS=10, LEARNING_RATE=0.15, SEED=123, show_training_progress_plot=False, return_saved_data=False):
    """
    Train the NN model.

    Parameters
    ----------
    out_dir : str
        The output file name/path.
    data_dir : str
        The input file name/path.
    ARCHITECTURE : str
        The ___
    NUM_EPOCHS : int, optional
        The total number of epochs (integer greater than 1). Default is 10.
    LEARNING_RATE : float, optional
        The learning rate (float greater than 0). Default is 0.15.
    SEED : int, optional
        The seed used for random initialization of weights. Default is 123.
    plot_training_progress : bool, optional
        Whether or not to show a plot of the training progress. Default is False.
    return_saved_data : bool, optional
        Whether or not to return the saved data. Default is False.

    Returns
    -------
    None or dict: 
        If `return_saved_data` is True, returns a dictionary of saved data.
        Otherwise, returns None.

    Notes
    -----
    The output file prefix is generated using the following format:
        nn_{ARCHITECTURE}_e{NUM_EPOCHS}_l{LEARNING_RATE}_s{SEED}
        - {ARCHITECTURE}: The architecture for the NN model. There are 9 different pre-defined architectures.
        - {NUM_EPOCHS}: The total number of epochs for training.
        - {LEARNING_RATE}: The learning rate used for training.
        - {SEED}: The seed used for random initialization of weights.

    Additional Information:
        - The model state is saved in a file named 
            'nn_{ARCHITECTURE}_e{NUM_EPOCHS}_l{LEARNING_RATE}_s{SEED}_model_state.pt'.
        - The validation predictions are saved in a NetCDF file named 
            'nn_{ARCHITECTURE}_e{NUM_EPOCHS}_l{LEARNING_RATE}_s{SEED}_valid_preds.nc'.
    """    
    # WRANGLE DATA FOR THE NN MODEL:   
    train_ds = xr.open_dataset(data_dir + "preprocessed_train_ds.nc") 
    valid_ds = xr.open_dataset(data_dir + "preprocessed_valid_ds.nc")
    test_ds = xr.open_dataset(data_dir + "preprocessed_test_ds.nc")
    x_vars = [
        'scaled_deseas_d18O_pr', 'scaled_E', 'scaled_N', 
        'scaled_dist_to_coast', 'scaled_oro', 'landsfc'
    ]
    y_vars = [
        'scaled_deseas_tmp2m', 
        'scaled_deseas_hgtprs', 
        'scaled_deseas_pratesfc'
    ]
    X_train, Y_train = xArray_to_tensor(train_ds, x_vars, y_vars)
    X_valid, Y_valid = xArray_to_tensor(valid_ds, x_vars, y_vars)


    # CREATE THE NN MODEL:
    # Set seed of random weights for reproducibility
    # Note model performance can be heavily influenced by initial weights
    manual_seed(SEED)
    # There are 9 pre-defined architectures; see documentation.
    NN_model = NeuralNetworkModel(ARCHITECTURE, x_vars, y_vars)

    # PRINT NN MODEL SUMMARY:
    print('\nSummary of selected model architecture:')
    print(f'Architecture name: {ARCHITECTURE}')
    summary(NN_model, X_train[0].shape)

    # TRAIN THE NN MODEL:
    print('\nTraining:')
    print('-----------')
    results = NN_model.fit(X_train, Y_train, X_valid, Y_valid, e=1e-4, stall_limit=20, max_epochs=NUM_EPOCHS)

    # SAVE MODEL PARAMETERS:
    output_file_prefix = f"nn_{ARCHITECTURE}_e{NUM_EPOCHS}_l{LEARNING_RATE}_s{SEED}"
    model_state_file = output_file_prefix + "_model_state.pt"
    torch.save(NN_model.state_dict(), out_dir + model_state_file)

    # VISUALIZE TRAINING PROGRESS - model stops learning once valid_rmse line flattens out
    results_df = pd.DataFrame(results)
    results_df[['train_rmse', 'valid_rmse']].plot(
        figsize = (6, 4),
        title = "Learning effectiveness",
        xlabel = "Epoch number",
        ylabel = "RMSE"
    )
    training_progress_plot_file = output_file_prefix + "_training_progress_plot.png"
    plt.savefig(out_dir + training_progress_plot_file)
    if show_training_progress_plot:
        plt.show()

    # GET MODEL PREDICTIONS
    train_preds_ds = NN_model.predict_inplace(train_ds)
    valid_preds_ds = NN_model.predict_inplace(valid_ds)
    # test_preds_ds = model.predict_inplace(test_ds)

    # SAVE MODEL PREDICTIONS
    train_preds_file = output_file_prefix + "_train_preds.nc"
    valid_preds_file = output_file_prefix + "_valid_preds.nc"
    # test_preds_file = output_file_prefix + "_test_preds.nc"
    train_preds_ds.to_netcdf(path=out_dir + train_preds_file)
    valid_preds_ds.to_netcdf(path=out_dir + valid_preds_file)
    # test_preds_ds.to_netcdf(path=out_dir + test_preds_file)

    # CALCULATE AND PRINT VALIDATION RMSE
    valid_resids_ds = calculate_residuals(valid_preds_ds) # Calculate model residuals on the validation set
    rmse_results = calculate_model_RMSE(valid_resids_ds, scale='scaled_deseas') # Calculate model RMSE (on the scaled anomalies)
    rmse_results = {k: round(v, 4) for k,v in rmse_results.items()} # This rounds the values of the results dictionary
    print()
    pprint(rmse_results)

    print()
    print('nn_main.py for Neural Networks completed')

    if return_saved_data:
        return (train_preds_ds, valid_preds_ds, NN_model.model.state_dict())
    return

def parse_arguments():
    """
    Get arguments from the command line.

    Returns
    -------
        tuple: A tuple of parsed arguments as follows:
            - out_dir (str): The output file name/path.
            - data_dir (str): The input file name/path.
            - ARCHITECTURE (str): The architecture for the NN model. There are 9 pre-defined architectures; see documentation.
            - NUM_EPOCHS (int): The total number of epochs (integer greater than 1).
            - LEARNING_RATE (float): The learning rate (float greater than 0).
            - SEED (int): The seed used for random initialization of weights. Default is 123.

    Notes
    -----
    The available options for ARCHITECTURE are:
        - Foo 
        - Bar
        - FooBar
    """
    
    # Declare a "parser" variable
    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument('-o', dest='out_dir', required=True, help='Specify the output file name/path')
    parser.add_argument('-d', dest='data_dir', required=True, help='Specify the input file name/path')
    parser.add_argument('-a', dest='ARCHITECTURE', default='CNN-deep2',  help='Specify the architecture for the NN model.')
    parser.add_argument('-e', dest='NUM_EPOCHS', default=10, type=int, help='Specify the total number of epochs (integer greater than 1)')
    parser.add_argument('-l', dest='LEARNING_RATE', default=0.15, type=float, help='Specify the learning rate (float greater than 0)')
    parser.add_argument('-s', dest='SEED', default=123, type=int, help='The seed used for random initialization of weights.')
    # Store all the arguments in "args"
    args = parser.parse_args()
    
    # Save arguments as variables
    out_dir = args.out_dir
    data_dir = args.data_dir
    ARCHITECTURE = args.ARCHITECTURE
    NUM_EPOCHS = int(args.NUM_EPOCHS)
    LEARNING_RATE = float(args.LEARNING_RATE)
    SEED = int(args.SEED)

    VALID_ARCHITECTURES = [
        "CNN-simple", "CNN-wide", "CNN-deep", "CNN-deep2", 
        "Linear-narrow", "Linear-wide", "Linear-deep", 
        "Hybrid", "Hybrid-deep", "Hybrid-narrow"
    ]
    assert(ARCHITECTURE in VALID_ARCHITECTURES), f"The architecture component must be one of {VALID_ARCHITECTURES}."
  
    return out_dir, data_dir, ARCHITECTURE, NUM_EPOCHS, LEARNING_RATE, SEED

if __name__ == "__main__":
    # PARSE COMMAND LINE ARGUMENTS:
    main(*parse_arguments())

