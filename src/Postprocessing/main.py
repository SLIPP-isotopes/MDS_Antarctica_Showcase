import xarray as xr
import numpy as np
from pprint import pprint
import argparse
import sys
sys.path.append('../src/') # This lets us access the code in ../src/
sys.path.append('./src/') # This lets us access the code in ./src/
from Postprocessing.utils import *
from Preprocessing.utils import get_elements

OUTPUT_VAR_NAMES = ['hgtprs', 'pratesfc', 'tmp2m']

def main(data_dir,
        out_dir,
        OUTPUT_VAR_IDS,
        KERNEL_CONFIG_IDS,
        TRAIN_SPLIT_N,
        TRAIN_SPLIT_ID,
        NUM_EPOCHS_GP,
        LEARNING_RATE_GP,
        MEAN_ID,
        PROCESSING_UNIT,
        ARCHITECTURES,
        NUM_EPOCHS_NN,
        LEARNING_RATE_NN,
        SEED):
    """
    Main function for evaluating models and calculating metrics.
    This is thee central piece of code responsible for evaluating models and calculating metrics. 

    First, it performs calculations and generates plots for Gaussian Process (GP) models. 
    It iterates over the provided KERNEL_CONFIG_IDS and OUTPUT_VAR_IDS to 
    evaluate different combinations of kernel configurations and output variables. 
    For each combination, it loads the relevant data, calculates residuals, and 
    calculates model root mean square error (RMSE). 
    The results are printed and saved to a metrics file. 
    Additionally, residual plots and time series plots are generated to 
    visualize the model performance.

    The function also includes a placeholder for evaluating Neural Network (NN) models. 
    This is not yet implemented, but it can be extended to iterate over 
    the provided ARCHITECTURES and perform similar calculations and 
    plot generation as done for the GP models.

    Parameters
    ----------
    data_dir : str
        Directory to load the data from.
    out_dir : str
        Directory to save the results to.
    OUTPUT_VAR_IDS : list of int
        The output variable(s) to be evaluated (0=hgtprs, 1=pratesfc, 2=tmp2m).
    KERNEL_CONFIG_IDS : list of int
        List of GP kernel ID(s) to evaluate.
    TRAIN_SPLIT_N : int
        The number of splits in the training set for GP models.
    TRAIN_SPLIT_ID : int
        The split ID of the training set to process for GP models.
    NUM_EPOCHS_GP : int
        The number of epochs for training GP models.
    LEARNING_RATE_GP : float
        The learning rate for training GP models.
    MEAN_ID : int
        The ID of the mean component for GP models.
    PROCESSING_UNIT : str
        The device for training the GP models.
    ARCHITECTURES : list of str
        List of NN model architecture(s) to evaluate.
    NUM_EPOCHS_NN : int
        The number of epochs for training NN models.
    LEARNING_RATE_NN : float
        The learning rate for training NN models.
    SEED: int
        The seed used for random initialization of weights for NN models.

    Notes
    -----
    The available options for OUTPUT_VAR_IDS are:
        - 0: hgtprs (Geopotential height)
        - 1: pratesfc (surface Precipitation rate)
        - 2: tmp2m (Temp. 2m above surface)
    - ARCHITECTURES should be a list of strings specifying the NN architecture names.
    """
    # Calculate metrics and generate plots for GP models:
    for KERNEL_CONFIG_ID in KERNEL_CONFIG_IDS:
        for output_var in get_elements(OUTPUT_VAR_NAMES, OUTPUT_VAR_IDS):
            output_file_prefix = f"gp_{output_var}_k{KERNEL_CONFIG_ID}_m{MEAN_ID}_n{TRAIN_SPLIT_N}_s{TRAIN_SPLIT_ID}_e{NUM_EPOCHS_GP}_l{LEARNING_RATE_GP}_{PROCESSING_UNIT}"
            ds = xr.open_dataset(f"{data_dir}{output_file_prefix}_valid_preds.nc")

            ds = calculate_residuals(ds)

            if MEAN_ID == 0:
                mean_name = 'constant'
            elif MEAN_ID == 1:
                mean_name = 'linear'
            
            results = calculate_model_RMSE(ds, scale='scaled_deseas') # Calculate model performance
            results = {k: round(v, 4) for k,v in results.items()}     # This rounds the values of the results dictionary
            print(f"\nGP (kernel={KERNEL_CONFIG_ID}, mean={mean_name}):")
            pprint(results)

            with open(f'{out_dir}{output_file_prefix}_metrics.txt', 'w') as f:
                f.write("RMSE:\n")
                for key, val in results.items():
                    f.write(f"\t{key}: {val}\n")

            # Visualize residuals -- the less interesting, the better
            # More color or patterns suggests our model failed to capture something
            generate_residual_plots(
                ds,
                variable=f'resid_scaled_deseas_{output_var}',
                label=f'{output_var} anomalies (scaled)', 
                model_name=f'GP Model (kernel {KERNEL_CONFIG_ID})',
                out_file=f"{out_dir}{output_file_prefix}_resid_plots.png",
                show_plot=False
            )
            generate_time_series_plots(
                dataset=ds, 
                output_var=output_var,
                output_scale="scaled_deseas",
                include_random_deltas=True,
                label=f'{output_var} predictions over time', 
                model_name=f'GP Model (kernel {KERNEL_CONFIG_ID})',
                out_file=f"{out_dir}{output_file_prefix}_time_series_plots_scaled_deseas.png",
                show_plot=False
            )
            generate_time_series_plots(
                dataset=ds, 
                output_var=output_var, 
                output_scale="original",
                include_random_deltas=True,
                label=f'{output_var} predictions over time', 
                model_name=f'GP Model (kernel {KERNEL_CONFIG_ID})',
                out_file=f"{out_dir}{output_file_prefix}_time_series_plots_original.png",
                show_plot=False
            )

    # Calculate metrics and generate plots for NN models:
    for ARCHITECTURE in ARCHITECTURES:
        for output_var in get_elements(OUTPUT_VAR_NAMES, OUTPUT_VAR_IDS):
            preds_file_prefix = f"nn_{ARCHITECTURE}_e{NUM_EPOCHS_NN}_l{LEARNING_RATE_NN}_s{SEED}"
            output_file_prefix = f"nn_{output_var}_{ARCHITECTURE}_e{NUM_EPOCHS_NN}_l{LEARNING_RATE_NN}_s{SEED}"
            ds = xr.open_dataset(f"{data_dir}{preds_file_prefix}_valid_preds.nc")

            ds = calculate_residuals(ds)

            results = calculate_model_RMSE(ds, scale='scaled_deseas') # Calculate model performance
            results = {k: round(v, 4) for k,v in results.items()}     # This rounds the values of the results dictionary
            print(f"\nNN (architecture={ARCHITECTURE}):")
            pprint(results)

            with open(f'{out_dir}{output_file_prefix}_metrics.txt', 'w') as f:
                            f.write("RMSE:\n")
                            for key, val in results.items():
                                f.write(f"\t{key}: {val}\n")

            # Visualize residuals -- the less interesting, the better
            # More color or patterns suggests our model failed to capture something
            generate_residual_plots(
                ds, 
                variable = f'resid_scaled_deseas_{output_var}', 
                label = f'{output_var} anomalies (scaled)', 
                model_name = f'NN Model ({ARCHITECTURE})',
                out_file=f"{out_dir}{output_file_prefix}_resid_plots.png",
                show_plot=False
            )
            generate_time_series_plots(
                dataset=ds, 
                output_var=output_var,
                output_scale="scaled_deseas",
                label=f'{output_var} predictions over time', 
                model_name = f'NN Model ({ARCHITECTURE})',
                out_file=f"{out_dir}{output_file_prefix}_time_series_plots_scaled_deseas.png",
                show_plot=False
            )
            generate_time_series_plots(
                dataset=ds, 
                output_var=output_var, 
                output_scale="original",
                label=f'{output_var} predictions over time', 
                model_name = f'NN Model ({ARCHITECTURE})',
                out_file=f"{out_dir}{output_file_prefix}_time_series_plots_original.png",
                show_plot=False
            )
            
def parse_arguments():
    """
    Get arguments from the command line.

    Returns
    -------
    tuple
        A tuple of parsed arguments as follows:
        - data_dir (str): Directory to load the data from.
        - out_dir (str): Directory to save the results to.
        - OUTPUT_VAR_IDS (int): The output variable(s) to be evaluated (0=hgtprs, 1=pratesfc, 2=tmp2m).
        - KERNEL_CONFIG_IDS (list): List of GP kernel ID(s) to evaluated.
        - TRAIN_SPLIT_N (int): The number of splits in the training set for GP models.
        - TRAIN_SPLIT_ID (int): The split ID of the training set to process for GP models.
        - NUM_EPOCHS_GP (int): The number of epochs for training GP models.
        - LEARNING_RATE_GP (float): The learning rate for training GP models.
        - MEAN_ID (int): The ID of the mean component for GP models.
        - PROCESSING_UNIT (str): The device for training the GP model.
        - ARCHITECTURES (list): List of NN architecture(s) to evaluated.
        - NUM_EPOCHS_NN (int): The number of epochs for training NN models.
        - LEARNING_RATE_NN (float): The learning rate for training NN models.
        - SEED (int): The seed used for random initialization of weights for NN models.
    Notes
    -----
    The available options for OUTPUT_VAR_ID are:
        - 0: hgtprs (Geopotential height)
        - 1: pratesfc (surface Precipitation rate)
        - 2: tmp2m (Temp. 2m above surface)
    """
    # Declare a "parser" variable
    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument('-d', dest='data_dir', required=True, 
                        help='Directory to load the prediction data from.')
    parser.add_argument('-o', dest='out_dir', required=True, 
                        help='Directory to save the residual diagnostic results to.')
    parser.add_argument('-v', dest='OUTPUT_VAR_IDS', type=int, nargs='+', default=[0, 1, 2],
                        help='Specify the output variable(s) to be evaluated (0=hgtprs, 1=pratesfc, 2=tmp2m). Default is [0, 1, 2].')
    parser.add_argument('-k', dest='KERNEL_CONFIG_IDS', type=int, nargs='+', default=[0, 1],
                        help='List of GP kernel IDs to process. Default is [0, 1].')
    parser.add_argument('-n', dest='TRAIN_SPLIT_N', type=int, default=6,
                        help='The number of splits in the training set for GP models. Default is 6.')
    parser.add_argument('-sgp', dest='TRAIN_SPLIT_ID', type=int, default=3,
                        help='The split ID of the training set to process for GP models. Default is 3.')
    parser.add_argument('-egp', dest='NUM_EPOCHS_GP', type=int, default=10,
                        help='The number of epochs for training GP models. Default is 10.')
    parser.add_argument('-lgp', dest='LEARNING_RATE_GP', type=float, default=0.001,
                        help='The learning rate for training GP models. Default is 0.001.')
    parser.add_argument('-m', dest='MEAN_ID', type=int, default=1,
                        help='The ID of the mean component for GP models. Default is 1.')
    parser.add_argument('-pu', dest='PROCESSING_UNIT', type=str, default='cpu',
                        help='The device for training the GP model. Default is "cpu".')
    parser.add_argument('-a', dest='ARCHITECTURES', type=str, nargs='+', default=['foo', 'bar'],
                        help="List of NN architecture(s) to evaluate. Default is ['foo', 'bar'].")
    parser.add_argument('-enn', dest='NUM_EPOCHS_NN', type=int, default=10,
                        help='The number of epochs for training NN models. Default is 10.')
    parser.add_argument('-lnn', dest='LEARNING_RATE_NN', type=float, default=0.001,
                        help='The learning rate for training GP models. Default is 0.001.')
    parser.add_argument('-snn', dest='SEED', type=int, default=123,
                        help='The seed used for random initialization of weights for NN models. Default is 123.')

    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    OUTPUT_VAR_IDS = args.OUTPUT_VAR_IDS
    KERNEL_CONFIG_IDS = args.KERNEL_CONFIG_IDS
    TRAIN_SPLIT_N = args.TRAIN_SPLIT_N
    TRAIN_SPLIT_ID = args.TRAIN_SPLIT_ID
    NUM_EPOCHS_GP = args.NUM_EPOCHS_GP
    LEARNING_RATE_GP = args.LEARNING_RATE_GP
    MEAN_ID = args.MEAN_ID
    PROCESSING_UNIT = args.PROCESSING_UNIT
    ARCHITECTURES = args.ARCHITECTURES
    NUM_EPOCHS_NN = args.NUM_EPOCHS_NN
    LEARNING_RATE_NN = args.LEARNING_RATE_NN
    SEED = args.SEED

    return (
        data_dir,
        out_dir,
        OUTPUT_VAR_IDS,
        KERNEL_CONFIG_IDS,
        TRAIN_SPLIT_N,
        TRAIN_SPLIT_ID,
        NUM_EPOCHS_GP,
        LEARNING_RATE_GP,
        MEAN_ID,
        PROCESSING_UNIT,
        ARCHITECTURES,
        NUM_EPOCHS_NN,
        LEARNING_RATE_NN,
        SEED
    )

if __name__ == "__main__":
    # PARSE COMMAND LINE ARGUMENTS:
    main(*parse_arguments())
