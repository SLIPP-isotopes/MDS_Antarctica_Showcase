import sys
sys.path.append('./src/')
import torch
import gpytorch as gp
import warnings
import argparse
from pprint import pprint
from torch import tensor
from numpy.random import (uniform, normal, seed)
from GaussianProcesses.models import GP, ExactGPModel
from GaussianProcesses.kernels import get_kernel
from GaussianProcesses.utils import wrangle, flat_validation_predictions_to_xarray, reduce_training_size, RMSE
from Postprocessing.utils import *

def main(out_dir, data_dir, OUTPUT_VAR_ID, TRAIN_SPLIT_N, TRAIN_SPLIT_ID, \
        KERNEL_CONFIG_ID, NUM_EPOCHS=10, LEARNING_RATE=0.15, NUM_RESTARTS=4, \
            NUM_EPOCHS_PER_RESTART=5, MEAN_COMPONENT='linear', PROCESSING_UNIT='cpu', return_saved_data=False):
    """
    Train the GP model.

    Parameters
    ----------
    out_dir : str
        The output file name/path.
    data_dir : str
        The input file name/path.
    OUTPUT_VAR_ID : int
        The output variable to be predicted (0=hgtprs, 1=pratesfc, 2=tmp2m).
    TRAIN_SPLIT_N : int
        The number of splits in the training set.
    TRAIN_SPLIT_ID : int
        The split of the training set to be used (integer between 0 and TRAIN_SPLIT_N - 1).
    KERNEL_CONFIG_ID : int
        The kernel configuration (integer between 0 and 7) (but not 6) (see kernels.py).
    NUM_EPOCHS : int, optional
        The total number of epochs (integer greater than 1). Default is 10.
    LEARNING_RATE : float, optional
        The learning rate (float greater than 0). Default is 0.15.
    NUM_RESTARTS : int, optional
        The number of restarts for warmup (integer greater than 0). Default is 4.
    NUM_EPOCHS_PER_RESTART : int, optional
        The number of epochs per restart during warmup (integer greater than 0). Default is 5.
    MEAN_COMPONENT : str, optional
        The mean component for the GP model. Default is 'linear'.
    PROCESSING_UNIT : str, optional
        The device (CPU or GPU) for training the GP model. Default is 'cpu'.
    return_saved_data : bool, optional
        Whether or not to return the saved data. Default is False.

    Returns
    -------
    None or dict: 
        If `return_saved_data` is True, returns a dictionary of saved data.
        Otherwise, returns None.

    Notes
    -----
    The available options for OUTPUT_VAR_ID are:
    - 0: hgtprs (Geopotential height)
    - 1: pratesfc (surface Precipitation rate)
    - 2: tmp2m (Temp. 2m above surface)

    The available options for `KERNEL_CONFIG_ID` are:
    - 0: RBF + PeriodicKernel for month
    - 1: RBF (deseasonalized; no month)
    - 2: PiecewisePolynomialKernel + PeriodicKernel for month
    - 3: PiecewisePolynomialKernel (deseasonalized; no month)
    - 4: RQKernel + PeriodicKernel for month
    - 5: RQKernel (deseasonalized; no month)
    - 7: SpectralMixture (deseasonalized; no month)

    The output file prefix is generated using the following format:
        gp_{output_feat_name}_k{KERNEL_CONFIG_ID}_m{MEAN_ID}_n{TRAIN_SPLIT_N}_s{TRAIN_SPLIT_ID}_e{NUM_EPOCHS}_l{LEARNING_RATE}_{PROCESSING_UNIT}
        - {output_feat_name}: The name of the output feature from the wrangled data.
        - {KERNEL_CONFIG_ID}: The ID of the chosen kernel configuration.
        - {MEAN_ID}: The ID of the chosen mean component.
        - {TRAIN_SPLIT_N}: The number of splits (batches) in the training dataset.
        - {TRAIN_SPLIT_ID}: The ID of the current split (batch) used for training.
        - {NUM_EPOCHS}: The total number of epochs for training.
        - {LEARNING_RATE}: The learning rate used for training.
        - {PROCESSING_UNIT}: The processing unit used, either 'cpu' or 'gpu'.

    Additional Information:
        - The model state is saved in a file named 
            'gp_{output_feat_name}_k{KERNEL_CONFIG_ID}_m{MEAN_ID}_n{TRAIN_SPLIT_N}_s{TRAIN_SPLIT_ID}_e{NUM_EPOCHS}_l{LEARNING_RATE}_{PROCESSING_UNIT}_model_state.pth'.
        - The validation predictions are saved in a NetCDF file named 
            'gp_{output_feat_name}_k{KERNEL_CONFIG_ID}_m{MEAN_ID}_n{TRAIN_SPLIT_N}_s{TRAIN_SPLIT_ID}_e{NUM_EPOCHS}_l{LEARNING_RATE}_{PROCESSING_UNIT}_valid_preds.nc'.
    """    
    # WRANGLE DATA FOR THE GP MODEL:    
    train_X_no_nans, train_Y_no_nans_flat, \
        valid_X_no_nans, valid_Y_no_nans_flat, \
        valid_Y, valid_ds, valid_nan_mask, output_feat_name = wrangle(
                                                                data_dir=data_dir, 
                                                                output_var=OUTPUT_VAR_ID, 
                                                                kernel_config_id=KERNEL_CONFIG_ID
                                                            )

    # REDUCE TRAINING DATA SIZE:
    # This is where we decide which split (i.e. batch) of the training dataset to use. 
    # This is done due to memory limitations when training GP models.
    train_X_no_nans_split, train_Y_no_nans_flat_split = reduce_training_size(
                                                            train_Y_no_nans_flat, 
                                                            train_X_no_nans, 
                                                            train_split_n=TRAIN_SPLIT_N, # number of splits
                                                            train_split_id=TRAIN_SPLIT_ID, # integer b/w 0 and TRAIN_SPLIT_N - 1
                                                            quick_test_run=True # FOR QUICK LOCAL TESTING ONLY
                                                        )
    
    # CHOOSE KERNEL CONFIGURATION:
    kernel = get_kernel(config_id=KERNEL_CONFIG_ID) 

    # GPU STUFF:
    if PROCESSING_UNIT == 'gpu':
        train_X_no_nans_split = train_X_no_nans_split.cuda()
        train_Y_no_nans_flat_split = train_Y_no_nans_flat_split.cuda()

    # =========
    # The code below this point should be the same for all GP models.
    # (except maybe we change the learning rate and number of epochs)
    # ==========

    # CREATE THE GP MODEL:
    GP_model = GP(
        model=ExactGPModel, 
        likelihood=gp.likelihoods.GaussianLikelihood,
        train_X=train_X_no_nans_split, # Training inputs 
        train_Y=train_Y_no_nans_flat_split, # Training outputs 
        mean=MEAN_COMPONENT, # Use a linear or a constant mean (the regression component) [specified by user input]
        kernel=kernel, # Use the kernel we created above (the covariance component)
    )

    # GPU STUFF:
    if PROCESSING_UNIT == 'gpu':
        GP_model.model = GP_model.model.cuda()
        GP_model.likelihood = GP_model.likelihood.cuda()


    # TRAIN THE GP MODEL
    print('\nWarming up:')
    print('-----------')
    GP_model.warmup(num_restarts=NUM_RESTARTS, num_epochs_per_restart=NUM_EPOCHS_PER_RESTART, learning_rate=LEARNING_RATE)
    print('\nTraining:')
    print('-----------')
    GP_model.train(num_epochs=NUM_EPOCHS - NUM_EPOCHS_PER_RESTART, learning_rate=LEARNING_RATE)
    print('Training complete.')

    # SAVE MODEL PARAMETERS
    if MEAN_COMPONENT == 'linear':
        MEAN_ID = 1
    elif MEAN_COMPONENT == 'constant':
        MEAN_ID = 0
    output_feat_name = output_feat_name.split("_")[-1]
    output_file_prefix = f"gp_{output_feat_name}_k{KERNEL_CONFIG_ID}_m{MEAN_ID}_n{TRAIN_SPLIT_N}_s{TRAIN_SPLIT_ID}_e{NUM_EPOCHS}_l{LEARNING_RATE}_{PROCESSING_UNIT}"
    model_state_file =  output_file_prefix + "_model_state.pth"
    torch.save(GP_model.model.state_dict(), out_dir + model_state_file)

    # GET MODEL PREDICTIONS
    ## TRAINING PREDICTIONS
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train_pred = GP_model.predict(train_X_no_nans_split)
    ## VALIDATION PREDICTIONS
    NUM_VALID_BATCHES = 100
    batch_size = valid_X_no_nans.size()[0] // NUM_VALID_BATCHES
    print(f"\nGenerating validation predictions in batches.\nUsing {NUM_VALID_BATCHES} batches:")
    print("-----------")

    seed(0)
    valid_batch_preds = []
    rand_uniform_batch_preds = []
    rand_normal_batch_preds = []
    for i in range(NUM_VALID_BATCHES):
        # Print progress
        NUM_PRINTS = 5
        if i % (NUM_VALID_BATCHES // NUM_PRINTS) == 0:
            print(f"{round(i / NUM_VALID_BATCHES * 100)}% complete.")

        start_index, end_index = i * batch_size, (i + 1) * batch_size

        if i == NUM_VALID_BATCHES - 1: # valid_X_no_nans.size()[0] // NUM_VALID_BATCHES rounds down. 
            valid_batch = valid_X_no_nans[start_index:] # Make sure we get predictions for every example!
        else:
            valid_batch = valid_X_no_nans[start_index:end_index]

        rand_uniform_batch = valid_batch.clone().detach()
        rand_uniform_batch[:, 0] = tensor(uniform(-1.5, 1.5, size=rand_uniform_batch[:, 0].size()))

        rand_normal_batch = valid_batch.clone().detach()
        rand_normal_batch[:, 0] = tensor(normal(0, 1, size=rand_normal_batch[:, 0].size()))

        # GPU STUFF:
        if PROCESSING_UNIT == 'gpu':
            valid_batch_pred = GP_model.predict(valid_batch.cuda())
            rand_uniform_batch_pred = GP_model.predict(rand_uniform_batch.cuda())
            rand_normal_batch_pred = GP_model.predict(rand_normal_batch.cuda())
        else:
            valid_batch_pred = GP_model.predict(valid_batch)
            rand_uniform_batch_pred = GP_model.predict(rand_uniform_batch)
            rand_normal_batch_pred = GP_model.predict(rand_normal_batch)

        valid_batch_preds.append(valid_batch_pred)
        rand_uniform_batch_preds.append(rand_uniform_batch_pred)
        rand_normal_batch_preds.append(rand_normal_batch_pred)
    valid_pred = torch.cat(valid_batch_preds, dim=0)
    rand_uniform_pred = torch.cat(rand_uniform_batch_preds, dim=0)
    rand_normal_pred = torch.cat(rand_normal_batch_preds, dim=0)
    print("100% complete.")

    # GPU STUFF:
    if PROCESSING_UNIT == 'gpu':
        valid_batch_pred = GP_model.predict(valid_batch.cpu())
        rand_uniform_batch_pred = GP_model.predict(rand_uniform_batch.cpu())
        rand_normal_batch_pred = GP_model.predict(rand_normal_batch.cpu())

    # CONVERT MODEL PREDICTIONS BACK TO XARRAY:
    valid_preds_ds = flat_validation_predictions_to_xarray(valid_pred, rand_uniform_pred, rand_normal_pred, valid_Y, valid_ds, valid_nan_mask, output_feat_name, kernel_config_id=KERNEL_CONFIG_ID, train_split_id=TRAIN_SPLIT_ID)
    
    # print(f"\nTrain RMSE: {round(RMSE(train_Y_no_nans_flat_split, train_pred).item(), 3)}") # RMSE of the model on the training data
    # print(f"Valid RMSE: {round(RMSE(valid_Y_no_nans_flat, valid_pred).item(), 3)}") # RMSE of the model on the validation data
    # print(f"Valid RMSE: {round(RMSE(valid_Y_no_nans_flat, rand_uniform_pred).item(), 3)} [randomized deltas; Uniform(-1.5, 1.5)]") # RMSE of the model on the randomized validation data (w/ uniform random deltas)
    # print(f"Valid RMSE: {round(RMSE(valid_Y_no_nans_flat, rand_normal_pred).item(), 3)} [randomized deltas; Normal(0,1)]") # RMSE of the model on the randomized validation data (w/ normal random deltas)

    # SAVE VALIDATION PREDICTIONS
    preds_file = output_file_prefix + "_valid_preds.nc"
    valid_preds_ds.to_netcdf(path=out_dir + preds_file)

    # CALCULATE AND PRINT VALIDATION RMSE
    valid_resids_ds = calculate_residuals(valid_preds_ds) # Calculate model residuals on the validation set
    results = calculate_model_RMSE(valid_resids_ds, scale='scaled_deseas') # Calculate model RMSE (on the scaled anomalies)
    results = {k: round(v, 4) for k,v in results.items()} # This rounds the values of the results dictionary
    print("\nModel RMSEs:")
    print("-----------")
    pprint(results)

    # PRINT TRAINED MODEL PARAMETERS
    print("\nTrained model parameters:")
    print("-----------")
    print("\tKernel (covariance component):")
    pprint(dict(GP_model.model.covar_module.named_parameters()))
    print("\tMean (regression component):")
    pprint(dict(GP_model.model.mean_module.named_parameters()))

    print()
    print('gp_main.py for Gaussian Processes completed')

    if return_saved_data:
        return (valid_preds_ds, GP_model.model.state_dict())
    return

def parse_arguments():
    """
    Get arguments from the command line.

    Returns
    -------
        tuple: A tuple of parsed arguments as follows:
            - out_dir (str): The output file name/path.
            - data_dir (str): The input file name/path.
            - OUTPUT_VAR_ID (int): The output variable to be predicted (0=hgtprs, 1=pratesfc, 2=tmp2m).
            - TRAIN_SPLIT_N (int): The number of splits in the training set.
            - TRAIN_SPLIT_ID (int): The split of the training set to be used (integer between 0 and TRAIN_SPLIT_N - 1).
            - KERNEL_CONFIG_ID (int): The kernel configuration (integer between 0 and 7) (but not 6) (see kernels.py).
            - NUM_EPOCHS (int): The total number of epochs (integer greater than 1).
            - LEARNING_RATE (float): The learning rate (float greater than 0).
            - NUM_RESTARTS (int): The number of restarts for warmup (integer greater than 0).
            - NUM_EPOCHS_PER_RESTART (int): The number of epochs per restart during warmup (integer greater than 0).
            - MEAN_COMPONENT (str): The mean component for the GP model.
            - PROCESSING_UNIT (str): The device (CPU or GPU) for training the GP model.

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
    parser.add_argument('-o', dest='out_dir', required=True, help='Specify the output file name/path')
    parser.add_argument('-d', dest='data_dir', required=True, help='Specify the input file name/path')
    parser.add_argument('-v', dest='OUTPUT_VAR_ID', required=True, help='Specify the output variable to be predicted (0=hgtprs, 1=pratesfc, 2=tmp2m)')
    parser.add_argument('-k', dest='KERNEL_CONFIG_ID', required=True, type=int, help='Specify the kernel configuration (integer between 0 and 7)')
    parser.add_argument('-n', dest='TRAIN_SPLIT_N', default=13, type=int, help='Specify the number of splits in the training set')
    parser.add_argument('-s', dest='TRAIN_SPLIT_ID', required=True, type=int, help='Specify the split of the training set (integer between 0 and 12)')
    parser.add_argument('-e', dest='NUM_EPOCHS', default=10, type=int, help='Specify the total number of epochs (integer greater than 1)')
    parser.add_argument('-l', dest='LEARNING_RATE', default=0.15, type=float, help='Specify the learning rate (float greater than 0)')
    parser.add_argument('-r', dest='NUM_RESTARTS', default=4, type=int, help='Specify the number of restarts for warmup (integer greater than 0)')
    parser.add_argument('-er', dest='NUM_EPOCHS_PER_RESTART', default=5, type=int, help='Specify the number of epochs per restart during warmup (integer greater than 0)')
    parser.add_argument('-m', dest='MEAN_COMPONENT', default='linear',  help='Specify the mean component for the GP model.')
    parser.add_argument('-pu', dest='PROCESSING_UNIT', default='cpu',  help='Specify the device (CPU or GPU) for training the GP model.')
    
    # Store all the arguments in "args"
    args = parser.parse_args()
    
    # Save arguments as variables
    out_dir = args.out_dir
    data_dir = args.data_dir
    OUTPUT_VAR_ID = int(args.OUTPUT_VAR_ID)
    TRAIN_SPLIT_N = int(args.TRAIN_SPLIT_N)
    TRAIN_SPLIT_ID = int(args.TRAIN_SPLIT_ID)
    KERNEL_CONFIG_ID = int(args.KERNEL_CONFIG_ID)
    NUM_EPOCHS = int(args.NUM_EPOCHS)
    LEARNING_RATE = float(args.LEARNING_RATE)
    NUM_RESTARTS = int(args.NUM_RESTARTS)
    NUM_EPOCHS_PER_RESTART = int(args.NUM_EPOCHS_PER_RESTART)
    MEAN_COMPONENT = args.MEAN_COMPONENT
    PROCESSING_UNIT = args.PROCESSING_UNIT


    VALID_MEANS = ['constant', 'linear']
    VALID_PROCESSING_UNITS = ['gpu', 'cpu']
    assert(NUM_EPOCHS_PER_RESTART < NUM_EPOCHS), "The total number of epochs must be greater than the number of epochs per restart."
    assert(TRAIN_SPLIT_ID < TRAIN_SPLIT_N), "The train split id must be an integer b/w 0 (inclusive) and the total number of training splits (exclusive)."
    assert(MEAN_COMPONENT in VALID_MEANS), f"The mean component must be one of {VALID_MEANS}."
    assert(PROCESSING_UNIT in VALID_PROCESSING_UNITS), f"The processing unit must be one of {VALID_PROCESSING_UNITS}."

    # TO DO: TESTS AND EXCEPTION HANDLING
    # - Check if required arguments are passed (I think this happens automatically just by setting them to be required)
    # - Check if directories and files exist
    # - Check if inputs passed are the right type/make sense (like not 0.0)
   
    return out_dir, data_dir, OUTPUT_VAR_ID, TRAIN_SPLIT_N, TRAIN_SPLIT_ID, KERNEL_CONFIG_ID, NUM_EPOCHS, LEARNING_RATE, NUM_RESTARTS, NUM_EPOCHS_PER_RESTART, MEAN_COMPONENT, PROCESSING_UNIT

if __name__ == "__main__":
    # PARSE COMMAND LINE ARGUMENTS:
    main(*parse_arguments())

