import argparse
from pprint import pprint
import sys 
sys.path.insert(1, './src/') 
from Baselines.models import *
from Postprocessing.utils import *

def main(out_dir, data_dir, OUTPUT_VAR_ID, BASELINE_METHOD, VALTEST, VERBOSE=True):
    """
    Train and predict with a baseline model. 

    Parameters
    ----------
    out_dir : str
        The output file name/path.
    data_dir : str
        The input file name/path.
    OUTPUT_VAR_ID : int 
        The output variable to be predicted (0=hgtprs, 1=pratesfc, 2=tmp2m).
    BASELINE_METHOD : str  
        The type of baseline model to build and generate predictions from. 
    VERBOSE : bool  
        Whether or not to display a message to the console. Default is True.

    Returns
    -------
    None
    """
    
    # LOAD PREPROCESSED DATA
    train_ds = xr.open_dataset(data_dir + 'preprocessed_train_ds.nc')
    valid_ds = xr.open_dataset(data_dir + 'preprocessed_' + VALTEST + '_ds.nc') 

    # SET THE Y-VARIABLE
    y_variables = ['hgtprs', 'pratesfc', 'tmp2m']
    y_variable = 'scaled_deseas_' + y_variables[OUTPUT_VAR_ID]
    
    # OLS BASELINE
    if BASELINE_METHOD == 'ols':
        # Create an object 
        ols_baseline_model = OLS_baseline(x_var='scaled_deseas_d18O_pr', y_var=y_variable) 
        # Fit 
        ols_baseline_model.fit(train_ds=train_ds)
        # Predict on train and validation
        train_preds_ds = ols_baseline_model.predict(test_ds=train_ds, return_ds=True) 
        valid_preds_ds = ols_baseline_model.predict(test_ds=valid_ds, return_ds=True) 

    # MEAN BASELINES 
    elif BASELINE_METHOD != 'ols':
        # Create an object 
        mean_baseline_model = mean_baseline(y_var=y_variable) 
        # Fit 
        mean_baseline_model.fit(train_ds=train_ds)
        # Predict 
        train_preds_ds = mean_baseline_model.predict(test_ds=train_ds, method=BASELINE_METHOD, return_ds=True) 
        valid_preds_ds = mean_baseline_model.predict(test_ds=valid_ds, method=BASELINE_METHOD, return_ds=True) 

    # SAVE OUTPUT FILE
    file_name = f'{BASELINE_METHOD}_{y_variables[OUTPUT_VAR_ID]}_{VALTEST}_preds.nc'
    output_file = out_dir + file_name
    valid_preds_ds.to_netcdf(output_file)

    # RMSEs 
    train_resids_ds = calculate_residuals(train_preds_ds) # Calculate model residuals on the training set
    train_rmse_results = calculate_model_RMSE(train_resids_ds, scale='scaled_deseas') # Calculate model RMSE (on the scaled anomalies)

    valid_resids_ds = calculate_residuals(valid_preds_ds) 
    valid_rmse_results = calculate_model_RMSE(valid_resids_ds, scale='scaled_deseas')
    
    train_rmse_results = {k: round(v, 3) for k,v in train_rmse_results.items()} # This rounds the values of the results dictionary
    valid_rmse_results = {k: round(v, 3) for k,v in valid_rmse_results.items()}
        
    # VERBOSE MESSAGE 
    if VERBOSE:
        
        print(f'-------------------------------')
        print(f'"{BASELINE_METHOD}" baseline model')
        print(f'-------------------------------')
        print('TRAINING:')
        print(f'X-variable: scaled_deseas_d18O_pr\nY-variable: {y_variable}\n')
        
        if BASELINE_METHOD == 'ols':
            print(f'Slope: {ols_baseline_model.slope}')
            print(f'Intercept: {ols_baseline_model.intercept}')
            print(f'R^2 score: {ols_baseline_model.r_sq}\n')

        print('PREDICTING:')
        print(f'Output file saved: {output_file}')
        
        print(f'-------------------------------') 
        print('RMSEs')
        print(f'Train RMSE: {train_rmse_results}')
        print(f'Valid RMSE: {valid_rmse_results}')
        print(f'-------------------------------') 

    print('\nbl_main.py for Baselines completed')
        
    return 

def parse_arguments():
    """
    Get arguments from the command line.

    Returns
    -------
    tuple: A tuple of parsed arguments as follows:
        - out_dir (str): The output directory to save the predictions as netCDF files. 
        - data_dir (str): The input directory to load the preprocessed data from. 
        - OUTPUT_VAR_ID (int): The output variable to be predicted (0=hgtprs, 1=pratesfc, 2=tmp2m).
        - BASELINE_METHOD (str): The type of baseline model to build and generate predictions from. 
        - VERBOSE (bool): Whether or not to display a message to the console. 

    Notes
    -----
    - The default values are set according to the `main()` function's default arguments.
    """
    
    # Declare a "parser" variable
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-o', dest='out_dir', required=True, help='Specify the output path to save the predictions')
    parser.add_argument('-d', dest='data_dir', required=True, help='Specify the input path containing the preprocessed data')
    parser.add_argument('-v', dest='OUTPUT_VAR_ID', required=True, type=int, help='Specify the output variable to be predicted (0=hgtprs, 1=pratesfc, 2=tmp2m)')
    parser.add_argument('-b', dest='BASELINE_METHOD', required=True, help='Specify the type of baseline to use ("overall_mean", "latlon_mean", "latlonmonth_mean", "ols")')
    parser.add_argument('-vt', dest='VALTEST', default='valid', help='Specify whether to predict on the validation or test set ("valid", "test")')
    parser.add_argument('-verbose', dest='VERBOSE', default=1, type=int, help='Specify whether to display an informative message to the console (0=False, 1=True)') 

    # Store all the arguments in "args"
    args = parser.parse_args()

    # Save arguments as variables
    out_dir = args.out_dir
    data_dir = args.data_dir
    OUTPUT_VAR_ID = int(args.OUTPUT_VAR_ID)
    BASELINE_METHOD = args.BASELINE_METHOD
    VALTEST = args.VALTEST
    VERBOSE = bool(args.VERBOSE)

    return out_dir, data_dir, OUTPUT_VAR_ID, BASELINE_METHOD, VALTEST, VERBOSE

if __name__ == "__main__":
    # PARSE COMMAND LINE ARGUMENTS:
    main(*parse_arguments())