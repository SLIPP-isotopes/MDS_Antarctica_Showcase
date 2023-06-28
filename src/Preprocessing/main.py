import xarray as xr
import argparse
import sys
sys.path.append('./src/')
from Preprocessing.utils import *
from Preprocessing.preprocess import preprocess

def main(out_dir='data/', input_dir='data/', input_file='IsoGSM/Total.IsoGSM.ERA5.monmean.nc', return_data=False):
    """
    Handles Train/Valid/Test split and I/O for preprocessing.
    
    Parameters:
    ----------
    out_dir : str, optional
        The directory to save the preprocessed data files to. Default is 'data/'.
    input_dir : str, optional
        The directory to load the raw dataset from. Default is 'data/'.
    input_file : str, optional
        The filename of the raw dataset file. Default is 'IsoGSM/Total.IsoGSM.ERA5.monmean.nc'.
    return_data : bool, optional
        Whether or not to return the preprocessed datasets. Default is False.
    
    Returns:
    -------
    None or tuple
        If return_data is True, returns a tuple of preprocessed datasets (train_ds, valid_ds, test_ds).
        Otherwise, returns None.
    
    Notes:
    ------
    - Loads the raw dataset from the specified input file.
    - Performs train/valid/test split on the raw dataset.
    - Calls the `preprocess` function to preprocess each dataset:
        - Deseasonalizes the temporal variables ('d18O_pr', 'hgtprs', 'pratesfc', 'tmp2m')
        - Additional spatial variables are added through feature engineering (polar coords, month, land-sea mask, distance-to-coast, orography)
        - Interpolation of missing data is performed for the training set. 
        - All variables are scaled to have *global* mean=0 and stdev=1  
    - Saves the preprocessed datasets as NetCDF files in the specified output directory.
    - If return_data is True, the preprocessed datasets are returned as a tuple.
    """
    print("Preprocessing data. This may take a couple minutes.")
    # Load Data
    file = input_dir + input_file
    raw_ds = xr.open_dataset(file)

    # train/valid/test split - THESE MUST BE HARD-CODED
    TEST_YEAR_RANGE = slice("1979-01-01", "1986-12-31") # First 8 years of data
    VALID_YEAR_RANGE = slice("1987-01-01", "1994-12-31") # Next 8 years of data (after test data)
    TRAIN_YEAR_RANGE = slice("1995-01-01", "2020-12-31") # Remaining 26 years of (after test & valid data)
    raw_test = raw_ds.sel(time=TEST_YEAR_RANGE, latitude=LAT_RANGE)
    raw_train = raw_ds.sel(time=TRAIN_YEAR_RANGE, latitude=LAT_RANGE)
    raw_valid = raw_ds.sel(time=VALID_YEAR_RANGE, latitude=LAT_RANGE)
    
    print("Processing training data...")
    train_ds = preprocess(raw_train, is_training_data=True)
    print("\tComplete.")

    print("Processing validation data...")
    valid_ds = preprocess(raw_valid, is_training_data=False, train_ds=train_ds)
    print("\tComplete.")

    print("Processing validation data...")
    test_ds = preprocess(raw_test, is_training_data=False, train_ds=train_ds)
    print("\tComplete.")

    train_ds.to_netcdf(path=out_dir + 'preprocessed_train_ds.nc')
    valid_ds.to_netcdf(path=out_dir + 'preprocessed_valid_ds.nc')
    test_ds.to_netcdf(path=out_dir + 'preprocessed_test_ds.nc')

    print("Preprocessing completed.")

    if return_data:
        return (train_ds, valid_ds, test_ds)
    return

def parse_arguments():
    """
    Get arguments from the command line.

    Returns
    -------
        tuple: A tuple of parsed arguments as follows:
            - out_dir (str): The output directory to save the preprocessed data files.
            - input_dir (str): The input directory to load the raw dataset from.
            - input_file (str): The filename of the raw dataset file.
            - return_data (bool): Whether or not to return the preprocessed datasets (in addition to saving them to disk).

    Notes
    -----
    - The default values are set according to the `main()` function's default arguments.
    """

    # Declare a "parser" variable
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-o', dest='out_dir', default='data/', help='Specify the output directory to save the preprocessed data files')
    parser.add_argument('-i', dest='input_dir', default='data/', help='Specify the input directory to load the raw dataset from')
    parser.add_argument('-f', dest='input_file', default='IsoGSM/Total.IsoGSM.ERA5.monmean.nc', help='Specify the filename of the raw dataset file')
    parser.add_argument('-r', dest='return_data', default=False, help='Whether or not to return the preprocessed datasets in addition to saving them to disk')

    # Store all the arguments in "args"
    args = parser.parse_args()

    # Save arguments as variables
    out_dir = args.out_dir
    input_dir = args.input_dir
    input_file = args.input_file
    return_data = bool(args.return_data)

    return out_dir, input_dir, input_file, return_data

if __name__ == "__main__":         
    # PARSE COMMAND LINE ARGUMENTS:
    main(*parse_arguments())
    