import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
import cartopy.crs as ccrs 
import sys
sys.path.append('../../src/') # This lets us access the code in ../../src/
sys.path.append('../src/') # This lets us access the code in ../src/
sys.path.append('./src/') # This lets us access the code in ./src/
from Preprocessing.utils import datetime_to_years

MONTH_LABELS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

def calculate_residuals(ds):
    """
    Given an xArray dataset with predicted model outputs, calculate the residuals for each predicted variable and each scale.

    The function will use `scale_unscale_predictions` to convert all predicted variables in the dataset to all other possible scales.
    The scales which residuals are calculated on are:
        - 'scaled_deseas': Residuals scaled and deseasonalized.
        - 'scaled': Residuals scaled only.
        - 'deseas': Residuals deseasonalized only.
        - 'original': Residuals on the original scale of the predicted variable. No scaling nor deseasonalization.
    
    Parameters:
    -----------
    ds : xr.Dataset
        The input XArray dataset containing the predicted model outputs.

    Returns:
    --------
    xr.Dataset
        A new XArray dataset with additional variables representing the predicted variables and their corresponding residuals.

    Notes:
    ------
    - The function assumes that the input dataset contains predicted variables in the format 'pred_{var}'.
    - The function performs the following calculations for each predicted variable:
        - Scaled residuals: 'resid_scaled_{var}' = 'scaled_{var}' - 'pred_scaled_{var}'
        - Unscaled residuals: 'resid_{var}' = '{var}' - 'pred_{var}'
        - Deseasonalized residuals: 'resid_deseas_{var}' = 'deseas_{var}' - 'pred_deseas_{var}'
        - Scaled and deseasonalized residuals: 'resid_scaled_deseas_{var}' = 'scaled_deseas_{var}' - 'pred_scaled_deseas_{var}'
    - The function assumes that the necessary scaling factors and seasonal means are available as attributes in the input dataset.
    """
    ds = ds.copy()

    # Find predictions
    predicted_vars = []
    for var in ds.keys():
        if var.startswith("pred_"):
            predicted_vars.append(var)
    assert len(predicted_vars) > 0, "No predictions present in dataset."

    ds = scale_unscale_predictions(ds)

    # Calculate residuals (Y - Y-hat)
    for var in predicted_vars:
        # The variable names for randomized delta predictions end with "__<dist>_random_deltas" and so have to be dealt with separately.
        if not var.endswith("random_deltas"): 
            # Strip the "pred_" prefix from the predicted variable name
            var_name = var.split("_")[-1]  
            true_var_name = var_name
        else:
            # Strip the "pred_" prefix AND the "__<dist>_random_deltas" suffix from the predicted variable name
            true_var_name = var.split("__")[0].split("_")[-1]   # The actual variable name
            var_name = true_var_name + "__" +  var.split("__")[-1] # Add back the the "__<dist>_random_deltas" suffix (this is for the corresponding "resid_" var)
        ds[f'resid_{var_name}'] = ds[true_var_name] - ds[f'pred_{var_name}']
        ds[f'resid_scaled_{var_name}'] = ds[f'scaled_{true_var_name}'] - ds[f'pred_scaled_{var_name}']
        ds[f'resid_deseas_{var_name}'] = ds[f'deseas_{true_var_name}'] - ds[f'pred_deseas_{var_name}']
        ds[f'resid_scaled_deseas_{var_name}'] = ds[f'scaled_deseas_{true_var_name}'] - ds[f'pred_scaled_deseas_{var_name}']
        
    return ds

def scale_unscale_predictions(ds):
    """
    Given an xArray dataset with predicted model outputs, convert the predictions for each variable to each output scale.

    The function will perform the necessary calculations to convert all predicted variables in the dataset to all other possible scales.
    The scales to which predictions are converted are:
        - 'scaled_deseas': Predictions scaled and deseasonalized.
        - 'scaled': Predictions scaled only.
        - 'deseas': Predictions deseasonalized only.
        - 'original': Predictions on the original scale of the variable. No scaling nor deseasonalization.

    Parameters:
    -----------
    ds : xr.Dataset
        The input XArray dataset containing the predicted model outputs.

    Returns:
    --------
    xr.Dataset
        A new XArray dataset with the converted predictions in each scale.

    Notes:
    ------
    - The function assumes that the input dataset contains predicted variables in the format 'pred_{var}'.
    - The function performs the following conversions for each predicted variable:
        - Scaled predictions: 'pred_scaled_{var}' = ('pred_{var}' - scaler_mean_{var}) / scaler_std_{var}
        - Deseasonalized predictions: 'pred_deseas_{var}' = 'pred_{var}' - 'seasonal_means_{var}'
        - Scaled and deseasonalized predictions: 'pred_scaled_deseas_{var}' = (pred_deseas_{var} - scaler_mean_deseas_{var}) / scaler_std_deseas_{var}
    - The function assumes that the necessary scaling factors and seasonal means are available as attributes in the input dataset.
    """
    ds = ds.copy()

    # Find predictions
    predicted_vars = []
    for var in ds.keys():
        if var.startswith("pred_"):
            predicted_vars.append(var)
    assert len(predicted_vars) > 0, "No predictions present in dataset."

    # Convert the predicted variables to all scales
    for var in predicted_vars:
        if var.startswith("pred_scaled_"): # This should always be true but check for sanity
            var_name = var[12:]
            if var_name.endswith("random_deltas"):
                true_var_name = var_name.split("__", maxsplit = 1)[0]
            else:
                true_var_name = var_name
            scaler_mean = ds.attrs[f'scaler_mean_{true_var_name}']
            scaler_std = ds.attrs[f'scaler_stdev_{true_var_name}']
            ds[f'pred_{var_name}'] = ds[var] * scaler_std + scaler_mean

            if var_name.startswith('deseas_'): # Add seasonal means back if required
                true_var_name_deseas = true_var_name.split("_", maxsplit = 1)[-1]
                var_name_deseas = var_name.split("_", maxsplit = 1)[-1]
                ds[f'pred_{var_name_deseas}'] = ds[f'pred_{var_name}'] + \
                                                ds[f'seasonal_means_{true_var_name_deseas}']
                scaler_mean = ds.attrs[f'scaler_mean_{true_var_name_deseas}']
                scaler_std = ds.attrs[f'scaler_stdev_{true_var_name_deseas}']
                ds[f'pred_scaled_{var_name_deseas}'] = (ds[f'pred_{var_name_deseas}'] - scaler_mean) / scaler_std
            else:   # Otherwise subtract seasonal means
                ds[f'pred_deseas_{var_name}'] = ds[f'pred_{var_name}'] - \
                                                ds[f'seasonal_means_{true_var_name}']
                scaler_mean = ds.attrs[f'scaler_mean_deseas_{true_var_name}']
                scaler_std = ds.attrs[f'scaler_stdev_deseas_{true_var_name}']
                ds[f'pred_scaled_deseas_{var_name}'] = (ds[f'pred_deseas_{var_name}'] - scaler_mean) / scaler_std

    return ds
        
def calculate_model_RMSE(dataset, scale='scaled_deseas', return_overall=False):
    """
    Calculate the root mean square errors (RMSEs) of residuals for all predicted variables in the dataset.

    Parameters:
    -----------
    dataset : xr.Dataset
        The input XArray dataset containing the residuals.
    scale : str, optional
        The scale of the residuals. Default is 'scaled_deseas'.
        Valid options:
        - 'scaled_deseas': Residuals scaled and deseasonalized.
        - 'scaled': Residuals scaled only.
        - 'deseas': Residuals deseasonalized only.
        - 'original': Residuals on the original scale of the predicted variable. No scaling nor deseasonalization.
    return_overall : bool, optional
        Whether to include the overall RMSE in the output dictionary. Default is False.

    Returns:
    --------
    dict
        A dictionary containing the RMSEs of residuals for each predicted variable.
        The keys are the variable names, and the values are the corresponding RMSEs.
        If `return_overall` is True, an additional entry with key 'overall' and value of the overall RMSE is included.

    Raises:
    -------
    ValueError
        If the specified scale is not valid.

    Notes:
    ------
    - Residual variables are expected to be named in the format 'resid_{scale}_{var}', where {scale} is the specified scale and {var} is the variable name.
    - The overall RMSE is the square root of the average MSE across all predicted variables.
    """
    VALID_SCALES = ['scaled_deseas', 'scaled', 'deseas', 'original']
    if not scale in VALID_SCALES:
        raise ValueError(f"Error in calculate_model_RMSE: 'scale' should be one of {VALID_SCALES}.") 
    if scale == 'original':
        scale = ''

    rmses = {}
    overall_mse = 0
    for var in dataset.keys():
        if var.startswith("resid_" + scale):
            resids = dataset[var].to_numpy().flatten()
            mse = np.nanmean((resids)**2)
            overall_mse += mse
            rmses[var[6:]] = np.sqrt(mse)
    if return_overall:
        overall_rmse = np.sqrt(overall_mse / len(rmses))
        rmses['overall'] = overall_rmse

    return rmses

def generate_residual_plots(dataset, variable, label, model_name, out_file, show_plot=True): 
    """
    Given an xArray dataset and a 'resid_' variable it contains, generate a 2x2 grid of residual diagnostics plots to visualize the performance of a model.

    Parameters:
    -----------
    dataset : xr.Dataset
        The input XArray dataset.
    variable : str
        The name of the resid_ variable in the dataset.
    label : str, optional
        The label to be included in the plot title. Default is an empty string.
    model_name : str, optional
        The name of the model to be included in the plot title. Default is an empty string.
    out_file : str or None
        The output file path to save the plot. If None, the plot is not saved.
    show_plot : bool, optional
        Whether to display the plot. Default is True.

    Returns:
    --------
    None

    Notes:
    ------
    - The function generates a 2x2 grid of residual diagnostic plots:
        - a histogram of residual
        - a "pie chart" of average residuals by month (model accuracy)
        - a heatmap of the average residuals (model accuracy)
        - a heatmap of the standard deviation of residuals (model precision)
    - The histogram plot displays the binned residuals with dashed lines indicating 2 standard deviations.
    - The "pie chart" plot shows the average residuals by month.
    - The mean residuals and standard deviation of residuals are plotted on a map projection.
    - The function uses standard deviation values from the dataset attributes for color scaling.
    - The `label` and `model_name` parameters are used in the overall plot title.
    """
    
    # Calculate std for color scaling (2 * standard deviation will be deepest color)
    std = 1
    if "_scaled_" not in variable:
        # Read stdev from dataset attributes
        key = variable.replace('resid_', 'scaler_stdev_')
        if key.endswith("random_deltas"):
            key = key.split("__")[0]
        std = dataset.attrs[key]
    stdev_lines = [-2*std, -1*std, 0, 1*std, 2*std]
    
    
    # Plots 2x2 Grid
    fig = plt.figure(figsize = (8,8), layout = 'constrained')
    ax1 = fig.add_subplot(224)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(221, projection = ccrs.SouthPolarStereo())
    ax4 = fig.add_subplot(223, projection = ccrs.SouthPolarStereo())
    # I rearranged the plots so 1-2-3-4 are not in the right order in the code
    
    
    # Plot #1 Histogram of reiduals
    dataset[variable].plot.hist(
        ax = ax1,
        color = 'black',
        bins = [
            -5, -4.6, -4.2, -3.8, -3.4, -3, 
            -2.6, -2.2, -1.8, -1.4, -1, -0.6, -0.2, 
            0.2, 0.6, 1, 1.4, 1.8, 2.2, 2.6, 
            3, 3.4, 3.8, 4.2, 4.6, 5
        ],
        rwidth = 0.9
    )
    ax1.set_title("Binned Residuals (Stdev. in Pink)")
    ax1.set_xlabel(label)
    for line in stdev_lines:
        ax1.axvline(x=line, color='pink', linestyle='dashed')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.get_yaxis().set_visible(False)
    
    # Plot #2 "Pie Chart" of Time
    mo_means = dataset[variable].groupby('time.month').mean(dim = ['time', 'latitude', 'longitude'])
    ax2.pie(
        x = [1]*12, 
        labels = MONTH_LABELS,
        labeldistance = 0.6,
        startangle = 90,
        counterclock = False,
        colors = get_pie_slice_colors(mo_means, std),
        rotatelabels = True
    )
    ax2.set_title("Average Residuals by Month")
    
    # Plot #3 Mean of residuals
    mean_data = dataset[variable].mean(dim = "time")
    mean_data.plot.pcolormesh(
        ax = ax3,
        transform = ccrs.PlateCarree(), 
        cmap = plt.cm.RdBu, 
        robust = True,
        vmin = -1.5 * std,
        vmax = 1.5 * std,
        levels = 50,
        cbar_kwargs = {
            "orientation": "vertical",
            "label": label,
            "pad": 0.25,
            "shrink": 0.8,
            "ticks": stdev_lines
        }
    )
    ax3.coastlines()
    ax3.gridlines()
    ax3.set_title("Average Residuals (accuracy)")
    
    # Plot #4 Standard Deviation of Residuals
    std_data = dataset[variable].std(dim = "time")
    std_data.plot.pcolormesh(
        ax = ax4,
        transform = ccrs.PlateCarree(), 
        cmap = mpl.colormaps['viridis_r'],
        robust = True,
        vmin = 0,
        vmax = 2 * std,
        levels = 50,
        cbar_kwargs = {
            "orientation": "vertical",
            "label": label,
            "pad": 0.25,
            "shrink": 0.8,
            "ticks": [0, std, 2*std]
        }
    )
    ax4.coastlines()
    ax4.gridlines()
    ax4.set_title("Std. Dev. of Residuals (precision)")

    plt.suptitle('Model Residual Analysis - ' + model_name + '\n' + label)

    if out_file is not None:
        plt.savefig(out_file)
    
    if show_plot:
        plt.show()
    
    return

def get_pie_slice_colors(mo_means, std):
    """
    Helper function to apply a colormap to pie slices based on the mean values and standard deviation.
    This function determines the colors corresponding to each pie slice.

    Parameters:
    -----------
    mo_means : array-like
        Array-like object containing the mean values of the pie slices.
    std : float
        The standard deviation used for color scaling.

    Returns:
    --------
    array-like
        Array-like object containing the colors corresponding to each pie slice.

    Notes:
    ------
    - The function applies a colormap to the pie slices based on the mean values.
    - The colormap is scaled using the provided standard deviation.
    """
    norm = col.Normalize(-1.5*std, 1.5*std)
    colors = plt.cm.RdBu(norm(mo_means))
    return colors

def generate_time_series_plots(
        dataset, output_var="tmp2m", output_scale='scaled_deseas_', num_years="all",
        sites=None, subplot_titles=None, label="", model_name="", include_random_deltas=False,
        out_file="./time_series_plots.png", show_plot=True):
    """
    Given an xArray dataset and a 'pred_' variable it contains, generate time series plots overlaying model predictions and actual values at fixed latitutde/longitude locations.

    Parameters:
    -----------
    dataset : xr.Dataset
        The input xArray dataset.
    output_var : str, optional
        The variable to be plotted. Default is "tmp2m".
    output_scale : str, optional
        The scale of the output variable. Default is "scaled_deseas_".
        - "scaled_deseas_": Scaled and deseasonalized output.
        - "scaled_": Scaled output.
        - "deseas_": Deseasonalized output.
        - "original": Original output scale.
    num_years : int or str, optional
        The number of years to be plotted. Default is "all" to plot all years.
        If an integer is provided, the specified number of years will be plotted.
    sites : list of dict, optional
        The site locations specified as a list of dictionaries.
        Each dictionary should contain 'lat' and 'lon' keys representing the latitude and longitude, respectively.
        Default is None, which uses the default site locations.
    subplot_titles : list of str, optional
        The titles for each subplot. Default is None, which uses default titles based on the site indices.
    label : str, optional
        The label to be included in the plot title. Default is an empty string.
    model_name : str, optional
        The name of the model to be included in the plot title. Default is an empty string.
    out_file : str or None, optional
        The output file path for saving the plot. Default is "./time_series_plots.png".
        If set to None, the plot will not be saved.
    show_plot : bool, optional
        Whether to display the plot. Default is True.

    Returns:
    --------
    None

    Notes:
    ------
    - The `sites` parameter allows specifying custom site locations as a list of dictionaries.
    - The default site locations are:
        - Kohnen station: 75°00’06"S, 0°04’04" E
        - James Ross Island: 64°10′S, 57°45′W
        - WAIS Divide: 79°28′03″S, 112°05′11″W
        - Dome C: 75°05′59″S, 123°19′56″E
    - The `subplot_titles` parameter allows customizing the titles of each subplot.
    - The `label` and `model_name` parameters are used in the overall plot title.
    """
        
    VALID_OUTPUT_VARS = ["hgtprs", "pratesfc", "tmp2m"]
    if not output_var in VALID_OUTPUT_VARS:
        raise ValueError(f"'output_var' should be one of {VALID_OUTPUT_VARS}")
    
    VALID_OUTPUT_SCALES = ["scaled_deseas_", "scaled_", "deseas_", "original", ""]
    if output_scale is None:
        output_scale = "scaled_deseas_"
    elif output_scale != "original" and output_scale != "":
        if not output_scale.endswith("_"):
            output_scale = output_scale + "_"
    if not output_scale in VALID_OUTPUT_SCALES:
        raise ValueError(f"'output_scale' should be one of {VALID_OUTPUT_SCALES}")
    if output_scale == "original":
        output_scale = ""

    # Default site locations:
    if sites is None:
        sites = [
            {'lat': -75.001667, 'lon': 0.067778},   # Kohnen station: 75°00’06"S, 0°04’04" E
            {'lat': -64.166667, 'lon': -57.75},     # James Ross Island: 64°10′S, 57°45′W
            {'lat': -79.4675, 'lon': -112.086389},   # WAIS Divide: 79°28′03″S, 112°05′11″W
            {'lat': -75.099722, 'lon': 123.332222}, # Dome C: 75°05′59″S, 123°19′56″E
            
        ]
        subplot_titles = [
            "Kohnen Station", 
            "James Ross Island", 
            "WAIS Divide", 
            "Dome C"
        ]
    elif not isinstance(sites, list):
        raise TypeError("'sites' should be a list of site locations.")
    elif not isinstance(sites[0], dict):
        raise TypeError("'sites' should be a list of dictionaries of site locations.")
    elif not len(sites) == 4:
        raise TypeError("'sites' should be a list of exactly *four* site locations.")

    # Default subplot titles (if not using default site locations):
    if subplot_titles is None:
        subplot_titles = [
            "Site 0", 
            "Site 1", 
            "Site 2", 
            "Site 3"
        ]

    # Get year range:
    all_years = sorted(np.unique(datetime_to_years(dataset.time.values)))
    start_year = all_years[0]
    if num_years == "all" or num_years is None:
        end_year = all_years[-1]
    elif not isinstance(num_years, int):
        raise TypeError("'num_years' must be an integer.")
    elif num_years > 0 and num_years <= len(all_years):
        end_year = all_years[num_years - 1]
    else: 
        raise ValueError(f"'num_years' must be an integer between 1 and {len(all_years)}.")
    
    # Create a 2x2 grid of subplots
    fig = plt.figure(figsize = (8,8))
    axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]

    for i in range(4):
        # Get the site location and wrangle the corresponding subset of the input dataset.
        site_lat = sites[i]['lat']
        site_lon = sites[i]['lon']
        site_dataset = dataset.copy().sel(latitude=site_lat, longitude=site_lon, method="nearest")
        site_dataset = site_dataset.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

        ground_truth_line = site_dataset[f"{output_scale}{output_var}"].plot.line(
            ax = axes[i],
            color = 'black',
            linewidth = 1
        )
        prediction_line = site_dataset[f"pred_{output_scale}{output_var}"].plot.line(
            ax = axes[i],
            color = 'blue',
            linewidth = 1
        )
        if include_random_deltas:
            random_delta_prediction_line = site_dataset[f"pred_{output_scale}{output_var}__uniform_random_deltas"].plot.line(
                ax = axes[i],
                color = 'red',
                linewidth = 0.6
            )
        if output_scale == "":
            seasonal_mean_line = site_dataset[f"seasonal_means_{output_var}"].plot.line(
                ax = axes[i],
                color = 'green',
                linestyle = 'dashed'
            )
        axes[i].set_title(subplot_titles[i])
        if i < 2:
            axes[i].set_xlabel(None)
            axes[i].get_xaxis().set_ticklabels([])
        if output_var == "tmp2m":
            if output_scale == "scaled_deseas_":
                axes[i].set_ylabel("Temperature (monthly anomalies)")
            else:
                axes[i].set_ylabel(f"{output_scale}Temperature")
        elif output_var == "hgtprs":
            if output_scale == "scaled_deseas_":
                axes[i].set_ylabel("Geopotential Height (monthly anomalies)")
            else:
                axes[i].set_ylabel(f"{output_scale}Geopotential Height")
        elif output_var == "pratesfc":
            if output_scale == "scaled_deseas_":
                axes[i].set_ylabel("Precipitation (monthly anomalies)")
            else:
                axes[i].set_ylabel(f"{output_scale}Precipitation")
        else:
            axes[i].set_ylabel(f"{output_scale}{output_var}")
    if include_random_deltas:
        if output_scale == "":
            legend_labels = ["ground-truth", "prediction", "prediction (random deltas)", "seasonal mean"]
        else:
            legend_labels = ["ground-truth", "prediction", "prediction (random deltas)"]
    else:
        if output_scale == "":
            legend_labels = ["ground-truth", "prediction", "seasonal mean"]
        else:
            legend_labels = ["ground-truth", "prediction"]
    
    plt.figlegend(
        legend_labels, 
        loc='upper center',
        bbox_to_anchor=(0.5, .55),
        ncol=2, 
        fancybox=True, 
        shadow=True
    )
    plt.suptitle(f"{model_name}\n{label}")
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    if out_file is not None:
        plt.savefig(out_file)
    
    if show_plot:
        plt.show()
    
    return