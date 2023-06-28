# Postprocessing.utils

Authors: Daniel Cairns, Jakob Thoms

Date: June 2023

[Source code](/src/Postprocessing/utils.py)

## `calculate_residuals(ds)`

Given an xArray dataset with predicted model outputs, calculate the residuals for each predicted variable and each scale.

The function will use `scale_unscale_predictions` to convert all predicted variables in the dataset to all other possible scales.
The scales which residuals are calculated on are:

- 'scaled_deseas': Residuals scaled and deseasonalized.
- 'scaled': Residuals scaled only.
- 'deseas': Residuals deseasonalized only.
- 'original': Residuals on the original scale of the predicted variable. No scaling nor deseasonalization.

**Parameters**

- `ds` : xr.Dataset
  - The input XArray dataset containing the predicted model outputs.

**Returns**

- `xr.Dataset`
  - A new XArray dataset with additional variables representing the predicted variables and their corresponding residuals.

**Notes**

- The function assumes that the input dataset contains predicted variables in the format 'pred_{var}'.
- The function performs the following calculations for each predicted variable:
  - Scaled residuals: 'resid_scaled_{var}' = 'scaled_{var}' - 'pred_scaled_{var}'
  - Unscaled residuals: 'resid_{var}' = '{var}' - 'pred_{var}'
  - Deseasonalized residuals: 'resid_deseas_{var}' = 'deseas_{var}' - 'pred_deseas_{var}'
  - Scaled and deseasonalized residuals: 'resid_scaled_deseas_{var}' = 'scaled_deseas_{var}' - 'pred_scaled_deseas_{var}'
- The function assumes that the necessary scaling factors and seasonal means are available as attributes in the input dataset.


## `scale_unscale_predictions(ds)`

Given an xArray dataset with predicted model outputs, convert the predictions for each variable to each output scale.

The function will perform the necessary calculations to convert all predicted variables in the dataset to all other possible scales.
The scales to which predictions are converted are:

- 'scaled_deseas': Predictions scaled and deseasonalized.
- 'scaled': Predictions scaled only.
- 'deseas': Predictions deseasonalized only.
- 'original': Predictions on the original scale of the variable. No scaling nor deseasonalization.

**Parameters**

- `ds` : xr.Dataset
  - The input XArray dataset containing the predicted model outputs.

**Returns**

- `xr.Dataset`
  - A new XArray dataset with the converted predictions in each scale.

**Notes**

- The function assumes that the input dataset contains predicted variables in the format 'pred_{var}'.
- The function performs the following conversions for each predicted variable:
  - Scaled predictions: 'pred_scaled_{var}' = ('pred_{var}' - scaler_mean_{var}) / scaler_std_{var}
  - Deseasonalized predictions: 'pred_deseas_{var}' = 'pred_{var}' - 'seasonal_means_{var}'
  - Scaled and deseasonalized predictions: 'pred_scaled_deseas_{var}' = (pred_deseas_{var} - scaler_mean_deseas_{var}) / scaler_std_deseas_{var}
- The function assumes that the necessary scaling factors and seasonal means are available as attributes in the input dataset.


## `calculate_model_RMSE(dataset, scale='scaled_deseas', return_overall=False)`

Calculate the root mean square errors (RMSEs) of residuals for all predicted variables in the dataset.

**Parameters**

- `dataset` : xr.Dataset
  - The input XArray dataset containing the residuals.
- `scale` : str, optional
  - The scale of the residuals. Default is 'scaled_deseas'.
  - Valid options:
    - 'scaled_deseas': Residuals scaled and deseasonalized.
    - 'scaled': Residuals scaled only.
    - 'deseas': Residuals deseasonalized only.
    - 'original': Residuals on the original scale of the predicted variable. No scaling nor deseasonalization.
- `return_overall` : bool, optional
  - Whether to include the overall RMSE in the output dictionary. Default is False.

**Returns**

- `dict`
  - A dictionary containing the RMSEs of residuals for each predicted variable.
  - The keys are the variable names, and the values are the corresponding RMSEs.
  - If `return_overall` is True, an additional entry with key 'overall' and value of the overall RMSE is included.

**Raises**

- `ValueError`
  - If the specified scale is not valid.

**Notes**

- Residual variables are expected to be named in the format 'resid_{scale}_{var}', where {scale} is the specified scale and {var} is the variable name.
- The overall RMSE is the square root of the average MSE across all predicted variables.


## `generate_residual_plots(dataset, variable, label, model_name, out_file, show_plot=True)`

Given an xArray dataset and a 'resid_' variable it contains, generate a 2x2 grid of residual diagnostics plots to visualize the performance of a model.

**Parameters**

- `dataset` : xr.Dataset
  - The input XArray dataset.
- `variable` : str
  - The name of the resid_ variable in the dataset.
- `label` : str, optional
  - The label to be included in the plot title. Default is an empty string.
- `model_name` : str, optional
  - The name of the model to be included in the plot title. Default is an empty string.
- `out_file` : str or None
  - The output file path to save the plot. If None, the plot is not saved.
- `show_plot` : bool, optional
  - Whether to display the plot. Default is True.

**Returns**

- None

**Notes**

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


## `get_pie_slice_colors(mo_means, std)`

Helper function to apply a colormap to pie slices based on the mean values and standard deviation.
This function determines the colors corresponding to each pie slice.

**Parameters**

- `mo_means` : array-like
  - Array-like object containing the mean values of the pie slices.
- `std` : float
  - The standard deviation used for color scaling.

**Returns**

- array-like
  - Array-like object containing the colors corresponding to each pie slice.

**Notes**

- The function applies a colormap to the pie slices based on the mean values.
- The colormap is scaled using the provided standard deviation.


## `generate_time_series_plots(dataset, output_var="tmp2m", output_scale='scaled_deseas_', num_years="all", sites=None, subplot_titles=None, label="", model_name="", include_random_deltas=False, out_file="./time_series_plots.png", show_plot=True)`

Given an xArray dataset and a 'pred_' variable it contains, generate time series plots overlaying model predictions and actual values at fixed latitutde/longitude locations.

**Parameters**

- `dataset` : xr.Dataset
  - The input xArray dataset.
- `output_var` : str, optional
  - The variable to be plotted. Default is "tmp2m".
- `output_scale` : str, optional
  - The scale of the output variable. Default is "scaled_deseas_".
  - "scaled_deseas_": Scaled and deseasonalized output.
  - "scaled_": Scaled output.
  - "deseas_": Deseasonalized output.
  - "original": Original output scale.
- `num_years` : int or str, optional
  - The number of years to be plotted. Default is "all" to plot all years.
  - If an integer is provided, the specified number of years will be plotted.
- `sites` : list of dict, optional
  - The site locations specified as a list of dictionaries.
  - Each dictionary should contain 'lat' and 'lon' keys representing the latitude and longitude, respectively.
  - Default is None, which uses the default site locations.
- `subplot_titles` : list of str, optional
  - The titles for each subplot. Default is None, which uses default titles based on the site indices.
- `label` : str, optional
  - The label to be included in the plot title. Default is an empty string.
- `model_name` : str, optional
  - The name of the model to be included in the plot title. Default is an empty string.
- `out_file` : str or None, optional
  - The output file path for saving the plot. Default is "./time_series_plots.png".
  - If set to None, the plot will not be saved.
- `show_plot` : bool, optional
  - Whether to display the plot. Default is True.

**Returns**

- None

**Notes**

- The `sites` parameter allows specifying custom site locations as a list of dictionaries.
- The default site locations are:
  - Kohnen station: 75°00’06"S, 0°04’04" E
  - James Ross Island: 64°10′S, 57°45′W
  - WAIS Divide: 79°28′03″S, 112°05′11″W
  - Dome C: 75°05′59″S, 123°19′56″E
- The `subplot_titles` parameter allows customizing the titles of each subplot.
- The `label` and `model_name` parameters are used in the overall plot title.
