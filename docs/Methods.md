## Methods 

Sections:
> 1. Preprocessing
> 2. Baseline Models 
> 3. Gaussian Process (GP) Models 
> 4. Deep Learning (DL) models 
> 5. Postprocessing and Evaluation 

### 1. Preprocessing 

<img width="1424" alt="preprocessing_flowchart" src="https://github.com/SLIPP-isotopes/MDS_Antarctica/assets/70875076/0b66ecda-4af7-4d74-bdf1-8a2267f34701">


The full IsoGSM xArray dataset (42 years) is first split into 3 smaller datasets: 
> 1. `train_ds`: Has data from 1995 - 2020 inclusive (26 years). This data will be used to train our ML models. 
> 2. `valid_ds`: Has data from 1987 - 1994 inclusive (8 years). This data will be used to continuously evaluate the different ML models that we build.
> 3. `test_ds`: Has data from 1979 - 1986 inclusive (8 years). This data will be used at the very end of our project to get a performance score of the final ML models chosen. 

The `preprocessing.py` script will be called separately on each of these 3 datasets, performing the following steps: 

**1. Select only the data variables in the dataset that we are interested in** 

- `d18O_pr`, `hgtprs`, `pratesfc`, and `tmp2m`

**2. Change the latitude/longitude dimensions for `hgtprs` (geopotential height)**

- Specific to the IsoGSM dataset, the variable `hgtprs` (geopotential height) is encoded on a different latitude and longitude grid than the other variables (dims = `latitude_2`, `longitude_2`). 
- Thus, we use linear interpolation to remap these latitude and longitudes to the grid used by the other variables. 

**3. Deseasonalize our data (optional)**

- When deseasonalizing is selected, the monthly means across all years for each latitude and longitude coordinate is computed for each variable (from data in `train_ds`). Then, the monthly means are subtracted from the original values. 
- When deseasonalizing `valid_ds` and `test_ds`, the monthly means computed from the training data are subtracted from the original values. 

**4. Add new features including polar coordinates, months, and years** 

- Using the [`polarstereo-lonlat-convert-py`](https://github.com/nsidc/polarstereo-lonlat-convert-py) package, the latitudes and longitudes are converted to polar coordinates and added to the preprocessed dataset as new data variables (features). 
- Add a feature indicating the month as an integer (1 to 12).
- Add a feature indicating the year as an integer. 

**5. Interpolate missing values (i.e. NaN entries)**

- For any data variables with missing values (NaNs), the Xarray [advanced interpolation](https://docs.xarray.dev/en/stable/user-guide/interpolation.html#advanced-interpolation) method using the dimensions time, latitude, and longitude to fill in the missingness. 
- Interpolation is only done for `train_ds` - the NaN values are retained in `valid_ds` and `test_ds`.   

**6. Scale variables** 

- The means and standard deviations of each of the data variables are computed (from data in `train_ds`). Then, the original values are scaled (value - mean / standard deviation). 
- When scaling `valid_ds` and `test_ds`, the means and standard deviations from `train_ds` are used. 

### 2. Baseline Models 

The RMSE scores the following simple baseline models are used as a comparison for the final models' performances: 

| Baseline Model  | Description |
| ------------- | ------------- |
| `overall_mean`  | For a given output variable in the validation or test set, it predicts that each `y` value will be the overall mean of the variable in the training set.  |
| `latlon_mean`  | For a given output variable at a specific latitude and longitude position in the validation or test set, it predicts that each `y` value will be the overall mean of the variable in the training set at that specific latitude and longitude over all time points. |
| `latlonmonth_mean`  | For a given output variable at a specific latitude and longitude position and time (month) in the validation or test set, it predicts that each `y` value will be the overall mean of the variable in the training set at that specific latitude and longitude and month over all years. |
| `ols` | For a given output variable, it fits a simple ordinary least squares (ols) regression model relating `x` (`d18O_pr`) to `y` (i.e. `y ~ x`) using the training set. Then, it takes the slope and intercept from the fitted model to predict `y` values from `x` values in the test set. | 

### 3. Gaussian Process (GP) Models 

(unfinished)

GP models were first introduced by {INSERT CITATION}. A GP model extends the OLS model in a way that removes the assumption of uncorrelated observations while also introducing some non-linearity within the model. Assuming that we have a training data set with $n$ observations, let $x_i = [x_{i,1} \ x_{i,2} \ x_{i,3} \ x_{i,4} \ x_{i,5}]^T$ denote the corresponding observed values of the model's input variables, where:

$$
\begin{align*}
    x_{i,1} &= \text{delta-18O} \\
    x_{i,2} &= \text{UPS Easting} \\
    x_{i,3} &= \text{UPS Northing} \\
    x_{i,4} &= \text{Distance-to-coast} \\
    x_{i,5} &= \text{Surface Altitude} \\
\end{align*}
$$

where $i = 1,\dots,n$ and $n$ is the number of training examples. See table 3.1 for more details on the GP model's input variables.

The GP model can be defined as:

$$
y_i = \mu(\mathbf{x}_i) + Z(\mathbf{x}_i)
$$

where:


$$
\mu(\mathbf{x}_i) = ...
$$

$$... = \beta_0 + \sum_{k=1}^5 \beta_k x_{i,k},$$

$$
Z(\mathbf{x}_i) \quad \overset{\text{marginal}}{\sim} \mathcal{N}(0, \sigma^2),
$$

$$
\text{Cov}(Z(\mathbf{x}_i), Z(\mathbf{x}_j)) = \sigma^2 R(\mathbf{x}_i, \mathbf{x}_j).
$$

The kernel function $R$ that models the correlation between observations is given by:

$$
R(\mathbf{x}_i, \mathbf{x}_j) = ...,
$$

$$
... = \prod_{k=1}^5 K_{\text{RBF}}(x_{i,k}, x_{j,k})
$$

$$
= \prod_{k=1}^5 \exp \left( -\frac{1}{2} \theta_k^{-2} \left|x_{i,k} - x_{j,k} \right|^2 \right)
$$

To implement GP models for our project, we will consider different kernel configurations and split our data into training and validation sets. The training set will be used to fit the GP model, while the validation set will be used to tune hyperparameters and select the best kernel configuration.

**Table 3.1: Input (X) and Output (Y) Variables**

| Input Variables                                      | Output Variables                              |
|------------------------------------------------------|-----------------------------------------------|
| Scaled & Deseasonalized† delta 18-O of precipitation  | Scaled & Deseasonalized† Temperature            |
| Scaled Easting Coordinate                            | Scaled & Deseasonalized† Geopotential Height    |
| Scaled Northing Coordinate                           | Scaled & Deseasonalized† Precipitation Rate     |
| Scaled Distance to Coast                             |                                               |
| Scaled Surface Orography                             |                                               |
| Month (integer)††                                    |                                               |

   > Table 3.1 summarizes the GP model's input and output variables. These variables have been preprocessed following the procedure outlined in the [Preprocessing](#preprocessing) section. 
   >
   > † The temporal variables are only deseasonalized for certain kernels (those with an odd ID). 
   >
   > †† The "month" variable is only included for certain kernels (those with an even ID). This variable is an integer where Jan=1 and Dec=12. 

#### GP Kernels

See:
- [RBFKernel](https://docs.gpytorch.ai/en/stable/kernels.html#rbfkernel)
- [PiecewisePolynomialKernel](https://docs.gpytorch.ai/en/stable/kernels.html#piecewisepolynomialkernel)
- [RQKernel](https://docs.gpytorch.ai/en/stable/kernels.html#rqkernel)
- [SpectralMixtureKernel](https://docs.gpytorch.ai/en/stable/kernels.html#spectralmixturekernel)
- [PeriodicKernel](https://docs.gpytorch.ai/en/stable/kernels.html#periodickernel)


**Table 3.2: Kernel Configurations**

| Kernel ID | `GPytorch` Kernels used           | Uses month as an input variable? | Equation                                                                                      |
|-----------|----------------------------------|----------------------------------|-----------------------------------------------------------------------------------------------|
| 0         | RBFKernel + PeriodicKernel        | Yes (non-deseasonalized data)    |                              |
| 1         | RBFKernel                        | No (deseasonalized data)         |                                                               |
| 2         | PiecewisePolynomialKernel + PeriodicKernel | Yes (non-deseasonalized data)    |                |
| 3         | PiecewisePolynomialKernel        | No (deseasonalized data)         |                                                 |
| 4         | RQKernel + PeriodicKernel         | Yes (non-deseasonalized data)    |                            |
| 5         | RQKernel                         | No (deseasonalized data)         |                                                                  |
| 6         | NA                               | NA                               | NA                                                                                            |
| 7         | SpectralMixtureKernel             | No (deseasonalized data)         |                                              |

> Table 3.2 provides information on the kernel configurations used in the GP models. It lists the kernel IDs, the corresponding `GPytorch` kernels used, and whether the month is used as an input variable. It also presents the equation that represents the kernel for each configuration. Please note that kernel 6 (NA) indicates that the SpectralMixtureKernel cannot be combined with the PeriodicKernel.

Note that we use a [PeriodicKernel](https://docs.gpytorch.ai/en/stable/kernels.html#periodickernel) for the `month` feature if it is included. The `month` feature is included if the training data is *not* deseasonalized, but `month` is *not* included if the training data **is** deseasonalized.
Kernels with an even ID use the `month` feature and don't deseasonalize. Kernels with an odd ID deseasonalize and don't use the `month` feature.


#### Training GP Models (Sockeye)

Training GP models involves the computation of pairwise distances between all training data points in a matrix, which requires large memory and time resources. To overcome this challenge, we obtained computational resources from [UBC ARC Sockeye](https://arc.ubc.ca/ubc-arc-sockeye), which allowed us to use computational nodes of up to 186 GB. We also reduced our training dataset of 26 years and ~780,000 examples into smaller splits of a few consecutive years each. See table 3.3 for more information. 

**Table 3.3: GP Model Details and Validation RMSEs on Scaled Anomalies**

| Kernel | Learning Rate | Num. Epochs | Num. Splits | Num. Examples | Temp. RMSE | Geopt. RMSE | Precip. RMSE | Memory (GB) | Runtime (h) |
|--------|---------------|-------------|-------------|---------------|------------|------------|--------------|-------------|-------------|
| 1      | 0.15          | 10          | 13          | ~60,000       | 1.14       | 1.18       | 1.08         | 72          | 3-4         |
| 2      | 0.15          | 10          | 13          | ~60,000       | 1.11       | 1.15       | 1.14         | 155         | 2-3         |
| 3      | 0.15          | 10          | 13          | ~60,000       | 1.13       | 1.16       | 1.14         | 132         | 1-2         |
| **1**  | **0.0015**    | **10**      | **9**       | **~87,000**   | **0.99**   | **1.03**   | **1.00**     | **150**     | **2-3**     |
| 1      | 0.00075       | 10          | 9           | ~87,000       | 0.98       | 1.03       | 1.00         | 150         | 4-5         |

   > Table 3.3 displays a summary of notable configurations run, varying the type of kernel, learning rate, number of splits, and validation RMSE scores on the three climate variables. It also displays the memory and runtime utilized on Sockeye when training the models. The three kernels used were "RBFKernel" (kernel 1), "PiecewisePolynomialKernel" (kernel 2), and "RQKernel" (kernel 3). All three displayed similar performances when trained on ~60,000 examples. However, we noticed that kernel 1 required less memory resources (72 GB compared to >100 GB), which would allow us to include more examples during training. Decreasing the learning rate (from 0.0015 to 0.00075) also did not appear to significantly improve RMSE scores but did increase runtime. Thus, we decided to move forward with a GP model with kernel 1, a learning rate of 0.0015, and 9 splits.

### 4. Deep Learning (DL) Models 

The Neural Network model is a deep convolutional neural network with 6 hidden layers. Each CNN layer passes a 2-dimensional kernel across the 15 by 192 latitude x longitude grid of Antarctica. After each layer the outputs are passed through a ReLU activation function. The specific architecture by layer is specified in the table below:

| Layer  | Input Channels | Output Channels | Kernel Size |
| --- | --- | --- | ---- |
| 1     | 6<sup>1</sup>     | 32     | 5 x 17 |
| 2     | 32     | 32     | 5 x 15 |
| 3     | 32     | 16     | 5 x 13 |
| 4     | 16     | 16     | 3 x 13 |
| 5     | 16     | 8     | 3 x 11 |
| 6     | 8     | 3<sup>2</sup>     | 3 x 9 |

#### <sup>1</sup>Input Variables (6) per Lat/Lon point:
1. Scaled & Deseasonalized delta 18-O of precipitation 
2. Scaled Easting Coordinate
3. Scaled Northing Coordinate
4. Scaled Distance to Coast (negative distances == points at sea)
5. Scaled Surface Orography
6. Land / Sea Boolean Mask

#### <sup>2</sup>Output Variables (3) per Lat/Lon point:
1. Scaled & Deseasonalized Temperature
2. Scaled & Deseasonalized Geopotential Height
3. Scaled & Deseasonalized Precipitation Rate

The model was trained until there was no improvement in validation RMSE for 50 consecutive epochs, which occurred after 215 epochs.

![NN_learning](../img/NN_results/DeepCNNlearning.png)


#### Neural Network Performance


|  | Temperature | Precipitation Rate | Geopotential Height |
| ------------ | -------- | -------- | -------- |
| Test RMSE on Anomalies | 0.96 | 1.07 | 0.96 |
| Test RMSE in Original Units | 2.07 °K | 0.515 mm/day | 54.6 m |



### 5. Postprocessing and Evaluation 

The postprocessing analysis uses a set of 4 plots to check for patterns in the residuals, including across space (lat, lon) and over time (months). Below is an example of these plots using the temperature anomaly residuals from the neural network model on the test dataset:

![Residuals](../img/NN_results/DeepCNN_tmp2m.png)

The two maps on the left visualize the residuals over space. The top left plot shows the average of the residuals at each lat, lon point. If the model consistently over (under) predicts temperature in a certain region, that region will be dark red (blue). Below this is a map of the standard deviation of residuals on the same lat, lon point grid. The gradient from yellow (best) to blue (worst) indicates where the model is precise, regardless of how accurate it is. For example, a region that always overpredicts 1 standard deviation too high would be dark red on the accuracy map but yellow on the precision map, and represents a biased but consistent model.

The top-right plot groups the residuals by month and calculates the average of the residuals in each month using the same scale as the top-left map. This plot detects seasonal patterns in the residuals. In the above example, the darker reds in all months between June and November suggest that the model is systematically biased during this time of year and has failed to fully capture the seasonal cycle in the data.

The final plot on the bottom-right is a simple histogram of all the residuals. Heavier tails here suggest weaker models, and any skewness represents biases the model has learned. 
