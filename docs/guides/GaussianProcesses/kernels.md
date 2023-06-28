# GaussianProcesses.kernels

Author: Jakob Thoms

Date: June 2023

[Source code](/src/GaussianProcesses/kernels.py)

## get_kernel

Return the corresponding kernel function based on the provided GP kernel ID.

The function retrieves the desired Gaussian Process (GP) kernel function based on the given kernel ID. Each kernel corresponds to a specific configuration and set of input features. The available kernel configurations are described below:

- ID 0: RBF + PeriodicKernel for month
- ID 1: RBF (deseasonalized; no month)
- ID 2: PiecewisePolynomialKernel + PeriodicKernel for month
- ID 3: PiecewisePolynomialKernel (deseasonalized; no month)
- ID 4: RQKernel + PeriodicKernel for month
- ID 5: RQKernel (deseasonalized; no month)
- ID 7: SpectralMixture (deseasonalized; no month)

The input dataset for the GP model is assumed to have the following feature columns:

- Column 0: Delta 18-O
- Column 1: UPS Easting
- Column 2: UPS Northing
- Column 3: Orography
- Column 4: Distance to Coast
- Column 5: Month (as an integer from 1 to 12) [**Only required if the kenrel ID is even**]

Kernels with an even ID use the month feature as an input variable but do *not* deseasonalize the other variables. Kernels with an odd ID do *not* use the month feature as an input variable and instead deseasonalize the other variables.

### Parameters

- `config_id : int, optional` - The ID of the desired kernel function. Default is 0.

### Returns

- `gpytorch.kernels` - The desired GP kernel function.
