import gpytorch as gp
from torch import tensor

def get_kernel(config_id=0):
    """
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
    - Column 5: Month (as an integer from 1 to 12) [**Only required if the kernel ID is even**]

    Kernels with an even ID use the month feature as an input variable but do *not* deseasonalize the other variables.
    Kernels with an odd ID do *not* use the month feature as an input variable and instead deseasonalize the other variables.

    Parameters
    ----------
    config_id : int, optional
        The ID of the desired kernel function. Default is 0.

    Returns
    -------
    gpytorch.kernels
        The desired GP kernel function.
    """
    if not isinstance(config_id, int):
        raise TypeError("config_id should be an integer.")
    
    # RBF + Periodic kernel for month
    if config_id == 0:
        delta_spatial_kernel = gp.kernels.RBFKernel(
            ard_num_dims=5,         # Use 5 of the features in X, and give them each a separate parameter.
            active_dims=(0,1,2,3,4)     # Use the features corresponding to columns 0,1,2,3,4.
        )

        temporal_kernel = gp.kernels.PeriodicKernel(
                ard_num_dims=1,         # Use 1 of the features in X.
                active_dims=(5),        # Use the feature corresponding to column 5 (i.e. the month feature).
                # period_length_prior=gp.priors.NormalPrior(12, 0.1) # Set prior distribution of period length   (NOTE: this is optional).
        )
        temporal_kernel.period_length = tensor(12.) # Initialize the period length to 12 (prior knowledge) (should probably do this whether or not we use a prior).
        
        # Create the kernel, which is the product of the two kernels created above:
        kernel = delta_spatial_kernel * temporal_kernel
        
        return kernel

    # RBF
    # No month variable (uses deasonalized data instead).
    elif config_id == 1:
        delta_spatial_kernel = gp.kernels.RBFKernel(
            ard_num_dims=5,         # Use 5 of the features in X, and give them each a separate parameter.
            # active_dims=(0,1,2,3,4)     # Use the features corresponding to columns 0,1,2,3,4.
        )

        return delta_spatial_kernel
    
    # PiecewisePolynomial kernel + Periodic kernel for month
    elif config_id == 2:
        delta_spatial_kernel = gp.kernels.PiecewisePolynomialKernel(
            ard_num_dims=5,         # Use 5 of the features in X, and give them each a separate parameter.
            active_dims=(0,1,2,3,4)     # Use the features corresponding to columns 0,1,2,3,4.
        )

        temporal_kernel = gp.kernels.PeriodicKernel(
                ard_num_dims=1,         # Use 1 of the features in X.
                active_dims=(5),        # Use the feature corresponding to column 5 (i.e. the month feature).
                # period_length_prior=gp.priors.NormalPrior(12, 0.1) # Set prior distribution of period length   (NOTE: this is optional).
        )
        temporal_kernel.period_length = tensor(12.) # Initialize the period length to 12 (prior knowledge) (should probably do this whether or not we use a prior).
        
        # Create the kernel, which is the product of the kernels specified above:
        kernel = delta_spatial_kernel * temporal_kernel
        
        return kernel
    
    # PiecewisePolynomial kernel
    # No month variable (uses deasonalized data instead).
    elif config_id == 3:
        delta_spatial_kernel = gp.kernels.PiecewisePolynomialKernel(
            ard_num_dims=5,         # Use 5 of the features in X, and give them each a separate parameter.
            # active_dims=(0,1,2,3,4)     # Use the features corresponding to columns 0,1,2,3,4.
        )

        return delta_spatial_kernel
    
    # RQ kernel + Periodic kernel for month
    if config_id == 4:
        delta_spatial_kernel = gp.kernels.RQKernel(
            ard_num_dims=5,         # Use 5 of the features in X, and give them each a separate parameter.
            active_dims=(0,1,2,3,4)     # Use the features corresponding to columns 0,1,2,3,4.
        )

        temporal_kernel = gp.kernels.PeriodicKernel(
                ard_num_dims=1,         # Use 1 of the features in X.
                active_dims=(5),        # Use the feature corresponding to column 5 (i.e. the month feature).
                # period_length_prior=gp.priors.NormalPrior(12, 0.1) # Set prior distribution of period length   (NOTE: this is optional).
        )
        temporal_kernel.period_length = tensor(12.) # Initialize the period length to 12 (prior knowledge) (should probably do this whether or not we use a prior).
        
        # Create the kernel, which is the product of the kernels specified above:
        kernel = delta_spatial_kernel * temporal_kernel
        
        return kernel

    # RQ kernel
    # No month variable (uses deasonalized data instead).
    elif config_id == 5:
        delta_spatial_kernel = gp.kernels.RQKernel(
            ard_num_dims=5,         # Use 5 of the features in X, and give them each a separate parameter.
            # active_dims=(0,1,2,3,4)     # Use the features corresponding to columns 0,1,2,3,4.
        )

        return delta_spatial_kernel

    # SpectralMixture kernel
    # No month variable (uses deasonalized data instead).
    elif config_id == 7:
        delta_spatial_kernel = gp.kernels.SpectralMixtureKernel(
            num_mixtures=4, 
            ard_num_dims=5,         # Use 5 of the features in X, and give them each a separate parameter.
            # active_dims=(0,1,2,3,4)     # Use the features corresponding to columns 0,1,2,3,4.
        )

        return delta_spatial_kernel
    
    else:
        raise ValueError("invalid kernel ID supplied.")
    