# GaussianProcesses.models

Author: Jakob Thoms

Date: June 2023

[Source code](/src/GaussianProcesses/models.py)


## GP

Gaussian Process (GP) regression model.

This is a wrapper for storing all objects related to a GP model in GPytorch.

- e.g. the model, the likelihood, and the training data.

The `GP` class also handles training and predictions.

### Parameters

- `model` : gpytorch.models.ExactGP - GP model class to use.
- `likelihood` : gpytorch.likelihoods.Likelihood - Likelihood function for the GP model.
- `train_X` : torch.Tensor - Tensor containing the training input data.
- `train_Y` : torch.Tensor - Tensor containing the training target data.
- `**kwargs` : Additional keyword arguments to be passed to the GP model.

### Attributes

- `train_X` : torch.Tensor - Tensor containing the training input data.
- `train_Y` : torch.Tensor - Tensor containing the training target data.
- `initial_likelihood` : gpytorch.likelihoods.Likelihood - Initial likelihood function. Used for training with restarts.
- `initial_model` : gpytorch.models.ExactGP - Initial GP model. Used for training with restarts.
- `initial_kwargs` : dict - Initial keyword arguments. Used for training with restarts.
- `model` : gpytorch.models.ExactGP - Current (trained) GP model.
- `likelihood` : gpytorch.likelihoods.Likelihood - Current likelihood function.
- `total_epochs` : int - Total number of epochs trained.

### Methods

- `initialize_model(model, likelihood, **kwargs)` - Initialize the GP model and likelihood function. This updates the current `model` and `likelihood` attributes.
- `warmup(learning_rate=0.1, num_restarts=3, num_epochs_per_restart=3)` - Perform warm-up training for the GP model.
- `train(learning_rate=0.1, num_epochs=5)` - Train the GP model.
- `training_loop(model, loss, optimizer, train_inputs, train_outputs)` - Training loop for a single epoch.
- `_is_trained()` - Check if the model has been trained.
- `get_posterior(X)` - Get the posterior distribution for the given input data.
- `predict(X)` - Perform prediction for the given input data.

## ExactGPModel

The simplest form of GP model, exact inference.

This class represents a Gaussian Process (GP) model using exact inference.
It extends the `ExactGP` class from the `gpytorch` library.

### Parameters

- `train_x` : torch.Tensor - Tensor containing the training input data.
- `train_y` : torch.Tensor - Tensor containing the training target data.
- `likelihood` : gpytorch.likelihoods.Likelihood - Likelihood function for the GP model.
- `mean` : str or gpytorch.means.Mean, optional - Mean function for the GP model. Default is 'constant'.
- `kernel` : str or gpytorch.kernels.Kernel, optional - Kernel function for the GP model. Default is 'rbf'.

### Attributes

- `mean_module` : gpytorch.means.Mean - Mean module of the GP model.
- `covar_module` : gpytorch.kernels.Kernel - Covariance module of the GP model.
- `num_features` : int - Number of features in the training data.

### Methods

- `forward(x)` - Perform forward pass through the GP model.

### Notes

The input dataset for the GP model is assumed to have the following feature columns:

- Column 0: Delta 18-O
- Column 1: UPS Easting
- Column 2: UPS Northing
- Column 3: Orography
- Column 4: Distance to Coast
- Column 5: Month (as an integer from 1 to 12)

The mean and kernel parameters determine the mean and kernel functions used in the GP model. 
Please refer to the original docstring for the handling of these parameters.
