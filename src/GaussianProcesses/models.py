import torch
import gpytorch as gp

class GP():
    """
    Gaussian Process (GP) regression model.
    
    This is a wrapper for storing all objects related to a GP model in GPytorch.
    - e.g. the model, the likelihood, and the training data.

    The `GP` class also handles training and predictions.

    Parameters
    ----------
    model : gpytorch.models.ExactGP
        GP model class to use.
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the GP model.
    train_X : torch.Tensor
        Tensor containing the training input data.
    train_Y : torch.Tensor
        Tensor containing the training target data.
    **kwargs :
        Additional keyword arguments to be passed to the GP model.

    Attributes
    ----------
    train_X : torch.Tensor
        Tensor containing the training input data.
    train_Y : torch.Tensor
        Tensor containing the training target data.
    initial_likelihood : gpytorch.likelihoods.Likelihood
        Initial likelihood function. Used for training with restarts.
    initial_model : gpytorch.models.ExactGP
        Initial GP model. Used for training with restarts.
    initial_kwargs : dict
        Initial keyword arguments. Used for training with restarts.
    model : gpytorch.models.ExactGP
        Current (trained) GP model.
    likelihood : gpytorch.likelihoods.Likelihood
        Current likelihood function.
    total_epochs : int
        Total number of epochs trained.

    Methods
    -------
    initialize_model(model, likelihood, **kwargs)
        Initialize the GP model and likelihood function. 
        This updates the current `model` and `likelihood` attributes.
    warmup(learning_rate=0.1, num_restarts=3, num_epochs_per_restart=3)
        Perform warm-up training for the GP model.
    train(learning_rate=0.1, num_epochs=5)
        Train the GP model.
    training_loop(model, loss, optimizer, train_inputs, train_outputs)
        Training loop for a single epoch.
    _is_trained()
        Check if the model has been trained.
    get_posterior(X)
        Get the posterior distribution for the given input data.
    predict(X)
        Perform prediction for the given input data.
    """
    def __init__(self, model, likelihood, train_X, train_Y, **kwargs):
        """
        Initialize a Gaussian Process (GP) regression model.

        This is a wrapper for storing all objects related to a GP model in GPytorch.
        - e.g. the model, the likelihood, and the training data.

        Parameters
        ----------
        model : gpytorch.models.ExactGP
            GP model class to use.
        likelihood : gpytorch.likelihoods.Likelihood
            Likelihood function for the GP model.
        train_X : torch.Tensor
            Tensor containing the training input data.
        train_Y : torch.Tensor
            Tensor containing the training target data.
        **kwargs :
            Additional keyword arguments to be passed to the GP model.
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.initial_likelihood = likelihood
        self.initial_model = model
        self.initial_kwargs = {**kwargs}
        self.model = None
        self.likelihood = None
        self.initialize_model(self.initial_model, self.initial_likelihood, **kwargs)
        self.total_epochs = 0

    def _is_trained(self):
        """
        Check if the GP model has been trained.

        Returns
        -------
        bool
            True if the model has been trained, False otherwise.
        """
        if self.total_epochs > 0:
            return True
        return False

    def initialize_model(self, model, likelihood, **kwargs):
        """
        Initialize the GP model and likelihood function.
        
        Used for training with restarts.

        Parameters
        ----------
        model : gpytorch.models.ApproximateGP or gpytorch.models.ExactGP
            GP model class to use.
        likelihood : gpytorch.likelihoods.Likelihood
            Likelihood function for the GP model.
        **kwargs :
            Additional keyword arguments to be passed to the GP model.
        """
        self.likelihood = likelihood()
        self.model = model(
            train_x=self.train_X, train_y=self.train_Y, 
            likelihood=self.likelihood, 
            **kwargs
        )

    def warmup(self, learning_rate=0.1, num_restarts=3, num_epochs_per_restart=3):
        """
        Perform warm-up training for the GP model.

        Warm-up training is a short initial training phase that helps the model to converge faster and with more stability during the main training phase. 
        It randomly initializes the model hyperparameters and then trains the model for a few epochs.
        Then it saves the results (model hyperparameters and loss), and randomly re-initializes the model hyperparameters.
        After repeating for a given number of restarts, the hyperparameters giving the best (i.e. lowest) loss are chosen.
        This helps to mitigate the effects of random hyperparameter initialization and helps to ensure that a global optimum is found when minimizing the loss function.

        Parameters
        ----------
        learning_rate : float, optional
            Learning rate for the optimizer (default is 0.1).
        num_restarts : int, optional
            Number of times to restart the training (default is 3).
        num_epochs_per_restart : int, optional
            Number of epochs to train in each restart (default is 3).
        """
        if self._is_trained():
            raise RuntimeError("The GP model has already been trained. Cannot perform warm-up training.")
    
        self.initialize_model(self.initial_model, self.initial_likelihood, **self.initial_kwargs)
    
        results = []
        for restart in range(num_restarts):
            # Get current model and likelihood
            model, likelihood = self.model, self.likelihood

            # Get into training mode
            model.train()
            likelihood.train()

            # Use the adam optimizer. Optimization includes GaussianLikelihood parameters
            adam = torch.optim.Adam(model.parameters(), lr=learning_rate) 

            # "Loss" for GPs - the negative marginal log likelihood
            mll = gp.mlls.ExactMarginalLogLikelihood(likelihood, model)
            neg_mll = lambda x,y: -mll(x,y)

            for epoch in range(num_epochs_per_restart):
                print('Restart %d/%d, Epoch %d/%d:' % (restart + 1, num_restarts, epoch + 1, num_epochs_per_restart))
                self.training_loop(
                    model=model, loss=neg_mll, optimizer=adam, 
                    train_inputs=self.train_X, train_outputs=self.train_Y
                )
                print('--')

            with torch.no_grad():
                model_outputs = model(self.train_X) # Get model output
                loss = neg_mll(model_outputs, self.train_Y)  # Calc loss
            results.append(
                (loss.item(), self.model.state_dict())
            )

            self.initialize_model(self.initial_model, self.initial_likelihood, **self.initial_kwargs)
            print('===')

        results = sorted(results, key=lambda x: x[0])
        best_model_state_dict = results[0][1]
        self.model.load_state_dict(best_model_state_dict)
        self.total_epochs = num_epochs_per_restart

    def train(self, learning_rate=0.1, num_epochs=5):
        """
        Train the GP model.

        The `train()` method trains the GP model. It minimizes the model's negative marginal log likelihood using the Adam optimizer.
        The method sets the model and likelihood in training mode, initializes the Adam optimizer, and defines the negative marginal log likelihood as the loss function. 
        It then performs the training loop for the specified number of epochs, printing the loss and noise values at each epoch. 
        Finally, it updates the total number of epochs trained.

        Parameters
        ----------
        learning_rate : float, optional
            Learning rate for the optimizer (default is 0.1).
        num_epochs : int, optional
            Number of epochs to train the model (default is 5).
        """
        model, likelihood = self.model, self.likelihood
        prev_epochs = self.total_epochs

        # Get into training mode
        model.train()
        likelihood.train()

        # Use the adam optimizer, optimization includes GaussianLikelihood parameters
        adam = torch.optim.Adam(model.parameters(), lr=learning_rate) 

        # "Loss" for GPs - the negative marginal log likelihood
        mll = gp.mlls.ExactMarginalLogLikelihood(likelihood, model)
        neg_mll = lambda x,y: -mll(x,y)

        for i in range(num_epochs):
            print('Epoch %d/%d:' % (i + prev_epochs + 1, num_epochs + prev_epochs))
            self.training_loop(
                model=model, loss=neg_mll, optimizer=adam, 
                train_inputs=self.train_X, train_outputs=self.train_Y
            )
            print('--')
            self.total_epochs += 1


    def training_loop(self, model, loss, optimizer, train_inputs, train_outputs):
        """
        Training loop for a single epoch.

        Parameters
        ----------
        model : gpytorch.models.ExactGP
            The GP model to train.
        loss : callable
            The loss function to use.
        optimizer : torch.optim.Optimizer
            The optimizer used for updating the model parameters.
        train_inputs : torch.Tensor
            The input data for training.
        train_outputs : torch.Tensor
            The target data for training.
        """
        optimizer.zero_grad() # Zero gradients from previous epoch
        model_outputs = model(train_inputs) # Get model output

        # Calc loss and backprop gradients
        loss = loss(model_outputs, train_outputs)
        loss.backward()

        # Print progress
        print('\t Loss: %.3f   noise: %.3f' % (
            loss.item(),
            model.likelihood.noise.item()
        ))

        optimizer.step()
    
    @torch.no_grad()
    def get_posterior(self, X):
        """
        Get the posterior distribution for the given input data.

        Parameters
        ----------
        X : torch.Tensor
            Tensor containing the input data for which the posterior is computed.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Posterior distribution over the output values.
        """
        model, likelihood = self.model, self.likelihood

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gp.settings.fast_pred_var():
            posterior = likelihood(model(X)) # Make predictions by feeding model through likelihood
                
        return posterior

    @torch.no_grad()
    def predict(self, X):
        """
        Perform prediction for the given input data.

        Parameters
        ----------
        X : torch.Tensor
            Tensor containing the input data for which predictions are made.

        Returns
        -------
        torch.Tensor
            Predicted mean values of the output.
        """
        with torch.no_grad():
            posterior = self.get_posterior(X)
            pred = posterior.mean
        return pred 
        
class ExactGPModel(gp.models.ExactGP):
    """
    The simplest form of GP model, exact inference.

    This class represents a Gaussian Process (GP) model using exact inference. 
    It extends the `ExactGP` class from the `gpytorch` library.

    Parameters
    ----------
    train_x : torch.Tensor
        Tensor containing the training input data.
    train_y : torch.Tensor
        Tensor containing the training target data.
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the GP model.
    mean : str or gpytorch.means.Mean, optional
        Mean function for the GP model. Default is 'constant'.
    kernel : str or gpytorch.kernels.Kernel, optional
        Kernel function for the GP model. Default is 'rbf'.

    Attributes
    ----------
    mean_module : gpytorch.means.Mean
        Mean module of the GP model.
    covar_module : gpytorch.kernels.Kernel
        Covariance module of the GP model.
    num_features : int
        Number of features in the training data.

    Methods
    -------
    forward(x)
        Perform forward pass through the GP model.

    Notes
    -----
    The input dataset for the GP model is assumed to have the following feature columns:
    - Column 0: Delta 18-O
    - Column 1: UPS Easting
    - Column 2: UPS Northing
    - Column 3: Orography
    - Column 4: Distance to Coast
    - Column 5: Month (as an integer from 1 to 12)
    
    The `mean` and `kernel` parameters determine the mean and kernel functions used in the GP model. 

    The following logic is applied to handle the `mean` parameter:
    - If `mean` is None, then it is set to 'constant'.
    - If `mean` is 0 or 'constant', the GP model's mean module is set to `gpytorch.means.ConstantMean()`.
    - If `mean` is 1 or 'linear', the GP model's mean module is set to `gpytorch.means.LinearMean(5)`,  with a dimension of 5 corresponding to the feature columns 'd18O_pr', 'E', 'N', 'oro', and 'dist_to_coast'.
    - Otherwise, it is assumed that `mean` is a `gpytorch.means` object.

    The following logic is applied to handle the `kernel` parameter:
    - If `kernel` is 'rbf' or None, the covariance module is set to `gpytorch.kernels.RBFKernel()`.
    - Otherwise, it is assumed that `kernel` is a `gpytorch.kernels` object.
    - Some pre-made kernel architectures can be loaded using get_kernel(config_id) (see src/GaussianProcesses/kernels.py):
        - ID 0: RBF + PeriodicKernel for month
        - ID 1: RBF (deseasonalized; no month)
        - ID 2: PiecewisePolynomialKernel + PeriodicKernel for month
        - ID 3: PiecewisePolynomialKernel (deseasonalized; no month)
        - ID 4: RQKernel + PeriodicKernel for month
        - ID 5: RQKernel (deseasonalized; no month)
        - ID 7: SpectralMixture (deseasonalized; no month)

    """
    def __init__(
            self, train_x, train_y, likelihood, 
            mean='constant', kernel='rbf'
        ):
        """
        Create an object that represents an Gaussian Process (GP) model using exact inference. 
        It extends the `ExactGP` class from the `gpytorch` library.

        Parameters
        ----------
        train_x : torch.Tensor
            Tensor containing the training input data.
        train_y : torch.Tensor
            Tensor containing the training target data.
        likelihood : gpytorch.likelihoods.Likelihood
            Likelihood function for the GP model.
        mean : str or gpytorch.means.Mean, optional
            Mean function for the GP model. Default is 'constant'.
        kernel : str or gpytorch.kernels.Kernel, optional
            Kernel function for the GP model. Default is 'rbf'.

        Notes
        -----

        The following logic is applied to handle the `mean` parameter:
        - If `mean` is None, then it is set to 'constant'.
        - If `mean` is 0 or 'constant', the GP model's mean module is set to `gpytorch.means.ConstantMean()`.
        - If `mean` is 1 or 'linear', the GP model's mean module is set to `gpytorch.means.LinearMean(5)`,  with a dimension of 5 corresponding to the feature columns 'd18O_pr', 'E', 'N', 'oro', and 'dist_to_coast'.
        - Otherwise, it is assumed that `mean` is a `gpytorch.means` object.

        The following logic is applied to handle the `kernel` parameter:
        - If `kernel` is 'rbf' or None, the covariance module is set to `gpytorch.kernels.RBFKernel()`.
        - Otherwise, it is assumed that `kernel` is a `gpytorch.kernels` object.
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_features = train_x.shape[1]

        if mean == 0:
            mean = 'constant'
        elif mean == 1:
            mean = 'linear'
        
        if mean == 'constant' or mean is None:
            mean = gp.means.ConstantMean()
        elif mean == 'linear':
            mean = gp.means.LinearMean(5)
            # 'd18O_pr', 'E', 'N', 'oro', 'dist_to_coast'

        if kernel == 'rbf' or kernel is None:
            kernel = gp.kernels.RBFKernel()
        

        self.mean_module = mean
        self.covar_module = kernel

        if isinstance(self.covar_module, gp.kernels.spectral_mixture_kernel.SpectralMixtureKernel):
            print('spectral!')
            self.covar_module.initialize_from_data(train_x, train_y)
        
    def forward(self, x):
        """
        Perform a forward pass through the GP model.
        In other words, compute the predictive distribution for the given input data.

        Parameters
        ----------
        x : torch.Tensor
            Tensor containing the input data for which to compute the predictive distribution.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            MultivariateNormal distribution representing the predictive mean and covariance for the input data.

        Notes
        -----
        The input tensor `x` is expected to have shape equal to (batch_size, num_features), 
            where `num_features` is the number of features in the training data and `batch_size` is the number of examples in the input batch.
        For our purposes, the shape should be either (batch_size, 5) or (batch_size, 6), 
            where the first five feature columns are 'd18O_pr', 'E', 'N', 'oro', and 'dist_to_coast', and the the (optional) sixth feature column is 'month'.

        The method first computes the mean of the predictive distribution by passing the relevant subset of `x` to the GP model's mean module. 
        The subset is determined by slicing the input tensor using `x[:, :5]` when the mean is 'linear', and `x` itself when the mean is 'constant' or custom.

        Next, the method computes the covariance of the predictive distribution by 
            passing the complete input tensor `x` to the GP model's covariance module (i.e. the GP kernel function).

        Finally, it returns a MultivariateNormal distribution object representing the predictive mean and covariance for the input data.
        """
        mean_x = self.mean_module(x[:, :5])  # Compute the mean of the predictive distribution; columns 0,1,2,3,4<=>'d18O_pr', 'E', 'N', 'oro', 'dist_to_coast'
        covar_x = self.covar_module(x)  # Compute the covariance of the predictive distribution

        return gp.distributions.MultivariateNormal(mean_x, covar_x)
        


