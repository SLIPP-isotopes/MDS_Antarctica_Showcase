import numpy as np
import sys
sys.path.append('./src/')
sys.path.append('../src/')
import torch
from torch import nn, optim, tensor
from torchsummary import summary
from Postprocessing.utils import calculate_residuals

class NeuralNetworkModel(torch.nn.Module):
    """Class for training Neural Network architectures using Pytorch.
    
    Extends torch.nn.Module for integration with standard pytorch training
    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    
    initialize_architectures method contains 10 sequential NN architectures, 
    one of which must be specified when initializing the class.
    
    All architectures assume 4d tensor inputs with dimensions: (time, vars, lat, lon)
    """
    
    def __init__(self, architecture, x_vars, y_vars, lr=1e-3):
        """Initialize Neural Network Model with a specified architecture
        
        Parameters
        ----------
        architecture : str
            String identifying which architecture to initialize. 
            Must be a key in self.architectures -- see initialize_architectures()
        x_vars : list of str
            List containing strings of the variable names to include as model inputs.
            These are specified before supplying data so the architecture is initialized with the right # of channels
            Strings should exist in training dataset supplied to model.fit()
        y_vars : list of str
            List contains strings of the variable names to include as model outputs.
            These are specified before supplying data so the architecture is initialized with the right # of channels
            Strings should exist in the training dataset supplied to model.fit()
        lr : numeric
            Learning rate to pass to the optimizer. Default = 0.001
            
        Attributes
        --------
        device : torch.device() object
            Device to run the model on. Currently hardcoded to 'cpu'
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
        loss_fx : pytorch loss function
            Loss function for the model to optimize. Currently hardcoded to nn.MSELoss()
            https://pytorch.org/docs/stable/nn.html#loss-functions
        optimizer : pytorch optimizer
            Optimizer for the model to use. Currently hardcoded to optim.Adam()
            https://pytorch.org/docs/stable/optim.html
            
        Examples
        ----------
        >>> x_vars = ['scaled_deseas_d18O_pr']
        >>> y_vars = ['scaled_deseas_tmp2m', 'scaled_deseas_hgtprs', 'scaled_deseas_pratesfc']
        >>> NeuralNetworkModel('CNN-simple', x_vars, y_vars)
        NeuralNetworkModel(
            (main): Sequential(
                (0): Conv2d(1, 64, kernel_size=(5, 17), stride=(1, 1), padding=(2, 8))
                (1): ReLU()
                (2): Conv2d(64, 3, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4))
                (3): ReLU()
            )
            (loss_fx): MSELoss()
        )
        """
        super().__init__()
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.lr = lr
        
        self.initialize_architectures()
        self.set_architecture(architecture)
        
        self.device = torch.device('cpu') # Switch to GPU here
        self.to(self.device)
        
        self.loss_fx = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return
        
    def initialize_architectures(self):
        """Helper class to store and initialize pre-defined architectures we can train.
        
        These are effectively class attributes but they are loaded on init(). Add new architectures here.
        
        All architectures are some combination of 2d convolution layers along the lat, lon dimensions (CNN),
        or flattened and fully-connected "linear" layers (FCNN). All use ReLU() activation functions between layers.
                    
        The size of the latitude and longitude dimensions must be known before defining the architectures, 
        and are hardcoded to match the current format of our data (15 lats x 192 lons).
        
        
        Current Defined Architectures
        ----------
        'CNN-simple' : 2 Layer CNN
        'CNN-wide': 2 Layer CNN with 1024 hidden channels
            CNN hidden channels means 1024 values per latxlon point
        'CNN-deep': 4 Layer CNN
        'CNN-deep2': 6 Layer CNN 
            ** THIS WAS OUR BEST PERFORMING MODEL
        'Linear-narrow': 3 Layer FCNN with only 128 hidden channels
            FCNN hidden channels means 128 values for all latxlon points
            ** THIS MODEL TRAINS VERY QUICKLY, AND HAS GOOD PERFORMANCE ON hgtprs
        'Linear-wide': 3 Layer FCNN with 8192 hidden channels
        'Linear-deep': 5 Layer FCNN
        'Hybrid': 1 CNN Layer + 1 FCNN Layer
        'Hybrid-deep': 2 CNN Layers + 2 FCNN Layers
        'Hybrid-narrow': 2 CNN layers + 2 FCNN Layers with only 128 hidden channels
        
        
        """
        lats = 15
        lons = 192
        self.architectures = {
            "CNN-simple": nn.Sequential(
                # 2 layer CNN, x -> 64 -> y
                nn.Conv2d(len(self.x_vars), 64, (5, 17), padding=(2, 8)),
                nn.ReLU(),
                nn.Conv2d(64, len(self.y_vars), (3, 9), padding=(1, 4)),
                nn.ReLU()),
            "CNN-wide": nn.Sequential(
                # 2 layer CNN, x -> 1024 -> y
                nn.Conv2d(len(self.x_vars), 1024, (5, 17), padding=(2, 8)),
                nn.ReLU(),
                nn.Conv2d(1024, len(self.y_vars), (3, 9), padding=(1, 4)),
                nn.ReLU()),
            "CNN-deep": nn.Sequential(
                # 4 layer CNN, x -> 64 -> 64 -> 16 -> y
                nn.Conv2d(len(self.x_vars), 64, (5, 17), padding=(2, 8)),
                nn.ReLU(),
                nn.Conv2d(64, 64, (5, 13), padding=(2, 6)),
                nn.ReLU(),
                nn.Conv2d(64, 16, (3, 11), padding=(1, 5)),
                nn.ReLU(),
                nn.Conv2d(16, len(self.y_vars), (3, 9), padding=(1, 4)),
                nn.ReLU()),
            "CNN-deep2": nn.Sequential(
                # 6 layer CNN, x -> 32 -> 32 -> 16 -> 16 -> 8 -> y
                nn.Conv2d(len(self.x_vars), 32, (5, 17), padding=(2, 8)),
                nn.ReLU(),
                nn.Conv2d(32, 32, (5, 15), padding=(2, 7)),
                nn.ReLU(),
                nn.Conv2d(32, 16, (5, 13), padding=(2, 6)),
                nn.ReLU(),
                nn.Conv2d(16, 16, (3, 13), padding=(1, 6)),
                nn.ReLU(),
                nn.Conv2d(16, 8, (3, 11), padding=(1, 5)),
                nn.ReLU(),
                nn.Conv2d(8, len(self.y_vars), (3, 9), padding=(1, 4)),
                nn.ReLU()),
            "Linear-narrow": nn.Sequential(
                # 3 layer flat linear, x-flat -> 128 -> 128 -> y-flat
                nn.Flatten(),
                nn.Linear(len(self.x_vars)*lats*lons, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, len(self.y_vars)*lats*lons),
                nn.ReLU(),
                nn.Unflatten(1, (-1, lats, lons))),
            "Linear-wide": nn.Sequential(
                # 3 layer flat linear, x-flat -> 8192 -> 8192 -> y-flat
                nn.Flatten(),
                nn.Linear(len(self.x_vars)*lats*lons, 8192),
                nn.ReLU(),
                nn.Linear(8192, 8192),
                nn.ReLU(),
                nn.Linear(8192, len(self.y_vars)*lats*lons),
                nn.ReLU(),
                nn.Unflatten(1, (-1, lats, lons))),
            "Linear-deep": nn.Sequential(
                # 5 layer flat linear, x-flat -> 2048 -> 1024 -> 1024 -> 2048 -> y-flat
                nn.Flatten(),
                nn.Linear(len(self.x_vars)*lats*lons, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, len(self.y_vars)*lats*lons),
                nn.ReLU(),
                nn.Unflatten(1, (-1, lats, lons))),
            "Hybrid": nn.Sequential(
                # 1 convolution layer plus 1 linear layer, x -> pool(3->1) -> x-flat -> y-flat
                nn.Conv2d(len(self.x_vars), 3, (5, 17), padding=(2, 8)),
                nn.ReLU(),
                nn.MaxPool2d((1, 3)),
                nn.Flatten(),
                nn.Linear(3*lats*lons//3, len(self.y_vars)*15*192),
                nn.ReLU(),
                nn.Unflatten(1, (-1, 15, 192))),
            "Hybrid-deep": nn.Sequential(
                # 2 conv layers plus 2 linear layers, x -> 64 -> pool(3->1) -> x-flat -> 3*lats*lons -> y-flat
                nn.Conv2d(len(self.x_vars), 64, (5, 17), padding=(2, 8)),
                nn.ReLU(),
                nn.Conv2d(64, 3, (3, 9), padding=(1, 4)),
                nn.ReLU(),
                nn.MaxPool2d((1, 3)),
                nn.Flatten(),
                nn.Linear(3*lats*lons//3, 3*lats*lons),
                nn.ReLU(),
                nn.Linear(3*lats*lons, len(self.y_vars)*lats*lons),
                nn.ReLU(),
                nn.Unflatten(1, (-1, lats, lons))),
            "Hybrid-narrow": nn.Sequential(
                # 2 conv layers plus 2 linear layers, x -> 32 -> x-flat -> 128 -> y-flat
                nn.Conv2d(len(self.x_vars), 32, (5, 17), padding=(2, 8)),
                nn.ReLU(),
                nn.Conv2d(32, 3, (3, 9), padding=(1, 4)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3*lats*lons, 128),
                nn.ReLU(),
                nn.Linear(128, len(self.y_vars)*lats*lons),
                nn.ReLU(),
                nn.Unflatten(1, (-1, lats, lons)))
        }
        return  
        
    def forward(self, x):
        out = self.main(x)
        return out
    
    def set_architecture(self, arch):
        self.main = self.architectures[arch]
        return
    
    def fit(self, X_train, Y_train, X_valid, Y_valid, e = 1e-4, stall_limit = 10, max_epochs = 1000):
        """
        Train this model using training and validation pytorch tensors. These tensors can be generated 
        from xArray datasets using xArray_to_tensor() helper function.
        
        Learned weights are stored in NeuralNetworkModel object. Call predict() after fit() to generate predictions.

        Has "smart" stopping criteria that stops training when model stops learning, 
        specifically when model shows no improvement in valid_RMSE after stall_limit consecutive epochs.
        
        Trains on X_train and Y_train using self.optimizer and self.loss_fx.
        Prints validation RMSE after each epoch. Outputs epoch-by-epoch training progress as dictionary.
        
        Parameters
        ----------
        X_train, Y_train, X_valid, Y_valid : pytorch tensors 
            Dimensions must match self.x_vars and self.y_vars respectively
        e : numeric 
            significant threshold. Improvement must be larger than e to count when determining stopping criteria
        stall_limit : int 
            number of consecutive epochs without learning before stopping training
        max_epochs : int 
            Max limit on number of epochs to train, even if model has not stopped according to stopping criteria
                
        Returns
        ----------
        dictionary with 3 keys ('train_rmse', 'valid_rmse', 'valid_std_resid') : 
            Contains 3 lists with epoch-by-epoch results
        """
       
        train_rmse, valid_rmse, valid_std_resid = [], [], []
        best_valid_rmse = np.inf
        no_improvement_count = 0

        for epoch in range(max_epochs): 
            # Training
            self.train()        

            X = X_train.to(self.device) 
            Y = Y_train.to(self.device)
            self.optimizer.zero_grad()
            Y_hat = self(X)
            loss = self.loss_fx(Y_hat, Y)
            loss.backward()
            self.optimizer.step()

            train_resids = Y - Y_hat
            train_resids = train_resids.detach().numpy().flatten()
            train_batch_rmse = np.sqrt(np.mean((train_resids)**2)) #Calculate RMSE
            train_rmse.append(train_batch_rmse) 


            # Validation
            self.eval()

            with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood
                Y_hat = self(X_valid)
                valid_resids = Y_valid - Y_hat
                valid_resids = valid_resids.numpy().flatten()

            valid_resids = valid_resids[~np.isnan(valid_resids)] #Remove NaNs if any
            valid_batch_rmse = np.sqrt(np.mean((valid_resids)**2))
            valid_rmse.append(valid_batch_rmse)
            valid_std_resid.append(np.std(valid_resids))

            # Check if model is still learning
            if valid_batch_rmse + e < best_valid_rmse: # Model learned something significant
                best_valid_rmse = valid_batch_rmse
                no_improvement_count = 0
            else: # Model didn't learn
                no_improvement_count += 1
                if no_improvement_count == stall_limit: # Model stopped learning
                    break
                
            print(f'\r Epoch: {epoch} Val. RMSE: {valid_batch_rmse:.4f}')

        # After break, return by epoch results as dictionary
        results = {"train_rmse": train_rmse,
                   "valid_rmse": valid_rmse,
                   "valid_std_resid": valid_std_resid}
        return results
    
    def predict(self, X):
        """Use this model to predict targets given an input pytorch tensor X.
        
        Should only be called after fit() otherwise we're predicting using an untrained model.
        
        Parameters
        ----------
        X : 4d pyTorch Tensor
            Dimensions should be (time, vars, lat, lon) where vars match the variables in self.x_vars,
            and lat and lon match the dimensions specified in initialize_architecture() -- 15 x 192
            
        Returns
        ----------
        preditions : 4d pyTorch Tensor
            Output dimensions are (vars, time, lat, lon) for easy integration back into xArray datasets
        """
        self.eval()
        with torch.no_grad(): 
            Y_hat = self(X)
            preds = Y_hat.numpy() 
            preds = np.swapaxes(preds, 0, 1)
        return preds
    
    def predict_inplace(self, ds):
        """Use this model to predict targets and residuals on an input xArray test dataset
        
        This method wraps predict() by taking a preprocessed xArray as is, making predictions on the 
        input x variables specified in self.x_vars, and inserting the resulting predictions directly
        back into the original dataset. For convenience, it also calculates and appends the prediction
        residuals into the original dataset.
        
        Parameters
        ----------
        ds : xArray dataset
            Must contain all the data variables in self.x_vars and self.y_vars, 
            all of which have 'time', 'latitude', and 'longitude' dimensions.
            
        Returns
        ----------
        ds : xArray dataset
            Returns the same dataset with 2 * y_vars additional data variables, 
            the model's predictions and residuals for each target.
        """
        X_test, _ = xArray_to_tensor(ds, self.x_vars, self.y_vars)
        preds = self.predict(X_test)

        for i in range(len(self.y_vars)):
            ds[f'pred_{self.y_vars[i]}'] = (['time', 'latitude', 'longitude'], preds[i])
        ds = calculate_residuals(ds)
        
        return ds
    
    
def xArray_to_tensor(ds, x_vars, y_vars):
    """Utility function that converts from xArray to Tensor via numpy
    
    Selects specified x_vars y_vars out of the input xArray dataset 
    
    Used to prepare pytorch tensors in the required format for deep learning.
    Includes converting to Float32 and swapping axes 0 and 1 to reorder dimensions like: 
        (time, vars, lat, lon).
    
    Parameters
    ----------
    ds : xArray dataset
        Must contain all of the data variables in self.x_vars and self.y_vars,
        all of which must have 'time', 'latitude', and 'longitude' dimensions.
    x_vars : list of str
        List of variables to include in the input (X) tensor - must be data variables in ds.
    y_vars : list of str
        List of variables to include in the target (Y) tensor - must be data variables in ds.
    
    Returns
    ----------
    X : 4d pyTorch tensor
        X tensor with dimensions (time, vars, lat, lon), where vars has size of len(x_vars)
    Y: 4d pyTorch tensor
        Y tensor with dimensions (time, vars, lat, lon), where vars has size of len(y_vars)
    """
    X = tensor(np.swapaxes(ds[x_vars].to_array().to_numpy().astype(np.float32), 0, 1))
    Y = tensor(np.swapaxes(ds[y_vars].to_array().to_numpy().astype(np.float32), 0, 1))
    
    return X, Y
