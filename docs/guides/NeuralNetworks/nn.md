# NeuralNetwork.nn

Author: Daniel Cairns

Date: June 2023

[Source code](/src/NeuralNetwork/nn.py)


## NeuralNetworkModel

Class for training Neural Network architectures using Pytorch.

Extends `torch.nn.Module` for integration with standard pytorch training
[PyTorch Neural Networks Tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

`initialize_architectures` method contains 10 sequential NN architectures,
one of which must be specified when initializing the class.

All architectures assume 4d tensor inputs with dimensions: (time, vars, lat, lon).

### Parameters

- `architecture` : str
  - String identifying which architecture to initialize.
  - Must be a key in `self.architectures` -- see `initialize_architectures()`
- `x_vars` : list of str
  - List containing strings of the variable names to include as model inputs.
  - These are specified before supplying data so the architecture is initialized with the right # of channels.
  - Strings should exist in training dataset supplied to `model.fit()`.
- `y_vars` : list of str
  - List contains strings of the variable names to include as model outputs.
  - These are specified before supplying data so the architecture is initialized with the right # of channels.
  - Strings should exist in the training dataset supplied to `model.fit()`.
- `lr` : numeric
  - Learning rate to pass to the optimizer. Default = 0.001

### Attributes

- `device` : `torch.device()` object
  - Device to run the model on. Currently hardcoded to 'cpu'
  - [torch.device documentation](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
- `loss_fx` : pytorch loss function
  - Loss function for the model to optimize. Currently hardcoded to `nn.MSELoss()`
  - [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- `optimizer` : pytorch optimizer
  - Optimizer for the model to use. Currently hardcoded to `optim.Adam()`
  - [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)

### Methods

- `initialize_architectures()`
  - Helper class to store and initialize pre-defined architectures we can train.
  - These are effectively class attributes but they are loaded on `init()`. Add new architectures here.
  - All architectures are some combination of 2d convolution layers along the lat, lon dimensions (CNN),
    or flattened and fully-connected "linear" layers (FCNN). All use ReLU() activation functions between layers.
  - The size of the latitude and longitude dimensions must be known before defining the architectures,
    and are hardcoded to match the current format of our data (15 lats x 192 lons).
  - Current Defined Architectures:
    - 'CNN-simple' : 2 Layer CNN
    - 'CNN-wide': 2 Layer CNN with 1024 hidden channels
      - CNN hidden channels means 1024 values per latxlon point
    - 'CNN-deep': 4 Layer CNN
    - 'CNN-deep2': 6 Layer CNN
      - **THIS WAS OUR BEST PERFORMING MODEL
    - 'Linear-narrow': 3 Layer FCNN with only 128 hidden channels
      - FCNN hidden channels means 128 values for all latxlon points
      - **THIS MODEL TRAINS VERY QUICKLY, AND HAS GOOD PERFORMANCE ON hgtprs
    - 'Linear-wide': 3 Layer FCNN with 8192 hidden channels
    - 'Linear-deep': 5 Layer FCNN
    - 'Hybrid': 1 CNN Layer + 1 FCNN Layer
    - 'Hybrid-deep': 2 CNN Layers + 2 FCNN Layers
    - 'Hybrid-narrow': 2 CNN layers + 2 FCNN Layers with only 128 hidden channels

- `forward(self, x)`
- Forward pass of the Neural Network model

- `set_architecture(self, arch)`
- Current Defined Architectures (valid values for the parameter `arch`):
  - 'CNN-simple' : 2 Layer CNN
  - 'CNN-wide': 2 Layer CNN with 1024 hidden channels
    - CNN hidden channels means 1024 values per latxlon point
  - 'CNN-deep': 4 Layer CNN
  - 'CNN-deep2': 6 Layer CNN
    - **THIS WAS OUR BEST PERFORMING MODEL
  - 'Linear-narrow': 3 Layer FCNN with only 128 hidden channels
    - FCNN hidden channels means 128 values for all latxlon points
    - **THIS MODEL TRAINS VERY QUICKLY, AND HAS GOOD PERFORMANCE ON hgtprs
  - 'Linear-wide': 3 Layer FCNN with 8192 hidden channels
  - 'Linear-deep': 5 Layer FCNN
  - 'Hybrid': 1 CNN Layer + 1 FCNN Layer
  - 'Hybrid-deep': 2 CNN Layers + 2 FCNN Layers
  - 'Hybrid-narrow': 2 CNN layers + 2 FCNN Layers with only 128 hidden channels

- `fit(self, X_train, Y_train, X_valid, Y_valid, e = 1e-4, stall_limit = 10, max_epochs = 1000)`
  - Train this model using training and validation pytorch tensors. These tensors can be generated
    from xArray datasets using `xArray_to_tensor()` helper function.
  - Learned weights are stored in NeuralNetworkModel object. Call `predict()` after `fit()` to generate predictions.
  - Has "smart" stopping criteria that stops training when model stops learning,
    specifically when model shows no improvement in valid_RMSE after `stall_limit` consecutive epochs.
  - Trains on `X_train` and `Y_train` using `self.optimizer` and `self.loss_fx`.
  - Prints validation RMSE after each epoch. Outputs epoch-by-epoch training progress as dictionary.
  - **Parameters**:
    - `X_train`, `Y_train`, `X_valid`, `Y_valid` : pytorch tensors
      - Dimensions must match `self.x_vars` and `self.y_vars` respectively.
    - `e` : numeric
      - significant threshold. Improvement must be larger than `e` to count when determining stopping criteria.
    - `stall_limit` : int
      - number of consecutive epochs without learning before stopping training.
    - `max_epochs` : int
      - Max limit on number of epochs to train, even if model has not stopped according to stopping criteria.
  - **Returns**:
    - dictionary with 3 keys (`'train_rmse'`, `'valid_rmse'`, `'valid_std_resid'`) :
      - Contains 3 lists with epoch-by-epoch results.

- `predict(self, X)`
  - Use this model to predict targets given an input pytorch tensor `X`.
  - Should only be called after `fit()` otherwise we're predicting using an untrained model.
  - **Parameters**:
    - `X` : 4d pyTorch Tensor
      - Dimensions should be (time, vars, lat, lon) where `vars` match the variables in `self.x_vars`,
        and `lat` and `lon` match the dimensions specified in `initialize_architecture()` -- 15 x 192
  - **Returns**:
    - `predictions` : 4d pyTorch Tensor
      - Output dimensions are (vars, time, lat, lon) for easy integration back into xArray datasets

- `predict_inplace(self, ds)`
  - Use this model to predict targets and residuals on an input xArray test dataset
  - This method wraps `predict()` by taking a preprocessed xArray as is, making predictions on the
    input `x` variables specified in `self.x_vars`, and inserting the resulting predictions directly
    back into the original dataset. For convenience, it also calculates and appends the prediction
    residuals into the original dataset.
  - **Parameters**:
    - `ds` : xArray dataset
      - Must contain all the data variables in `self.x_vars` and `self.y_vars`,
        all of which have 'time', 'latitude', and 'longitude' dimensions.
  - **Returns**:
    - `ds` : xArray dataset
      - Returns the same dataset with 2 * `y_vars` additional data variables,
        the model's predictions and residuals for each target.

- `xArray_to_tensor(ds, x_vars, y_vars)`
  - Utility function that converts from xArray to Tensor via numpy
  - Selects specified `x_vars` and `y_vars` out of the input xArray dataset
  - Used to prepare pytorch tensors in the required format for deep learning.
  - Includes converting to Float32 and swapping axes 0 and 1 to reorder dimensions like:
    (time, vars, lat, lon).
  - **Parameters**:
    - `ds` : xArray dataset
      - Must contain all of the data variables in `self.x_vars` and `self.y_vars`,
        all of which must have 'time', 'latitude', and 'longitude' dimensions.
    - `x_vars` : list of str
      - List of variables to include in the input (X) tensor - must be data variables in `ds`.
    - `y_vars` : list of str
      - List of variables to include in the target (Y) tensor - must be data variables in `ds`.
  - **Returns**:
    - `X` : 4d pyTorch tensor
      - X tensor with dimensions (time, vars, lat, lon), where `vars` has size of len(`x_vars`).
    - `Y` : 4d pyTorch tensor
      - Y tensor with dimensions (time, vars, lat, lon), where `vars` has size of len(`y_vars`).
