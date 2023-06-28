# MDS_Antarctica

![antarctica](./img/antarctica.png)
_<div dir = "rtl">Photo by <a href="https://unsplash.com/it/@goosegrease?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Derek Oyen</a> on <a href="https://unsplash.com/images/travel/antarctica?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a><div dir = "ltr">_

The isotopic composition of ice cores (i.e. $δ^{18}O$) is a proxy for understanding historical climate and its variability in Antarctica. This project aims to use machine learning (ML) and data science methods to model the relationship between $δ^{18}O$ and three climate variables: temperature, precipitation, and geopotential height. 

Simulated data from the IsoGSM climate model (Yoshimura et al. 2008) was preprocessed and fed into ordinary least squares (OLS) linear regression, Gaussian Process (GP), and Neural Network (NN) models. The models were built in Python scripts (found under [MDS_Antarctica/src/](./src/)) using a variety of libraries (outlined under [Prerequisites](#0-prerequisites)). Cloud computing resources from [Advanced Research Computing](https://arc.ubc.ca/ubc-arc-sockeye) at the University of British Columbia were also used to train computationally intensive GP models. An iterative process was used to try a variety of parameters and architectures. The predictions from each model were evaluated quantitatively using Root Mean Squared Error (RMSE) scores, and qualitatively using heatmaps of residuals. A discussion of the final results of this project can be found in the [final report](./docs/final_report/final_report.pdf). 

**This repo is a public facing showcase of our work**. It contains all the same code, notebooks, and reports, but the data supplied in the `data` folder has been randomized to ensure our partner's data stays confidential. Thus re-running the workflows will yield different results, but the functionality remains intact. The private repository with the most up to date work on the real data can be found in [SLIPP-isotopes/MDS_Antarctica](https://github.com/ajorsi/MDS_Antarctica).

## Sections

1. [About this repository](#about-this-repository)
2. [Usage](#usage)
   - [Prerequisites](#0-prerequisites)
   - [Installation](#1-installation)
   - [Running Locally](#2-running-locally) 
3. [License](#license)
4. [Acknowledgements](#acknowledgements)
5. [Authors and Contact](#authors-and-contact)
6. [References](#references)

## About this repository 

The goal of this repository is showcase our work as part of the requirements of the MDS Capstone 2023. It is a copy of the private "production" repo as of June 28, 2023 with sensitive data removed. Before diving into the code, we recommend gaining an understanding of the background and methods used through the final report and Methods.md. Then, we recommend following the workflow to reproduce the final results through the README.md document. The README.md will guide you in running the demo notebooks and command line scripts. 

_**Background information**_

#### 1. Final report: [docs/final_report/final_report.pdf](./docs/final_report/final_report.pdf)

The final report was a deliverable submitted to the Masters of Data Science program (MDS) in fulfillment of the 2023 capstone project. The report contains background about the project, methods, results, discussions, and recommended future steps. 

#### 2. Methods.md: [docs/Methods.md](./docs/Methods.md)

The Methods.md document supplements and extends the "Data Science Methods" section of the final report. It is a good place to get more information about the non-coding-related rationale behind each of the steps in the project. It contains theoretical explanations for the Gaussian Process and Neural Network machine learning models used, descriptions of model parameters that were tried along the course of the project, and the final models and outputs generated. 

_**Reproducing the workflow**_

#### 1. README.md: [README.md](./README.md)

The README.md file (the document you are currently reading), serves as the overview to the entire project and is where you should start when reproducing the workflow. Importantly, it contains installation instructions, instructions on how to run the notebooks, and how to run the scripts using the command line. 

#### 2. Notebook demos: [notebooks/](./notebooks)

In contrast to the Methods.md document, the notebooks give the coding-related rationales behind the implementation of the project. The notebooks demonstrate how to use the code stored in `MDS_Antarctica/src/`, and create outputs used in the final report. The notebooks should be run in the following order: 
- [notebooks/preprocessing_demo.ipynb](./notebooks/preprocessing_demo.ipynb)
- [notebooks/baselines_demo.ipynb](./notebooks/baselines_demo.ipynb)
- [notebooks/gp_demo.ipynb](./notebooks/gp_demo.ipynb)
- [notebooks/nn_workflow_demo.ipynb](./notebooks/nn_workflow_demo.ipynb)
- [notebooks/postprocessing_evaluation_demo.ipynb](./notebooks/postprocessing_evaluation_demo.ipynb)

The last notebook, [notebooks/build_dummy_dataset.ipynb](./notebooks/build_dummy_dataset.ipynb), is not part of the workflow of the project. Its purpose is to create fake data in the structure of the IsoGSM dataset to use when sharing our code with the public. 

#### 3. Scripts and guides: [src/](./src/), [docs/guides/](./docs/guides/)

All of the code for this project was written in Python scripts and stored under `src/`. Code was modularized into folders for each of the important sections of the project: 
- [src/Preprocessing](./src/Preprocessing/) 
- [src/Baselines](./src/Baselines/) 
- [src/GaussianProcesses](./src/GaussianProcesses/) 
- [src/NeuralNetwork](./src/NeuralNetwork/) 
- [src/Postprocessing](./src/Postprocessing/) 

Each module has corresponding "[guides](./docs/guides/)", which outline the inputs, outputs, and purposes of the code. Additionally, there is a Sockeye guide ([sockeye.md](./docs/guides/sockeye.md)) which details how to use the cloud computing resources provided by UBC ARC.

The `src/` folder also contains the module [polar_convert](./src/polar_convert), which contains a script required for the preprocessing portion of our project. This is a copy of the [polarstereo-lonlat-convert-py](https://github.com/nsidc/polarstereo-lonlat-convert-py) repository with slight modifications. Credits to the author of this repository can be found under [Acknowledgements](#acknowledgements).

_**Other parts of this repository**_

1. Data folder: [data/](./data/)

Holds the raw and preprocessed data used in the rest of the project. The directory contains fake raw data as well as the preprocessed input files.

2. Images folder: [img/](./img/)

Contains images used in the repository across the README.md, demo notebooks, and reports. 

3. Results folder: [results/](./results)

Holds the results from the model and post-processing notebooks/scripts, including model predictions, model states, and post-processing plots.

## Usage 

### 0. Prerequisites

`MDS_antarctica` is run under `Python 3.10`, and requires the following packages and libraries. We recommend using the provided [`slipp.yml`](https://github.com/SLIPP-isotopes/MDS_Antarctica/blob/main/slipp.yml) environment file to ensure our code runs expected. 

```
  - cartopy=0.21.1
  - ipykernel=6.22.0
  - matplotlib=3.7.1
  - netcdf4=1.6.3
  - pandoc=3.1.3
  - pip=23.1.2
  - pytorch=2.0.1
  - scikit-learn=1.2.2
  - statsmodels=0.13.5
  - torchmetrics=0.11.4
  - torchvision=0.15.2
  - xarray=2023.4.2
  - gpytorch==1.10
  - linear-operator==0.4.0
  - torch-summary==1.4.5
```

### 1. Installation 

**Clone the repository and create the `slipp` environment**
```
# Clone the repository 
$ git clone git@github.com:SLIPP-isotopes/MDS_Antarctica_Showcase.git

# Move into the repository 
$ cd MDS_Antarctica_Showcase

# Install and create the virtual environment (estimated run time: 2 - 15 minutes)
$ conda env create -f slipp.yml 

# Activate the virtual environment 
$ conda activate slipp
```

**To deactivate after use**
```
$ conda deactivate 
```

### 2. Running Locally

The workflow of our project is as follows:

1. Data Preprocessing
2. Baseline Models
3. Gaussian Process Models
4. Neural Network Models
5. Results Post-processing

The steps can either be run on a Jupyter Notebook or using the command line (with the exception of Gaussian Process Models). Demo notebooks were created to give in-depth explanations of the coding implementations. It is recommended to first read/work through the notebooks, and then try the command line instructions. The command line is meant to speed up the process when trying different model parameters and architectures. Do the following each time before you run a notebook or use the command line: 

---

**Demo Notebooks**

1. Open a Jupyter Notebook editor (e.g. Jupyter Lab, VSCode)
2. Navigate to `MDS_Antarctica/notebooks/`
3. Open a demo notebook
4. Select the **Python [conda env:slipp]** kernel
5. Run the code cells

**Command line**

1. Open a command line interface (e.g. terminal)
2. Navigate to the root: `MDS_Antarctica/`
3. Activate the `slipp.yml` environment: `$ conda activate slipp`
4. Follow the commands listed below, modifying the arguments to your requirements
5. To get more information about the scripts and how to use the flags at any time, you can use the "--help" flag, i.e.: `$ python src/Preprocessing/main.py --help`; you can also read the guide for the script for information about the flags 

---

#### 1. Data Preprocessing

**Note: data preprocessing has already been run, and the preprocessed data files uploaded to data/preprocessed/ in this showcase repository.** 

This step converts the raw IsoGSM climate data into preprocessed data ready to be used in models. It also performs the train/valid/test split.

*Demo Notebook*

> [preprocessing_demo.ipynb](./notebooks/preprocessing_demo.ipynb)

*Command line*

> [Preprocessing command line guide](docs/guides/Preprocessing/main.md)

Run the script `main.py`, saving preprocessed files under `data/preprocessed/`. Estimated run time: 20 - 40 seconds.

```
$ python src/Preprocessing/main.py -o 'data/preprocessed/' -i 'data/'

# Check that the preprocessed files were generated correctly: 
$ ls data/preprocessed/

# Expected output
preprocessed_test_ds.nc    preprocessed_train_ds.nc    preprocessed_valid_ds.nc
```

#### 2. Baseline Models 

*Demo Notebook*

> [baselines_demo.ipynb](./notebooks/baselines_demo.ipynb)

*Command line*

> [Baseline Models command line guide](docs/guides/Baselines/main.md)

Run the script `bl_main.py`, specifying that the variable `hgtprs` (-v 0) should be predicted using the baseline method "ols", and that we should predict on the validation set. Prediction files will be saved under `results/predictions/baseline/`. Estimated run-time: 1 - 10 seconds.

```
$ python src/Baselines/scripts/bl_main.py -o 'results/predictions/baseline/' -d 'data/preprocessed/' -v 0 -b 'ols' -vt 'valid' -verbose 1

# Check that the prediction files were generated correctly: 
$ ls results/predictions/baseline/

# Expected output
bl_ols_hgtprs_valid_preds.nc   
```
To generate predictions for the rest of the variables, repeat the above command two more times. Each time, only modify the "-v" argument (1 for `pratesfc`, 2 for `tmp2m`). 


#### 3. Gaussian Process Models 

*Demo Notebook*

> [gp_demo.ipynb](notebooks/gp_demo.ipynb)

*Command line*

> [Gaussian Process Models command line guide](docs/guides/GaussianProcesses/main.md)

The following line calls the `gp_main.py` script, specifying the output and data directories on Sockeye. It creates a GP model to predict the variable `hgtprs` (-v 0) using kernel configuration 1 (-k 1). It creates **100 splits** of the training data (-n 100) and fits the model on split 0 (-s 0). Finally, it sets the number of epochs to 10 (-e 10), the learning rate to 0.0015 (-l 0.0015), and the mean component to be linear (-m 'linear'). 

Estimated run-time: 30 - 60 seconds.

```
$ python src/GaussianProcesses/scripts/gp_main.py -o 'results/predictions/gp/' -d 'data/preprocessed/' -v 0 -k 1 -n 100 -s 0 -e 10 -l 0.0015 -m 'linear'

Check that the prediction files were generated correctly:
$ ls results/predictions/gp/

# Expected output 
gp_hgtprs_k1_m1_n100_s0_e10_l0.0015_cpu_model_state.pth    gp_hgtprs_k1_m1_n100_s0_e10_l0.0015_cpu_valid_preds.nc
``` 

**⚠️ NOTE: THIS IS NOT THE FULL TRAINING OF THE GP MODEL ⚠️**  

As seen above, the number of splits of the training data was 100 (-n 100). This creates a dataset of about 7,000 data points, which is far too little for the actual training of a GP model. Thus, the above line of code is for demonstration purposes only, to show how the command is run locally. In our final results, we found that number of splits = 9, or about 87,000 data points per split, was ideal. However, this would be infeasible to run locally. Thus, we require require large computational resources. 

#### 4. Neural Network Learning Models 

*Demo Notebook*

> [nn_workflow_demo.ipynb](notebooks/nn_workflow_demo.ipynb)

*Command line*

> [Neural Network Models command line guide](docs/guides/NeuralNetworks/main.md)

Run the script `nn_main.py`, specifying a "CNN-deep2" architecture (-a), 200 epochs (-e), and a seed of 123 (-s). Prediction files will be saved under `results/predictions/nn/`. Estimated run-time: 30 minutes.

```
$ python src/NeuralNetwork/scripts/nn_main.py -o 'results/predictions/nn/' -d 'data/preprocessed/' -a "CNN-deep2" -e 200 -s 123

# Check that the prediction files were generated correctly
$ ls results/predictions/nn/

# Expected output 
nn_CNN-deep2_e200_l0.15_s123_model_state.pt             nn_CNN-deep2_e200_l0.15_s123_training_progress_plot.png
nn_CNN-deep2_e200_l0.15_s123_train_preds.nc             nn_CNN-deep2_e200_l0.15_s123_valid_preds.nc
```

#### 5. Results Post-processing 

*Demo Notebook*

> [postprocessing_evaluation_demo.ipynb](notebooks/postprocessing_evaluation_demo.ipynb)

*Command line*

> [Results Post-processing command line guide](docs/guides/Postprocessing/main.md)

The following line calls the `main.py` script, ... ??? 

Estimated run-time: 10 - 30 seconds 

```
$ python src/Postprocessing/main.py -d 'results/predictions/nn' -o 'results/postprocessing/' -v 2 -k 0 1 -n 10 -sgp 1 -egp 7 -lgp 0.001 -m 1 -pu 'cpu' -a 'CNN-deep2' 'Linear-narrow' -enn 6 -lnn 0.02 -snn 1234

# Check that the outputs were generated correctly
$ ls results/postprocessing/

# Expected output 
???
```

## License

Our work is distributed under the [MIT License](https://github.com/SLIPP-isotopes/MDS_Antarctica/blob/main/LICENSE).

## Acknowledgements

Partner:

Our thanks to Anais Orsi, PhD, of the Earth Ocean and Atmospheric Science department (EOAS) at the University of British Columbia for proposing the project, and offering continued guidance, and providing insightful feedback throughout the duration of the project.

Mentor: 

Our thanks to Alexi Rodriguez-Arelis, PhD, of the Statistics department at the University of British Columbia for his continued feedback and guidance throughout the project. 

Sockeye: 

This research was supported in part through computational resources and services provided by [Advanced Research Computing](https://arc.ubc.ca/ubc-arc-sockeye) at the University of British Columbia. 

Climate models: 

Data from the IsoGSM climate model was used in this project. Its use was first described in the following paper: ["Historical isotope simulation using Reanalysis atmospheric data"](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2008JD010074) (Yoshimura et al. 2008).

Polar_convert: 

The `polar_convert.py` script and functions used during preprocessing were taken from the [polarstereo-lonlat-convert-py](https://github.com/nsidc/polarstereo-lonlat-convert-py) repository. Our thanks to Chris Torrence of the NASA National Snow and Ice Data Center Distributed Active Archive Center for the opportunity to use this code. 

## Authors and Contact 

Daniel Cairns, Jakob Thoms, and Shirley Zhang contributed to MDS_Antarctica between May - June 2023 as part of their capstone project "Data Science for polar ice core climate reconstructions". The project was in fulfillment of the requirements for the Masters of Data Science Program at the University of British Columbia. 

## References 

Yoshimura, K, M Kanamitsu, D Noone, and T Oki. 2008. “Historical Isotope Simulation Using Reanalysis Atmospheric Data.” Journal of Geophysical Research: Atmospheres 113 (D19).22
