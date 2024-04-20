# UniTraj

**A Unified Framework for Cross-Dataset Generalization of Vehicle Trajectory Prediction**

UniTraj is a framework for vehicle trajectory prediction, designed by researchers from VITA lab at EPFL. 
It provides a unified interface for training and evaluating different models on multiple dataset, and supports easy configuration and logging. 
Powered by [Hydra](https://hydra.cc/docs/intro/), [Pytorch-lightinig](https://lightning.ai/docs/pytorch/stable/), and [WandB](https://wandb.ai/site), the framework is easy to configure, train and logging.
In this project, you will be using UniTraj to train and evalulate a model we call PTR (predictive transformer) on the data we have given to you.

## Installation

First start by cloning the repository:
```bash
git clone https://github.com/vita-epfl/unitraj-DLAV.git
cd unitraj-DLAV
```

Then make a virtual environment and install the required packages. 
```bash
python3 -m venv venv
source venv/bin/activate

# Install MetaDrive Simulator
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .

# Install ScenarioNet
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/scenarionet.git
cd scenarionet
pip install -e .
```

Finally, install Unitraj and login to wandb via:
```bash
cd unitraj-DLAV # Go to the folder you cloned the repo
pip install -r requirements.txt
pip install -e .
wandb login
```
If you don't have a wandb account, you can create one [here](https://wandb.ai/site). It is a free service for open-source projects and you can use it to log your experiments and compare different models easily.


You can verify the installation of UniTraj via running the training script:
```bash
python train.py method=ptr
```
The incomplete PTR model will be trained on several samples of data available in `motionnet/data_samples`.

## Code Structure
There are three main components in UniTraj: dataset, model and config.
The structure of the code is as follows:
```
motionnet
├── configs
│   ├── config.yaml
│   ├── method
│   │   ├── ptr.yaml
├── datasets
│   ├── base_dataset.py
│   ├── ptr_dataset.py
├── models
│   ├── ptr
│   ├── base_model
├── utils
```
There is a base config, dataset and model class, and each model has its own config, dataset and model class that inherit from the base class.

## Data
You can access the data [here](https://drive.google.com/file/d/1mBpTqM5e_Ct6KWQenPUvNUBJWHn3-KUX/view?usp=sharing). For easier use on SCITAS, we have also provided the dataset in scitas on `/work/vita/datasets/DLAV_unitraj`. We have provided a train and validation set, as well as three testing sets of different difficulty levels: easy, medium and hard.
You will be evaluating your model on the easy test set for the first milestone, and the medium and hard test sets for the second and third milestones, respectively. 
Don't forget to put the path to the real data in the config file.


## Your Task
Your task is to complete the PTR model and train it on the data we have provided. 
The model is a transformer-based model that takes the past trajectory of the vehicle and its surrounding agents, along with the map, and predicts the future trajectory.
![system](https://github.com/vita-epfl/unitraj-DLAV/blob/main/docs/assets/PTR.png?raw=true)
This is the architecture of the encoder part of model (where you need to implement). Supposing we are given the past t time steps for M agents and we have a feature vector of size $d_K$ for each agent at each time step, the encoder part of the model consists of the following steps:
1. Add positional encoding to the input features at the time step dimension for distinguish between different time steps.
2. Perform the temporal attention to capture the dependencies between the trajectories of each agent separately.
3. Perform the spatial attention to capture the dependencies between the different agents at the same time step.
These steps are repeated L times to capture the dependencies between the agents and the time steps.

The model is implemented in `motionnet/models/ptr/ptr_model.py` and the config is in `motionnet/configs/method/ptr.yaml`. 
Take a look at the model and the config to understand the structure of the model and the hyperparameters.

You are asked to complete three parts of the model in `motionnet/models/ptr/ptr_model.py`:
1. The `temporal_attn_fn` function that computes the attention between the past trajectory and the future trajectory.
2. The `spatial_attn_fn` function that computes the attention between different agents at the same time step.
3. The encoder part of the model in the `_forward` function. 

You can find the instructions and some hints in the file itself. 

## Submission
You could follow the steps in the [easy kaggle competition](https://www.kaggle.com/competitions/dlav-vehicle-trajectory-prediction-2024/overview) to submit your results and compare them with the other students in the leaderboard.
Here are the [medium](https://www.kaggle.com/competitions/dlav-vehicle-trajectory-prediction-medium/overview) and [hard](https://www.kaggle.com/competitions/dlav-vehicle-trajectory-prediction-hard/overview) competitions for the second and third milestones, respectively.
We have developed a submission script for your convenience. You can run the following command to generate the submission file:
```bash
python generate_predictions.py method=ptr
```
Before running the above command however, you need to put the path to the checkpoint of your trained model on the config file under `ckpt_path`. You can find the checkpoint of your trained model in the `lightning_logs` directory in the root directory of the project. 
For example, if you have trained your model for 10 epochs, you will find the checkpoint in `lightning_logs/version_0/checkpoints/epoch=10-val/brier_fde=30.93.ckpt`. You need to put the path to this file in the config file.

Additionally, for the `val_data_path` in the config file, you need to put the path to the test data you want to evaluate your model on. For the easy milestone, you can put the path to the easy test data, and for the second and third milestones, you can put the path to the medium and hard test data, respectively.

The script will generate a file called `submission.csv` in the root directory of the project. You can submit this file to the kaggle competition. As this file could be big, we suggest you to compress it before submitting it.
