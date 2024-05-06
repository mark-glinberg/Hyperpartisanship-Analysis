# Hyperpartisan Article Detection - Model and Dataset Analysis Codebase

## Authors

[Mark Glinberg](https://github.com/mark-glinberg)

[Stephan Tserovski](https://github.com/tserovskis)

[Justin Huang](https://github.com/justin910113)

## Description

This is the codebase for our project revolving around analyzing existing solutions for detecting hyperpartisanship in articles.

### Files

dataset_and_model.ipynb - Jupyter notebook for model loading, running, and visualizing outputs

model_explaining.ipynb - Jupyter notebook for running LIME and transformers-interpret to explain how the model makes predictions

requirements.txt - Requirements file to setup virtual Conda environment in order to run our code

Project Guidelines.pdf - Class instructions for the project

Final Report.pdf - Our generated final report of our findings

## Usage

Navigate to the project directory using your anaconda command prompt. Then run the following commands to create a proper conda environment to run this code (you may skip creating the conda environment if you already can run numpy, pandas, and matplotlib):

```bash
>> conda create -n <env_name> python=3.10

>> conda activate <env_name>
```

From here, make sure pip is installed using the following line, and afterwards install the requirements needed:

```bash
>> conda install pip

>> pip install -r requirements.txt
```

From there, you can run the Jupyter notebooks as you'd like!
