# Transfer Learning for Predictive Process Monitoring
This repository is a fork of the [original repository](https://github.com/Mathieu-van-Luijken/Transfer-Learning-for-Predictive-Process-Monitoring-)
for the paper "An Experiment on Transfer Learning for Suffix Prediction on Event Logs"
by Mathieu van Luijken, István Ketykó, Felix Mannhardt.

This repository contains the code for the experiments in the paper and additional experiments for the term paper
"Assessment and analysis of 'An Experiment on Transfer Learning for Suffix Prediction on Event Logs'". 
In addition, we provide jupyter notebooks for collecting the results of the experiments and for visualizing the results.

## Installation
The code is written in Python 3.10. You can create a conda environment with the required dependencies using the
provided `environment.yml` file. To do so, run the following command in the root directory of the repository:
```shell
conda env create -f environment.yml
```
Note: If creating the environment fails, create a new environment with Python 3.10 and install the dependencies
listed in `environment.yml` manually.

In addition, you need to install the packages listed in `requirements.txt`. To do so, run the following command:
```shell
pip install -r requirements.txt
```

Lastly, you need to set the `PYTHONPATH` environment variable to the root directory of the repository (PyCharm does
this automatically). To do so, run the following command:
```shell
export PYTHONPATH=/path/to/repository:$PYTHONPATH
```

## Usage
First you need to generate the filtered RTFMP event log. This can be done by running the `create_additional_log.ipynb`
notebook. The notebook will generate the filtered RTFMP event log by filtering out variants that only occur once in the
original RTFMP event log. The filtered RTFMP event log will be saved in the `logs` directory.

Then you can train the models by running either the `src/training_rnn.py` or the `src/training_transformer.py` script.
The scripts will train all models described in the paper and our additional models using the BPIC 2015 event log and
the filtered RTFMP event log (using only a subset of the layer freezing combinations). The checkpoints of the trained
models, information about the training process and additional information will be saved in the `results` directory.

After training the models, you can evaluate the models by running either the `src/evaluation_rnn.py` or the
`src/evaluation_GPT.py` script. The scripts will evaluate all models described in the paper and our additional
models. The results include the predictions of the models and the evaluation metrics and will be saved in the
`Predictions` directory (under the `results` directory).

To collect the results of the experiments, run the `collecting_results.ipynb` notebook. The notebook will collect the
training statistics by the datetime of the training run and transform the collected data into a csv file. 
Note that we split our experiments into multiple runs and therefore use different datetimes for collecting the results.
When rerunning the experiments, you need to change the datetimes in the notebook accordingly. The csv file will be saved 
as `all_training_results.csv` in the `results` directory. 
In addition, the notebook will collect the evaluation results and transform the collected data into a csv file. As the
evaluation results can be easily collected from the `Predictions` directory, we do not need to explicitly list the
datetimes of the evaluation runs. The csv file will be saved as `all_evaluation_results.csv` in the `results` directory.
Moreover, also the results given in the paper will be saved as `paper_results.csv` in the `results` directory for
convenience.

In the `investigating_results.ipynb` notebook, we investigate the results of the experiments and in the
`log_statistics.ipynb` notebook, we compute statistics about the event logs used in the experiments. The notebooks will
generate the figures and tables used in our term paper. The figures and tables will be saved in the `reports` directory.
The figures are saved as `.pdf` files to easily use vector graphics in the term paper (written in LaTeX) and as `.svg`
files to use them in the presentation (created in PowerPoint). The tables are directly generated as `.tex` files and 
use the `NiceTabular` environment from the `nicematrix` package to create tables with a better spacing.
Please consult either the notebooks or the term paper for more information about the figures and tables.