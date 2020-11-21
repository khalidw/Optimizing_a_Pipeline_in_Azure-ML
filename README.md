# Optimizing an ML Pipeline in Azure

## Overview
This is first of the four projects required for fullfilment of the Nanodegree **Machine Learning Engineer with Microsoft Azure** from **Udacity**.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

You can find more information about Azure AutoML [here:](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)

## Summary
The data used in this project is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y). It consists of 20 input variables (columns) and 32,950 rows with 3,692 positive classes and 29,258 negative classes.

The data used in this project can be found [here:](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv)

Detailed description of the dataset can be found [here:](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
We use Logistric Regression algorithm from the SKLearn framework in conjuction with hyperDrive for hyperparameter tuning.

Pipeline consists of following steps:

1. Data collection
1. Data cleaning
1. Data splitting
1. Hyperparameter sampling
1. Model training
1. Model testing
1. Early stopping policy evaluation
1. Saving the model

We use a script **train.py**, to govern steps 1-3, 5, 6 and 8. Whereas step 4 and 7 is governed by hyperDrive. The overall execution of the pipeline is managed by hyperDrive. A brief description of each step is provided below.

**Data collection**

Dataset is collected from the link provided earlier, using TabularDatasetFactory.

**Data cleaning**

This process involves dropping rows with empty values and one hot encoding for categorical columns.

**Data splitting**

As a standard practice, datasets are split into train and test sets. This splitting of a dataset is helpful to validate/tune our model. For this experiment we split 70-30, 70% for training and 30% for testing.

**Hyperparameter selection**

Hyperparamters are adjustable parameters that let you control the model training process. This is a recurring step for each iteration of model training, controlled by hyperDrive.

There are two hyperparamters for this experiment, **C** and **max_iter**. **C** is the inverse regularization strength whereas **max_iter** is the maximum iteration to converge for the SKLearn Logistic Regression.

We have used random parameter sampling to sample over a discrete set of values. Random parameter sampling is great for discovery and getting hyperparameter combinations that you would not have guessed intuitively, although it often requires more time to execute.

The parameter search space used for **C** is `[1,2,3,4,5]` and for **max_iter** is `[80,100,120,150,170,200]`

**Model training**

Once we have train and test dataset available and have selected our hyperparameters for a particular iteration, we are all set for training our model. This process is also called as model fitting.

**Model testing**

Test dataset from previous split is used to test the trained model, metrics are generated and logged, these metrics are then used to benchmark the model. In our case we are using accuracy as model performance benchmark.

**Early stopping policy evaluation**

The benchmark metric from model testing is then evaluated using hyperDrive early stopping policy. Execution of the pipeline is stopped if conditions specified by the policy are met.

We have used the [BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py). This policy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. This helps to improves computational efficiency.

For this experiment the configuratin used is; `evaluation_interval=1`, `slack_factor=0.2`, and `delay_evaluation=5`. This configration means that the policy would be applied to every `1*5` iteration of the pipeline and if `1.2*`value of the benchmark metric for current iteration is smaller than the best metric value so far, the run will be cancelled.

**Saving the model**

The trained model is then saved, this is important if we want to deploy our model or use it in some other experiments.

## AutoML

AutoML uses the provided dataset to fit on a wide variety of algorithms. It supports classification, regression and time-series forecasting problems. An exit criterion is specified to stop the training which ensures that resources are not used further once the objectives are met, this saves cost also.

In our experiment we found out VotingEnsemble to be the best model based on the accuracy metric. The accuracy score for this model was `0.9167223065250379`.

A [voting ensemble](https://machinelearningmastery.com/voting-ensembles-with-python/) is an ensemble machine learning model that combines the predictions from multiple other models.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The model generated by AutoML had accuracy slighlty higher than the HyperDrive model. `0.9167223065250379` for autoML and `0.912797167425392` for HyperDrive
The architecture is different as one is VotingEnsemble where as the other is SKLearn

## Future work

**Improvements for hyperDrive**
1. Use different parameter sampling methods (Grid Parameter Sampling, Bayesian Parameter Sampling)
1. Use of different hyperparameter distributions (uniform, normal)
1. Use of different stopping policy
1. Use of different primary metric (Sometimes accuracy alone doesn't represent true picture)
1. Increasing max total runs to try a lot more combinations of hyperparameters

**Improvements for autoML**
1. Change experiment timout, would allow mfor more model experimentation
1. Use some other primary metric
1. Change number of cross validations
1. Address class imbalance, there are 3,692 positive classes whereas 29258 negative classes
