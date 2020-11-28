# Optimizing an ML Pipeline in Azure

## Overview
This is first of the three projects required for fullfilment of the Nanodegree **Machine Learning Engineer with Microsoft Azure** from **Udacity**.
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

In our experiment we found out VotingEnsemble to be the best model based on the accuracy metric. The accuracy score for this model was `0.9169044006069802`.

The VotingEnsemble consisted of six algorithms; the algorithms, their corresponding weightages and a few of the individual parameters including `learning_rate`, `n_estimators`, and `random_state` are summarized in the table below. Further details of each individual algorithm can be found in the corresponding Jupyter Notebook.

| Algorithm | Weightage | learning_rate | n_estimators| random_state |
| --------- | --------- | ------------- | ----------- | ------------ |
| xgboostclassifier with maxabsscaler | 0.06666666666666667 | 0.1 | 100 | 0 |
| lightgbmclassifier with maxabsscaler | 0.4666666666666667 | 0.1 | 100 | None |
| xgboostclassifier with sparsenormalizer | 0.2 | 0.1 | 25 | 0 |
| sgdclassifierwrapper with minmaxscaler | 0.06666666666666667 | constant | - | None |
| sgdclassifierwrapper with standardscalerwrapper | 0.06666666666666667 | constant | - | None |
| sgdclassifierwrapper with standardscalerwrapper | 0.13333333333333333 | balanced | - | None |

A [voting ensemble](https://machinelearningmastery.com/voting-ensembles-with-python/) is an ensemble machine learning model that combines the predictions from multiple other models.

## Pipeline comparison

The model generated by AutoML had accuracy slighlty higher than the HyperDrive model. `0.9169044006069802` for autoML and `0.912797167425392` for HyperDrive
The architecture is different as hyperDrive was restricted to Logistic Regression from SKLearn, whereas AutoML has access to wide variety of algorithms.

In some scenarios a certain model may not be suited best, hence this puts hyperDrive at a disadvantage as model slection is at the hand of the user which is not the case with AutoML. Hence the difference in accuracy is explainable.

## Future work

**Improvements for hyperDrive**
1. Use Bayesian Parameter Sampling instead of Random; Bayesian sampling tries to intelligently pick the next sample of hyperparameters, based on how the previous samples performed, such that the new sample improves the reported primary metric.
1. We could use different primary metric as sometimes accuracy alone doesn't represent true picture of the model performance.
1. Increasing max total runs to try a lot more combinations of hyperparameters, this would have an impact on cost too.

**Improvements for autoML**
1. Change experiment timeout, this would allow for more model experimentation but the longer runs may cost you more.
1. We could use different primary metric as sometimes accuracy alone doesn't represent true picture of the model performance.
1. Incresing the number of cross validations may reduce the bias in the model.
1. Address class imbalance, there are 3,692 positive classes whereas 29258 negative classes. This will reduce the model bias.
