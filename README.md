# Prediction of Used Motor Vehicle Prices

In this project I use Azure’s Machine Learning services for the development (and deployment) of a model that can predict the prices of used motor vehicles in Canada (regression). This is based on the characteristics of the motor vehicles, their sellers and their locations.

The project’s activities are performed in four phases, including:
1.	Identification, exploration and use of a suitable dataset (external to Azure) of interest
2.	Use of Azure’s AutoML tool to automatically perform training experiments and produce an optimal model
3.	Use of Azure’s HyperDrive tool to perform hyperparameter tuning experiments and produce an optimal model
4.	Deployment of the most optimal model (considering both experiments above) for consumption as an inference web service.


## Project Set Up and Installation
This is a general AzureML project and uses several different Azure and Azure ML SDK libraries, along with basic data processing libraries such as Pandas and Numpy and general Machine Learning libraries such as SciKitLearn.

That said, the data is being retrieved from Kaggle's API and requires use of the opendatasets library, which can be installed via:
!pip install opendatasets
import opendatasets as od

For the purpose of testing or troubleshooting, one may consider the following:
- Local install of Jupyter Notebook or plain python - test training HyperDrive script and general development without time limitation
- Local install of AzureML Inference Server tool - test web service entry script. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-inference-server-http
- Local install of Docker desktop - test web service container

## Dataset

### Overview
To get things started, several datasets on Kaggle were reviewed for interest, usability, data quality and suitability in relation to the ultimate purpose. With that said, I settled on a used car listings dataset (credit: MarketCheck.com) which provides data for cars being sold on the US and Canadian markets. This data is scraped from 65k used car sales websites across North America. Given my location and personal interest, I chose to use the Canadian portion of that dataset. 

### Task
The car sales data will be used to try and predict the prices of used cars in Canada, by specifying some key attributes about the desired cars (such as their year, make, model, miles, trim, engine, vehicle type and fuel type), their seller and their location. That being said, for the AutoML experiment, the raw dataset was used for training, so as to avoid tainting the full potential of the AutoML toolset. For the HyperDrive experiment, several features were deemed unnecessary (e.g. VIN #, Stock #) and were removed. 

### Access
Using the Kaggle API, the data was downloaded locally using the opendatasets library which only requires a Kaggle username and key for authentication. Using the Azure SDK, it was then uploaded to the default datastore and converted to a tabular dataset which was then registered in the workspace. This allowed the data to be accessible from all experiments.

## Automated ML
The AutoML Experiment was configured to run a regression experiment on the raw data where the price column is to be predicted based on the other columns. Featurization was enabled, allowing AutoML to autonomously perform data preparation steps including developing its own features for use along with the raw features.

For timing, the experiment was set to run with 5 concurrent iteration runs for 15* minutes overall with an individual iteration timeout of 10 minutes and with early stopping enabled. The early stopping iterations was limited to 5 to stop the experiment if only 5 iterations failed to improve on the score. This was done bearing in mind that AzureML's early stopping feature starts calculating after the first 20 iterations by default. Therefore, 25 experiments is the earliest time at which the experiment would be terminated early, albeit with 2 additional ensemble iterations being subsequently added.

5-fold cross-validation was used to evaluate the results of the iterations and model explainability was enabled for the best model. This was used to get a better understanding of the more important features, whether raw or engineered.

* In reality, the experiment was previously run for an hour a few times, which was later deemed unnecessary (The HyperDrive experiment always yielded better results).

### Results
The AutoML experimented completed around 11 iterations with the best, a stack ensemble, topping out at ~92.6% R2 score. This ensemble was comprised of a XGBoostRegressor that was preprocessed using a standard scaler and a LightGBMRegressor that was preprocessed using a max absolute scaler. 

For the XGBoost algorithm, the parameters included:
- Number of estimators: 100
- Learning rate: 0.1
- Max depth: 9
For the LightGBM algorithm, the only noted parameters included:
- Minimum data in leaf: 20
- Number of jobs (threads): 1
ElasticNetCV was used as the meta-learner for the ensemble. Some of its parameters included:
- Fit Intercept: True
- L1 ration: 0.5
- Max iterations: 1000

In general, XGBoost is one of the better performing algorithms available for regression with this kind of data. For the sake of improvement, we could possibly run a much longer experiment, allowing AutoML to engage in more rigorous hyperparameter tuning. That said, it is probably safer to assume that manually cleaning and feature selecting/engineering the data would have produced a much better result.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
Following research into suitable models for price prediction (regression), several experiments were performed locally using various shallow models. Given the obvious tradeoff between power/cost and efficiency, in light of those experiments' outcomes, a RandomForestRegressor was chosen for the HyperDrive experiment. 

Additional research into the most important parameters for that model (and local experiment results) led to the use of the ‘Number of Estimators’, ‘Max Depth’, ‘Criterion’ and ‘Max Features’ for hyperparameter tuning. Bayesian sampling was used for its optimized method of finding increasingly better combinations of parameters.

As it were, the above-mentioned parameters lend themselves only to simple choice-based ranges:
- Number of estimators: 10, 100
- Max Depth: 2, 10, 30
- Criterion : mse, mae - mean squared or absolute error
- Max Features: sqrt, log2, auto (=n_features)

The Max Features parameter could likely have been specified as a uniform range between floats, but that would have lost the default options (sqrt, log2, auto).

### Results
As part of the HyperDrive experiment’s training script, the data was cleaned to some extent. It was also encoded, split and standardized, before feeding to the model for training. This preparation exercise was informed by review of the raw data features and their correlations, using a correlation map. 

Correlation map here

From 24 iterations, the HyperDrive experiment's best model yielded a ~96% R2 score. Its notable parameters were as follows:
Number estimators: 100
Max Depth: 100
Criterion: mse
Max Features: log2

Without a doubt, the model could have been improved with more in-depth data preparation. Additionally, more rigorous hyperparameter tuning could have surfaced better models. The latter point was certainly a possibility but the observed training times were a prohibiting factor given the lab's time limitation.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
