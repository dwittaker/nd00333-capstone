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
The AutoML Experiment was configured to run a regression experiment on the raw data with featurization enabled. Featurization allows AutoML to autonomously perform data preparation steps and develop its own features for use along with the raw features.

For timing, the experiment was set to run with 5 concurrent iteration runs for 15* minutes overall with an individual iteration timeout of 10 minutes and with early stopping enabled. The early stopping iterations was limited to 5 to stop the experiment if only 5 iterations failed to improve on the score. This was done bearing in mind that AzureML's early stopping feature starts calculating after the first 20 iterations by default. Therefore, 25 experiments is the earliest time at which the experiment would be terminated early, albeit with 2 additional ensemble iterations being subsequently added.

5 fold cross validation was used to evaluate the results of the iterations and model explainability was enabled for the best model. This was used to get a better understanding of the important raw and engineered features.

* In reality, the experiment was previously run for an hour a few times, which was later deemed unnecessary (The HyperDrive experiment always yielded better results).

Further, it was only allowed to use shallow models (non-deep-learning). 
automl_settings = {
    "experiment_timeout_hours": 0.25,
    "iteration_timeout_minutes":10,
    "enable_early_stopping": True,
    "early_stopping_n_iters": 5,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'r2_score',
    "verbosity": logging.INFO,
    "n_cross_validations": 5,
    "featurization":'auto',
    "model_explainability":True

automl_config = AutoMLConfig(
    compute_target=compute_target,
    task = "regression",
    training_data=dataset,
    label_column_name="price",  
    debug_log = "automl_errors.log",
    **automl_settings


### Results
It eventually topped out at ~92% R2 score.
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

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
