# Prediction of Used Motor Vehicle Prices

## Table of Contents
- [Overview](#overview)
- [Screencast](#screen-recording)
- [Process Flow](#process-flow-diagram)
- [Project Set Up and Installation](#project-set-up-and-installation)
- [Dataset](#dataset)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Automated ML](#automated-ml)
- [Model Deployment](#model-deployment)
- [Standout Suggestions](#standout-suggestions)
- [Improving the Project in the Future](#improving-the-project-in-the-future)


## Overview
[Top](#table-of-contents)

In this project I use Azure’s Machine Learning services for the development (and deployment) of a model that can predict the prices of used motor vehicles in Canada (regression). This is based on the characteristics of the motor vehicles, their sellers and their locations.

The project’s activities are performed in four phases, including:
1.	Identification, exploration and use of a suitable dataset (external to Azure) of interest
2.	Use of Azure’s AutoML tool to automatically perform training experiments and produce an optimal model
3.	Use of Azure’s HyperDrive tool to perform hyperparameter tuning experiments and produce an optimal model
4.	Deployment of the most optimal model (considering both experiments above) for consumption as an inference web service.

## Screen Recording
[Top](#table-of-contents)

[![here](https://img.youtube.com/vi/2RdAcl6C6bg/mqdefault.jpg)](https://youtu.be/2RdAcl6C6bg)

## Process Flow Diagram
[Top](#table-of-contents)

![Process Diagram](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCAP_ML_Architecture.png)

## Project Set Up and Installation
[Top](#table-of-contents)

### Packages
This is a general AzureML project and uses several different Azure and Azure ML SDK libraries, along with basic data processing libraries such as Pandas and Numpy and general Machine Learning libraries such as SciKitLearn.

That said, the data is being retrieved from Kaggle's API and requires use of the [opendatasets](https://pypi.org/project/opendatasets/) library, which can be used as follows:

```python
!pip install opendatasets
import opendatasets as od
dataset_url = 'https://www.kaggle.com/user/dataset'
od.download(dataset_url)
```
### Local setup for work and troubleshooting
For the purpose of testing or troubleshooting, one may consider the following:
- Local install of Jupyter Notebook and/or Python - test training HyperDrive script and general development (without a time limitation)
- Local install of [AzureML Inference Server tool](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-inference-server-http) - test web service entry/scoring script. 
- Local install of Docker desktop - test web service container

### Compute
For the purpose of running training activities in parallel, a compute cluster is configured with 3 nodes minimum and 6 nodes maximum with 10 minute idle time for scale down.

![Compute](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_03.png)


## Dataset
[Top](#table-of-contents)

### Overview
To get things started, several datasets on Kaggle were reviewed for interest, usability, data quality and suitability in relation to the ultimate purpose. With that said, I settled on a [used car listings dataset](https://www.kaggle.com/rupeshraundal/marketcheck-automotive-data-us-canada) (credit: [MarketCheck.com](https://www.marketcheck.com/)) which provides data for cars being sold on the US and Canadian markets. This data is scraped from 65k used car sales websites across North America. Given my location and personal interest, I chose to use the Canadian portion of that dataset. 

### Task
The car sales data will be used to try and predict the prices of used cars in Canada, by specifying some key attributes about the desired cars (such as their year, make, model, miles, trim, engine, vehicle type and fuel type), their seller and their location. That being said, for the AutoML experiment, the raw dataset was used for training, so as to avoid tainting the full potential of the AutoML toolset. For the HyperDrive experiment, several features were deemed unnecessary (e.g. VIN #, Stock #) and were removed. 

### Access
Using the Kaggle API, the data was downloaded locally using the opendatasets library which only requires a Kaggle username and key for authentication. Using the Azure SDK, it was then uploaded to the default datastore and converted to a tabular dataset which was then registered in the workspace. This allowed the data to be accessible from all experiments.

Print out of the data being downloaded in the notebook
![Accessing the Data](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_01.png)

The registered dataset in AzureML Studio
![The Registered Dataset](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_02.png)


## Hyperparameter Tuning
[Top](#table-of-contents)

As part of the HyperDrive experiment’s training script, the data was cleaned to some extent. It was also encoded, split and standardized, for use in training. This preparation exercise was informed by review of the raw data features and their correlations, using a correlation map. 

<img src="https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_45.jpg" alt="Correlation Map" width="75%" height="75%">

Following research into suitable models for price prediction (regression), several experiments were performed on the preprocessed data locally using various shallow models. Given the outcome of those experiments and the obvious tradeoff between power/cost and efficiency, a RandomForestRegressor was chosen for the HyperDrive experiment. 

Additional research into the most important parameters for that model (and local experiment results) led to the use of the ‘Number of Estimators’, ‘Max Depth’, ‘Criterion’ and ‘Max Features’ for hyperparameter tuning. Bayesian sampling was used for its optimized method of finding increasingly better combinations of parameters.

As it were, the above-mentioned parameters lend themselves only to simple choice-based ranges which were specified as follows:
- Number of estimators: 10, 100
- Max Depth: 2, 10, 30
- Criterion : mse, mae - mean squared or absolute error
- Max Features: sqrt, log2, auto (=n_features)

The specifications for the numeric parameters were deliberately set to span the extremes. The Max Features parameter could likely have been specified as a uniform range between floats, but that would have lost the default options (sqrt, log2, auto).

### Results
From 24 iterations, the HyperDrive experiment's best model yielded a ~96% R2 score. Its notable parameters were as follows:
Number estimators: 100
Max Depth: 30
Criterion: mse
Max Features: log2

The RunWidget below shows the HyperDrive experiment in its final stages, as noted by the logs in the top right corner. The table at the bottom is sorted to show the best metrics produced along with some of the parameters used.
![HyperDrive RunWidget](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_07.png)

The experiment's view in ML Studio shows a better list of the runs with their metrics and the parameters used.
![HyperDrive RunWidget](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_11.png)

The completed experiment showing the range of parameters used for Bayesian optimization
![HyperDrive RunWidget](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_12.png)

A graphical representation portraying the parameter settings in use along with the produced R2 score. This provides a good idea of the more useful hyperparameter settings, such as using mse with higher max depth.
![Parallel Coordinate](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_10.png)

The Best Run was retrieved from the HyperDrive experiment and used to display the associated (best) model's parameters and other properties
![Getting the Best Run](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_13.png)
![Best Run Parameters](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_14.png)

The best model is downloaded and the Model's details are printed

![Best Model Download and Model Properties](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_15.png)

### Future Improvements

Without a doubt, the model could have been improved with more in-depth data preparation. Additionally, more rigorous hyperparameter tuning could have surfaced better models. The latter point was certainly a possibility but the observed training times were a prohibiting factor given the lab's time limitation. A number of iterations (especially with estimators=100, max depth=30 and high # of max features) had to be manually cancelled due to lengthy training.


## Automated ML
[Top](#table-of-contents)

The AutoML Experiment was configured to run a regression experiment on the raw data where the price column is to be predicted based on the other columns. Featurization was enabled, allowing AutoML to autonomously perform data preparation steps including developing its own features for use along with the raw features.

For timing, the experiment was set to run with 5 concurrent iteration runs for 15* minutes overall with an individual iteration timeout of 10 minutes and with early stopping enabled. The early stopping iterations was limited to 5 to stop the experiment if only 5 iterations failed to improve on the score. This was done bearing in mind that AzureML's early stopping feature starts calculating after the first 20 iterations by default. Therefore, 25 experiments is the earliest time at which the experiment would be terminated early, albeit with 2 additional ensemble iterations being subsequently added.

5-fold cross-validation was used to evaluate the results of the iterations and model explainability was enabled for the best model. This was used to get a better understanding of the more important features, whether raw or engineered.

*In reality, the experiment was previously run for an hour a few times, which was later deemed unnecessary (The HyperDrive experiment always yielded better results).

### Results
The AutoML experiment completed around 11 iterations with the best, a stack ensemble, topping out at ~92.6% R2 score. This ensemble was comprised of a XGBoostRegressor that was preprocessed using a standard scaler and a LightGBMRegressor that was preprocessed using a max absolute scaler. 

For the XGBoost algorithm, the parameters included:
- Number of estimators: 100
- Max depth: 9

For the LightGBM algorithm, the only noted parameters included:
- Minimum data in leaf: 20
- Number of jobs (threads): 1

ElasticNetCV was used as the meta-learner for the ensemble. Some of its parameters included:
- Fit Intercept: True
- L1 ration: 0.5
- Max iterations: 1000

NB: The automl widget screenshots below are from a run previous to the one in the notebook, as the runwidget screenshot wasn't captured during that last experiment run. The parameters and results are the same.
The RunWidget below shows the AutoML experiment upon completion. The table at the bottom is sorted to show the best metrics produced along with some of the parameters used.

![AutoML RunWidget](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_29_prevrun.png)

This shows comparative R2 scores of the different runs
![AutoML RunWidget](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_30_prevrun.png)

The screenshots below show a Notebook printout of the details of the best run from the AutoML Experiment

![Best Run params 1](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_31_prevrun.png)
![Best Run params 2](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_32_prevrun.png)
![Best Run params 3](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_33_prevrun.png)
![Best Run params 4](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_34_prevrun.png)
![Best Run params 5](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_35_prevrun.png)

A printout of the explanations of feature importance, showing the most useful raw and engineered features

![Model Explanation](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_39.png)

The model was then downloaded

![Model Download](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_40.png)

### Future Improvements

In general, XGBoost is one of the better performing algorithms available for regression with this kind of data. For the sake of improvement, we could possibly run a much longer experiment, allowing AutoML to engage in more rigorous hyperparameter tuning. That said, it is probably safer to assume that manually cleaning and feature selecting/engineering the data would have produced a much better result.




## Model Deployment
[Top](#table-of-contents)

### Metric
The models from both experiments were evaluated based on R2 score, which was chosen as the primary metric after analysis of the spread of target variable values in the dataset. As noted by Microsoft’s [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#metrics-for-regression-scenarios):
> Metrics like r2_score and spearman_correlation can better represent the quality of model when the scale of the value-to-predict covers many orders of magnitude. For instance salary estimation, where many people have a salary of $20k to $100k, but the scale goes very high with some salaries in the $100M range.

As such, the R2 score metric is applicable given the spread of car prices in the dataset. If the spread of car prices did not fit that scenario, normalized_root_mean_squared_error would have been used instead.

### Registering the Model
Based on the scoring metrics of the AutoML and HyperDrive experiments, the best model was taken from the latter experiment for deployment as a web service. The model was then registered.

![Register Model](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_16.png)

The registered model in AzureML Studio

![Registered Model](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_17.png)

The model file stored as an artifact

![Model File Artifact](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_18.png)

### Setup
As the deployed model will run from within a container that operates as a web server, we must provide:
- An entry script that will receive (and preprocess) inputs from the endpoint, use our model to make a prediction and return outputs
- The set of software packages (conda or pip based) that will need to be installed in order for the script to run
- Information about the deployment including compute and security requirements

In this scenario, the software package dependencies included python, scikitlearn, pandas, joblib and azureml defaults (responsible for the web server). The  deployment was made using an Azure Container Instance with 1 cpu core and 3gb of memory **. 

Finally, authentication was enabled, thereby requiring a key for interaction with the web service.

** While 1gb of memory is usually sufficient in most cases, the size of the fitted model's joblib file was over 1gb in size. Specifying 1gb memory for the container caused the container to continually reboot (due to low memory), resulting in failed deployments and otherwise, an unhealthy web service. If a web service's deployment logs fail to provide useful or timely information (e.g. while awaiting container start), a deployment can be troubleshooted locally or via the Container instances section of the Azure portal, where one can review the logs associated with the container. Observing multiple instances of text similar to "Worker with pid XXX was terminated due to signal 9" suggests an out-of-memory problem, which manifests as continous reboots of the container.

Checking the state of the deployment, retrieving the urls and keys and enabling app insight
![Service State](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_23.png)

### Usage

The following is an example of 2 inputs that would be posted simultaneously to the scoring URI. In this scenario, we would like to predict the price of a 2018 Ford F-150 and a 2018 Toyota Corolla.

The specified options are a stripped down set of the original features from the dataset. Once received by the entry script, some of these features (categorical options) would be encoded just before feeding to the model for prediction. Both the input and the output would then be logged by the entry script, just before returning the output to the user.

```python
data = {
    "data":
    [
        {
            'miles': "51000",
            'year': "2018",
            'make': "Ford",
            'model': "F-150",
            'engine_size': "5",
            'body_type': "Pickup",
            'vehicle_type': "Truck",
            'drivetrain': "4WD",
            'transmission': "Automatic",
            'fuel_type': "Gas",
            'state': "ON"
        }
    ,
        {
            'miles': "58900",
            'year': "2018",
            'make': "Toyota",
            'model': "Corolla",
            'engine_size': "1.8",
            'body_type': "Sedan",
            'vehicle_type': "Car",
            'drivetrain': "FWD",
            'transmission': "Automatic",
            'fuel_type': "Gas",
            'state': "AB"
        }       
    ]
}
# Convert to JSON string
input_data = json.dumps(data)
# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.text)
```


Retrieving Service Logs

![Service Logs](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_24.png)


Testing the Endpoint with sample data

![Service Test](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_25.png)

These 2 screenshots were captured from the Screencast as the latest Notebook may have been lost.

Service Logs with logged input and output

![Service Logs after Test](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_46.png)

Deleting the Service 

![Service Logs after Test](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_47.png)


## Standout Suggestions
[Top](#table-of-contents)

App Insights was enabled for the deployed web service. This allows the administrator to monitor the web service's performance, its availability and any other information that is desired surrounding the use of the service. 

In this instance, the entry script was modified to log relevant information upon each request, such as the inputs to the model and its outputs. Doing so provides the administrator with visibility on the model's behavior. 

While it was only monitored by pulling the logs directly in python, Azure's AppInsights interface allows us to review the performance, failures, availability and other issues from a graphical user interface. That interface also allows us to [directly monitor](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-enable-app-insights#view-logs-in-the-studio) any elements logged in the entry script via the tool's traces table under the logs option. 

Overview page of the App Insights User Interface
![App Insight Overview](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_26.png)

Performance page of the App Insights Tool after a few tests
![App Insights Performance Page](https://github.com/dwittaker/nd00333-capstone/blob/main/images/PCap_Img_27.png)


## Improving the Project in the Future
[Top](#table-of-contents)

As mentioned above, there are a few options to consider:
- Improving the data: further cleaning and feature selection/engineering
- Experimenting with other score metrics, or a combination, if possible: root mean square for comparison with r2 score
- Trying a wider range of hyperparameters: A uniform range on max features, poisson criterion, deeper forests
- Try other algorithms: XGBoost in HyperDrive (due to its boosting functionality) or use of Deep Learning models in either AutoML or HyperDrive


