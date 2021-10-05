import os
import logging
import json
import numpy
import joblib
# from sklearn.externals import joblib
import azureml.train.automl
from azureml.core.model import Model
########## Need to remember that this is running on the inference remote machine. Check if the model path would be the same
def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model

    model_path = os.path.join(
    # os.getenv("AZUREML_MODEL_DIR"), "sklearn_regression_model.pkl"
    os.getenv("AZUREML_MODEL_DIR"), "model.pkl"
    )
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    #     global automl_model
    #     automl_model_path = Model.get_model_path('automl_model')
    #     automl_model = joblib.load(automl_model_path)
    

    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    try:
        logging.info("Request received")
        data = json.loads(raw_data)["data"]
        data = numpy.array(data)
        result = model.predict(data)
        logging.info("Request processed")
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error

# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability-automl
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-advanced-entry-script