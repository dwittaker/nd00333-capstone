import os
import json
import numpy as np
import pandas as pd
import joblib
from azureml.core.model import Model
# This explains the use of azure model dir vs get model path
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-advanced-entry-script
dict_colmap = {}
json_file_path = "dict_colmap.txt"
########## Need to remember that this is running on the inference remote machine. Check if the model path would be the same
def init():
    global model
    print ("model initialized" + time.strftime("%H:%M:%S"))
#     model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'RF-AutoPrices_hdrive_model.joblib') #'model.pt'
#     model = joblib.load(model_path)
    
    model_path = Model.get_model_path('RFHD-AutoPrices')
#     model = joblib.load(os.path.join(model_root, 'RF-AutoPrices_hdrive_model.joblib'))
    model = joblib.load(model_path)
    
    global dict_colmap
    with open(json_file_path, 'r') as j:
         dict_colmap = json.loads(j.read())
    
    
def run(data):
    input_json = json.loads(data)['data']
    df_data = pd.DataFrame(input_json)

    # https://thispointer.com/pandas-apply-a-function-to-single-or-selected-columns-or-rows-in-dataframe/
    X_test = df_data.apply(lambda x: x.map(dict_colmap[x.name]) if x.name in dict_colmap.keys() else x )

    rslt = model.predict(np.array(X_test.values.tolist()))
    
    # Log the input and output data to appinsights:
    info = {
        "input": raw_data,
        "output": rslt.tolist()
        }
    print(json.dumps(info))
    return rslt

# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-enable-app-insights