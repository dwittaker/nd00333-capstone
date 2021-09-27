import os
import json
import pandas as pd
import joblib
from azureml.core.model import Model

dict_colmap = {}

def init():
    print("This is init")
    global model
#     model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'RF-AutoPrices_hdrive_model.joblib' #'model.pt')
    model_root = Model.get_model_path('RF-AutoPrices')
    model = joblib.load(os.path.join(model_root, 'RF-AutoPrices_hdrive_model.joblib'))
    global dict_colmap
    dict_colmap = json.loads("dict_colmap.txt")
    
    
def run(data):
#     input_data = json.loads(data)
    input_data = pd.read_json(data['data'])
#     lst_colsto
# https://thispointer.com/pandas-apply-a-function-to-single-or-selected-columns-or-rows-in-dataframe/
    X_test = input_data.apply(lambda x: dict_colmap[x.name][x] if x.name in dict_colmap.keys() else x)
    
#     loaded_model = joblib.load(filename)
    rslt = model.predict(X_test)
    return rslt