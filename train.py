import argparse
import os
import numpy as np
import joblib
import pandas as pd

from sklearn.metrics import r2_score
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


from azureml.core.run import Dataset, Run
import pprint
pp = pprint.PrettyPrinter(indent=4)


parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees in the forest")
parser.add_argument('--max_Depth', type=int, default=6, help="Maximum Depth of the tree")
parser.add_argument('--criterion', type=str, default="squared_error", help="Function used to measure the split quality")
parser.add_argument('--max_Features', type=str, default="auto", help="Number of features to consider when seeking best split")
# parser.add_argument('--data_file_path', type=str, help="Filepath for Data")
parser.add_argument("--input_data", type=str)
parser.add_argument('--output_dir', type=str, help="Folder to Dump Model")


args = parser.parse_args()

# file_data = "Data/ca-dealers-used.csv"
# ds = pd.read_csv(args.data_file_path)
ws = run.experiment.workspace
# get the input dataset by ID
dataset = Dataset.get_by_name(ws, name=args.input_data)

# load the TabularDataset to pandas DataFrame
ds = dataset.to_pandas_dataframe()


def clean_data(df_data):
    #https://pbpython.com/categorical-encoding.html
    lst_removemake = ['smart']
    df_data.dropna(inplace=True)
    df_data = df_data[(~df_data['price'].isnull()) & (df_data['price']>1)]
    df_data = df_data[~df_data['make'].isin(lst_removemake)]

    df_data.loc[((df_data['fuel_type'].str.contains('Electric /')) | df_data['fuel_type'].str.contains('/ Electric')),'fuel_type'] = 'Hybrid'
    df_data.loc[df_data['fuel_type'].str.contains('Compressed Natural Gas'),'fuel_type'] = 'Hybrid'
    df_data.loc[((df_data['fuel_type'].str.contains('/')) & ~(df_data['fuel_type'].str.contains('Electric'))),'fuel_type'] = 'Gas'
    df_data.loc[df_data['fuel_type'].isin(['Biodiesel','Diesel']),'fuel_type'] = 'Diesel'
    df_data.loc[df_data['fuel_type'].isin(['E85','Unleaded','Premium Unleaded','Premium Unleaded; Unleaded']),'fuel_type'] = 'Gas'

    df_data.drop(columns=['id','vin','stock_no','trim','seller_name','street','city','zip'],inplace=True)
    dict_colmap = {}
    lst_objcolumns = list(df_data.select_dtypes(include=['object']).columns)
    
    
    df_cleaned = df_data.copy()
    df_cleaned[lst_objcolumns] = df_cleaned[lst_objcolumns].astype('category')
    for col in lst_objcolumns:
        df_cleaned[col+"_cat"] = df_cleaned[col].cat.codes
        dict_colmap[col] = pd.Series(df_cleaned[col+"_cat"].values.tolist(), index=df_cleaned[col].values.tolist()).to_dict()
    df_cleaned.drop(columns=lst_objcolumns, inplace=True)


    df_price = df_cleaned.pop("price")
    
#     https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-scikit-learn
#     https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py
    return df_cleaned, df_price, dict_colmap

x, y, dct_cols = clean_data(ds)
with open('dict_colmap.txt', 'w') as outfile:
    json.dump(dct_cols, outfile)

correlation_map(x,y)


x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.95, test_size = 0.05, random_state=1)

run = Run.get_context()

def correlation_map(x,y):
    df_corr = x.copy()
    x['price'] = y
    correlations = x.corr()

    indx=correlations.index
    plt.figure(figsize=(20,15))
    sns.heatmap(x[indx].corr(),annot=True,cmap="YlGnBu")


def main():
    

    run.log("Number of estimators", np.float(args.n_estimators))
    run.log("Max Depth", np.int(args.max_Depth))
    run.log("Criterion", args.criterion)
    run.log("Max Features", args.max_Features)

    model = RandomForestRegressor(max_depth=args.max_Depth,n_estimators=args.n_estimators, criterion=args.criterion, ,max_features = args.max_Features) # , random_state=0)
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))
    
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(value=model, filename=f'{args.output_dir}/model/RF-AutoPrices_hdrive_model.joblib')


if __name__ == '__main__':
    main()
