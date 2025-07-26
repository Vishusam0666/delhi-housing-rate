import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import os
import joblib

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

def  build_pipeline(numerical_attributes,categorical_attributes):
    num_pipeline=Pipeline([("Imputer",SimpleImputer(strategy="median")),("Scaler",StandardScaler())])
    cat_pipeline=Pipeline([("Encoder",OneHotEncoder(handle_unknown="ignore"))])
    full_pipeline=ColumnTransformer([("num",num_pipeline,numerical_attributes),("cat",cat_pipeline,categorical_attributes)])
    return full_pipeline

if  not os.path.exists(MODEL_FILE):
   data = pd.read_csv("Delhi_v2.csv")
   if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])

 
    data["area_cat"]=pd.cut(data['area'],bins=[0,1000,2000,3000,4000,np.inf],labels=[1,2,3,4,5])
    print(data["area_cat"])

    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(data,data["area_cat"]):
     train_set=data.loc[train_index].drop("area_cat",axis=1)


    housing=train_set.copy()

    housing_feature=housing.drop('price',axis=1)
    housing_label=housing['price'].copy()
    

    categorical_column=['Address','Status','neworold','Furnished_status','Landmarks','type_of_building','desc']
    numerical_attributes=housing_feature.drop(columns=categorical_column,axis=1).columns.to_list()
    categorical_attributes=['Address','Status','neworold','Furnished_status','Landmarks','type_of_building','desc']

    PIPELINE=build_pipeline(numerical_attributes,categorical_attributes)
    housing_final=PIPELINE.fit_transform(housing_feature)

    model=RandomForestRegressor(random_state=42)
    model.fit(housing_final,housing_label)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(PIPELINE,PIPELINE_FILE)

    print("model train and saved")

else:
   
   model=joblib.load(MODEL_FILE)
   pipeline=joblib.load(PIPELINE_FILE)

   input_data = pd.read_csv("input2.csv").drop(columns=["Unnamed: 0"], errors="ignore")
   transformed_input=pipeline.transform(input_data)
   prediction=model.predict(transformed_input)
   input_data['price']=prediction
   input_data.to_csv("output.csv",index=False)
   print("Inference complete result saved in output")
