import numpy as np
import pandas as pd
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









data=pd.read_csv("Delhi_v2.csv")
print(data['area'].describe())

 
data["area_cat"]=pd.cut(data['area'],bins=[0,1000,2000,3000,4000,np.inf],labels=[1,2,3,4,5])
print(data["area_cat"])

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(data,data["area_cat"]):
    train_set=data.loc[train_index].drop("area_cat",axis=1)
    test_set=data.loc[test_index].drop("area_cat",axis=1)

housing=train_set.copy()

housing_feature=housing.drop('price',axis=1)
housing_label=housing['price'].copy()
    

categorical_column=['Address','Status','neworold','Furnished_status','Landmarks','type_of_building','desc']
numerical_attributes=housing_feature.drop(columns=categorical_column,axis=1).columns.to_list()
categorical_atributes=['Address','Status','neworold','Furnished_status','Landmarks','type_of_building','desc']

num_pipeline=Pipeline([("Imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
cat_pipeline=Pipeline([("encoder",OneHotEncoder(handle_unknown="ignore"))])
full_pipeline=ColumnTransformer([("num",num_pipeline,numerical_attributes),("cat",cat_pipeline,categorical_atributes)])


housing_final=full_pipeline.fit_transform(housing_feature)

#model training

lin_reg=LinearRegression()
lin_reg.fit(housing_final,housing_label)
lin_pred=lin_reg.predict(housing_final)
lin_rmse=-cross_val_score(lin_reg,housing_final,housing_label,scoring="neg_root_mean_squared_error",cv=10)
print("Linear regression\n",pd.Series(lin_rmse).describe())


Rf_reg=RandomForestRegressor()
Rf_reg.fit(housing_final,housing_label)
Rf_pred=Rf_reg.predict(housing_final)
Rf_rmse=-cross_val_score(Rf_reg,housing_final,housing_label,scoring="neg_root_mean_squared_error",cv=10)
print("Random forest refression\n",pd.Series(Rf_rmse).describe())

dt_reg=DecisionTreeRegressor()
dt_reg.fit(housing_final,housing_label)
dt_pred=dt_reg.predict(housing_final)
dt_rmse=-cross_val_score(dt_reg,housing_final,housing_label,scoring="neg_root_mean_squared_error",cv=10)
print("Decision Tree\n",pd.Series(dt_rmse).describe())