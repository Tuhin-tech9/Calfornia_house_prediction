from sklearn.datasets import fetch_california_housing
import pandas as pd

data=fetch_california_housing()

##data.data means data fetch from this data set columns means raw data to making column
df=pd.DataFrame(data.data,columns=data.feature_names)
df["price"]=data.target

print(df.shape)
print(df.head(10))
print(df.columns)
print(df.isnull().sum())
print(df.describe())