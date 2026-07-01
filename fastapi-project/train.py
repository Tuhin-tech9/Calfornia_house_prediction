from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,accuracy_score
import pandas as pd
import joblib as jb


data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df1=data.target
x_train,x_test,y_train,y_tes=train_test_split(
    df,df1,test_size=0.2,random_state=42
)
## n_estimate means decision tree making accurate result but training slow
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mae=mean_absolute_error(y_tes,y_pred)
r2=r2_score(y_tes,y_pred)
# accuracy=accuracy_score(y_tes,y_pred)
print(f"average error ${mae *100000:,.0f}")
print(f"{r2:.2f}")
## joblib is use for model save and load the model
jb.dump(model,"houseprice.joblib")
print("model successfully load")