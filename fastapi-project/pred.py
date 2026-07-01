import io ##like a treat your data as file
from fastapi import FastAPI,HTTPException,UploadFile,File
import joblib as jb
from pydantic import BaseModel,Field
from fastapi.responses import StreamingResponse
import numpy as np
import pandas as pd
app=FastAPI()

try:
    model=jb.load("houseprice.joblib")
except Exception:
    raise RuntimeError("model is not load")
class prediction(BaseModel):
    MedInc: float=Field(gt=0 ,description="median income value")     # Median Income
    HouseAge: float    # House Age
    AveRooms: float    # Average Rooms
    AveBedrms: float   # Average Bedrooms
    Population: float  # Population
    AveOccup: float    # Average Occupancy
    Latitude: float    # Latitude
    Longitude: float

@app.post("/predict")
def get_prediction(data:prediction):
    input=np.array([[data.MedInc,data.HouseAge,data.AveBedrms,data.AveRooms,data.Population,data.AveOccup,data.Longitude,data.Latitude]])
    prediction=model.predict(input)[0]

    if prediction<0:
        raise HTTPException(
            status_code=500,
            detail="something went wrong"
        )
    return{
        f"prediction price is ${prediction*100000:,.0f}"
    }


@app.post("/predict-file") ##async means not wait for others
async def get_predictfile(file:UploadFile=File(...)): ##file upload ... means required file is required and uploaded
       if not file.filename.endswith(".csv"):
            raise HTTPException(
                 status_code=404,
                 detail="please upload only csv file"
            )
       contents=await file.read() ## csv to bytes
       
       df=pd.read_csv(io.BytesIO(contents),sep=None,engine="python",header=0) 
       ##bytes to pnadas dataframe
       

       require_column=[
            "MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"
       ]
       if len(df.columns) == len(require_column):
            df.columns=require_column
       missing_column=[
            col for col in require_column
            if  col not in df.columns
       ]
       if missing_column:
            raise HTTPException(
                 status_code=400,
                 detail=f"The column are missing colums are {missing_column } {df.columns}"
            )
       if len(df)==0:
            raise HTTPException(
                 status_code=400,
                 detail="file is empty"
            )
       
       try:
            prediction=model.predict(df[require_column])
            
            df["predicted_price"]=pd.Series(prediction).apply(lambda x: f"${x:,.0f}")
            output=df.to_csv(index=False)
            return StreamingResponse(  ## streamingresponse fastapi object file data to user
                 io.StringIO(output),
                 media_type="text/csv",
                 headers={   ##attachment downlad the file
                      "Content-Disposition":"attachment;filename=prediction_houseprice"
                 }
            )
       
       except Exception as e:
            raise HTTPException(
                 status_code=400,
                 detail=f"prediction failed :{str(e)}"
            )