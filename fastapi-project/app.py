import streamlit as st
import requests
import pandas as pd

API_URL_SINGLE = "http://127.0.0.1:8000/predict"
API_URL_FILE = "http://127.0.0.1:8000//predict-file"

st.set_page_config(page_title="calfornia hosuse prediction",layout="centered")
st.title("Calfornia house Prediction")
tab1,tab2=st.tabs(["Singel prediction" ,"Batch prediction"])
with tab1:
    st.subheader("Enter the details")

    col1,col2=st.columns(2)
    with col1:
        med_inc = st.number_input("Median Income (in $10,000s)", min_value=0.1, value=3.5, step=0.1)
        house_age = st.number_input("House Age (Years)", min_value=1.0, value=25.0, step=1.0)
        ave_rooms = st.number_input("Average Rooms", min_value=1.0, value=5.0, step=0.1)
        ave_bedrms = st.number_input("Average Bedrooms", min_value=1.0, value=1.0, step=0.1)
    with col2:
        population = st.number_input("Population", min_value=1.0, value=1000.0, step=10.0)
        ave_occup = st.number_input("Average Occupancy", min_value=1.0, value=3.0, step=0.1)
        latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=35.5, step=0.01)
        longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-119.5, step=0.01)
    if st.button("Predict price",type="primary"):
         payload = {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude
        }
         with st.spinner("Calculating.."):
             ## response is call to backend json pai maina payload bheja hai
             ## response =requests("url",json="dataset")
             try:
                 response=requests.post(API_URL_SINGLE,json=payload)
                 if response.status_code==200:
                     result=response.json()
                     prediction_text=result[0] if isinstance(result,list) else str(result)
                     st.success(f"{prediction_text}")
                 else:
                     st.error(f"{response.status_code}:{response.json().get("detail")}")
             except Exception as e:
                 st.error(f"str{e}")
with tab2:
    st.subheader("Upload the csv file")
    st.info("Note:csv file must be exist of this column : Medinc")

    upload_file=st.file_uploader("choose csv file",type=["csv"])
    if upload_file is not None:
       
    
        preview_df=pd.read_csv(upload_file)
        st.write(f"preview of files",preview_df.head())
        upload_file.seek(0)  ## reset file ponter after the reading file
        if st.button("predict the file",type="primary"):
            with st.spinner("waiting"):
                try:
                    ##prepare the file upload to backend
                    files={"file":(upload_file.name,upload_file.getvalue(),"text/csv")}
                    response=requests.post(API_URL_FILE,files=files)
                    

                    if response.status_code==200:
                        st.success("prediction_complete")
                        st.download_button(
                            label="download predicted csv",
                            data=response.content,
                            file_name="prediction.csv",
                            mime="text/csv"

                        )
                    else:
                        st.error(f"error:{response.status_code} : {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"{str(e)}")

