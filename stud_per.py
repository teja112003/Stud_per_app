import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model():
    with open('E:\Complete Python\stud_per_app\student_lr_final_model.pkl','rb') as file:
        model,scaler,le=pickle.load(file)
    return model,scaler,le

def preprocessing_data(data,scaler,le):
    data['Extracurricular Activities']=le.transform([data['Extracurricular Activities']])[0]
    df=pd.DataFrame([data])
    df_transformed=scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le=load_model()
    processed_data=preprocessing_data(data,scaler,le)
    prediction=model.predict(processed_data)
    return prediction

def main():
    st.title('Student Performance Prediction')
    st.write('Please enter the following details to get the prediction')
    hour_studied=st.number_input('Hours studied',min_value=1,max_value=10,value=5)
    previous_score=st.number_input("Previous Score",min_value=40,max_value=100,value=70)
    extra=st.selectbox("extra curricular activities",['Yes','No'])
    sleeping_hour=st.number_input("Sleeping Hours",min_value=4,max_value=10,value=8)
    number_of_peper_solved=st.number_input("No of question papers solved",min_value=0,max_value=10,value=5)
    
    if st.button("predict your score"):
        user_data={'Hours Studied':hour_studied,
              'Previous Scores':previous_score,
              'Extracurricular Activities':extra,
              'Sleep Hours':sleeping_hour,
              'Sample Question Papers Practiced':number_of_peper_solved}
        prediction=predict_data(user_data)
        st.success(f"Your predicted score is {prediction}")

if __name__=="__main__":
    main()