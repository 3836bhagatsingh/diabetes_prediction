import pandas as pd
import streamlit as st
import pickle

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
uri = "mongodb+srv://3836bhagatsingh:bhagat1234@cluster0.a76tmyr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['Diabetes']
collection = db['prediction']

def load_model():
    with open('./predict_diabetes.pkl', 'rb') as file:
        ridge, lasso, scaler = pickle.load(file)
    return ridge, lasso, scaler

def preprocess_data(input_data, scaler):
    df = pd.DataFrame([input_data], columns=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])
    df_scaled = scaler.transform(df)
    return df_scaled

def prediction_ridge(data, ridge):
    return ridge.predict(data)

def prediction_lasso(data, lasso):
    return lasso.predict(data)

def main():
    st.header('Prediction of Diabetes Progression')
    st.write("Please provide the following health metrics:")

    # Collect input
    age = st.number_input('Age', min_value=0.0, max_value=100.0, value=50.0)
    sex = st.number_input('Sex (0 = female, 1 = male)', min_value=0.0, max_value=1.0, value=1.0)
    bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=25.0)
    bp = st.number_input('Blood Pressure', min_value=0.0, max_value=200.0, value=80.0)
    s1 = st.number_input('S1', value=100.0)
    s2 = st.number_input('S2', value=100.0)
    s3 = st.number_input('S3', value=100.0)
    s4 = st.number_input('S4', value=100.0)
    s5 = st.number_input('S5', value=100.0)
    s6 = st.number_input('S6', value=100.0)

    user_input = [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]

    if st.button("Predict"):
        ridge, lasso, scaler = load_model()
        processed_data = preprocess_data(user_input, scaler)
        ridge_ans = prediction_ridge(processed_data, ridge)
        lasso_ans = prediction_lasso(processed_data, lasso)
        input_dict = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'bp': bp,
        's1': s1,
        's2': s2,
        's3': s3,
        's4': s4,
        's5': s5,
        's6': s6,
        'ridge_prediction': float(ridge_ans[0]),
        'lasso_prediction': float(lasso_ans[0])
    }

        st.success(f"Ridge Regression Prediction: {ridge_ans[0]:.2f}")
        st.success(f"Lasso Regression Prediction: {lasso_ans[0]:.2f}")
        collection.insert_one(input_dict)

if __name__ == '__main__':
    main()
