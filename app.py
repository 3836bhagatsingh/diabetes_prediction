import pandas as pd
import streamlit as st
import pickle

def load_model():
    with open('./predict_diabetes.pkl','rb') as file:
        ridge,lasso,scaler  = pickle.load(file)
    return ridge,lasso,scaler

def preprocess_data(input_data,scaler):
    df = pd.DataFrame(input_data,columns = ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6'])
    df = scaler.transform(df)
    return df

def prediction_ridge(data,ridge):
    ans  = ridge.predict(data)
    return ans

def prediction_lasso(data,lasso):
    ans = lasso.predict(data)
    return ans

def main():
    st.header('Prediction of diabetes score')
    st.write("Please provide following details")

    user_input = [{
    st.number_input('age'),
    st.number_input('sex'),
    st.number_input('bmi'),
    st.number_input('bp'),
    st.number_input('s1'),
    st.number_input('s2'),
    st.number_input('s3'),
    st.number_input('s4'),
    st.number_input('s5'),
    st.number_input('s6')
    }]

    if st.button("Predict"):
        ridge,lasso,scaler = load_model()
        processed_data = preprocess_data(user_input,scaler)
        ridge_ans = prediction_ridge(processed_data,ridge)
        lasso_ans = prediction_lasso(processed_data,lasso)
        st.success(f"Ridge prediction is {ridge_ans[0]:.2f}")
        st.success(f"Lasso prediction is {lasso_ans[0]:.2f}")



    

    



if __name__ == '__main__':
    main()