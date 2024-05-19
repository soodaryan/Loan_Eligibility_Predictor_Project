import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

model = joblib.load('logistic_regression_model.pkl')
st.title('Loan Eligibility Predictor')

def user_input_features():
    Loan_ID = st.text_input("Enter Loan Id")
    Gender = st.selectbox('Gender', ('Male', 'Female'))
    Married = st.selectbox('Married', ('Yes', 'No'))
    Dependents = st.selectbox('Dependents', ('0', '1', '2', '3+'))
    Education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
    Self_Employed = st.selectbox('Self Employed', ('Yes', 'No'))
    ApplicantIncome = st.number_input('Applicant Income (in dollars)',value=0)
    CoapplicantIncome = st.number_input('Coapplicant Income (in dollars)', value=0)
    LoanAmount = st.number_input('Loan Amount (in dollars)', value=0)
    Loan_Amount_Term = st.number_input('Loan Amount Term', value=360)
    Credit_History = st.selectbox('Credit History', (1, 0))
    Property_Area = st.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))
    
    data = {
        'Gender_Male': Gender,
        'Married_Yes': Married,
        'Dependents': Dependents,
        'edu' : Education,
        'Self_Employed_Yes': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

input_df['Gender_Male'] = input_df['Gender_Male'].map({'Male': 1, 'Female': 0})
input_df['Married_Yes'] = input_df['Married_Yes'].map({'Yes': 1, 'No': 0})
input_df['Dependents'] = input_df['Dependents'].astype(str)
dependents_dummies = pd.get_dummies(input_df['Dependents'], prefix='Dependents')
dependents_dummies[dependents_dummies.columns[0]] = 1
cols = ["Dependents_0", "Dependents_1",	"Dependents_2",	"Dependents_3+"]
for col_name in cols :
    if col_name not in dependents_dummies.columns:
        dependents_dummies[col_name] = 0

input_df = pd.concat([input_df, dependents_dummies], axis=1)
input_df.drop(columns = "Dependents" , inplace = True)

input_df.loc[: , 'Education_Graduate'] = input_df.loc[: ,'edu'].map({'Graduate': 1, 'Not Graduate': 0})
input_df['Self_Employed_Yes'] = input_df['Self_Employed_Yes'].map({'Yes': 1, 'No': 0})

input_df['Property_Area'] = input_df['Property_Area'].astype(str)
property_area_dummies = pd.get_dummies(input_df['Property_Area'], prefix='Property_Area').astype(int)
property_area_dummies[property_area_dummies.columns[0]] = 1
for area in ['Urban', 'Semiurban', 'Rural']:
    col_name = f'Property_Area_{area}'
    if col_name not in property_area_dummies.columns:
        property_area_dummies[col_name] = 0
input_df = pd.concat([input_df, property_area_dummies], axis=1)
input_df.drop(columns = ["Property_Area" , "edu"] , inplace  = True) 

final_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Gender_Male', 'Married_Yes', 'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Graduate', 'Self_Employed_Yes', 'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']

st.subheader('User Input parameters')
st.write(input_df[final_cols])

input_df["ApplicantIncome"] = np.sqrt(input_df.ApplicantIncome)
input_df["CoapplicantIncome"] = np.sqrt(input_df.CoapplicantIncome)
input_df["LoanAmount"] = np.sqrt(input_df.LoanAmount)

if 'Unnamed: 0' in input_df.columns:
    input_df = input_df.drop(columns=['Unnamed: 0'])
    
loaded_scaler = joblib.load('my_scaler.pkl') 
X_new = input_df.copy()
scale = ["ApplicantIncome","CoapplicantIncome","LoanAmount"]
X_new[scale] = loaded_scaler.transform(input_df[scale])
input_df = X_new

if st.button('Submit'):
    prediction = model.predict(input_df[final_cols])

    st.subheader('Prediction')
    loan_status = np.array(['Not Eligible', 'Eligible'])

    st.markdown(
        f"""
        <div style="padding: 10px; border: 1px solid #000; border-radius: 5px; font-size: 20px;">
            {loan_status[prediction][0]}
        </div>
        """, unsafe_allow_html=True
    )

    