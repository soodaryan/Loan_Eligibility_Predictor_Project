import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sklearn

# Load pre-trained logistic regression model
model = joblib.load('logistic_regression_model.pkl')

# Title of the app
st.title('Loan Eligibility Predictor')

# # Input fields for the user to fill in the details
# st.sidebar.header('User Input Parameters')

def user_input_features():
    Loan_ID = st.text_input("Enter Loan Id")
    Gender = st.selectbox('Gender', ('Male', 'Female', 'Other'))
    Married = st.selectbox('Married', ('Yes', 'No'))
    Dependents = st.selectbox('Dependents', ('0', '1', '2', '3+'))
    Education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
    Self_Employed = st.selectbox('Self Employed', ('Yes', 'No'))
    ApplicantIncome = st.number_input('Applicant Income',value=0.0)
    CoapplicantIncome = st.number_input('Coapplicant Income', value=0.0)
    LoanAmount = st.number_input('Loan Amount', value=0.0)
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

# # Preprocess the input data to match the model training format
# # Note: Ensure this preprocessing step matches your model training process
input_df['Gender_Male'] = input_df['Gender_Male'].map({'Male': 1, 'Female': 0, 'Other': 2})
input_df['Married_Yes'] = input_df['Married_Yes'].map({'Yes': 1, 'No': 0})
input_df['Dependents'] = input_df['Dependents'].astype(str)
dependents_dummies = pd.get_dummies(input_df['Dependents'], prefix='Dependents')
dependents_dummies[dependents_dummies.columns[0]] = 1
cols = ["Dependents_0", "Dependents_1",	"Dependents_2",	"Dependents_3+"]
for col_name in cols :
    if col_name not in dependents_dummies.columns:
        dependents_dummies[col_name] = 0

# Concatenate the dummy columns to the original DataFrame
input_df = pd.concat([input_df, dependents_dummies], axis=1)
input_df.drop(columns = "Dependents" , inplace = True)

input_df.loc[: , 'Education_Graduate'] = input_df.loc[: ,'edu'].map({'Graduate': 1, 'Not Graduate': 0})
input_df['Self_Employed_Yes'] = input_df['Self_Employed_Yes'].map({'Yes': 1, 'No': 0})
# input_df['Property_Area'] = input_df['Property_Area'].map({'Urban': 0, 'Semiurban': 1, 'Rural': 2})

input_df['Property_Area'] = input_df['Property_Area'].astype(str)
property_area_dummies = pd.get_dummies(input_df['Property_Area'], prefix='Property_Area').astype(int)
property_area_dummies[property_area_dummies.columns[0]] = 1
for area in ['Urban', 'Semiurban', 'Rural']:
    col_name = f'Property_Area_{area}'
    if col_name not in property_area_dummies.columns:
        property_area_dummies[col_name] = 0
input_df = pd.concat([input_df, property_area_dummies], axis=1)
input_df.drop(columns = ["Property_Area" , "edu"] , inplace  = True) 

if 'Unnamed: 0' in input_df.columns:
    input_df = input_df.drop(columns=['Unnamed: 0'])

# Display user input
st.subheader('User Input parameters')
st.write(input_df)

# Make predictions
final_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Gender_Male', 'Married_Yes', 'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Graduate', 'Self_Employed_Yes', 'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']
prediction = model.predict(input_df[final_cols])
# prediction_proba = model.predict_proba(input_df)

# Display the prediction

st.subheader('Prediction')
loan_status = np.array(['Not Eligible', 'Eligible'])

st.markdown(
    f"""
    <div style="padding: 10px; border: 1px solid #000; border-radius: 5px; font-size: 20px;">
        {loan_status[prediction][0]}
    </div>
    """, unsafe_allow_html=True
)
# st.write(loan_status[prediction])
# # Display the prediction probability
# st.subheader('Prediction Probability')
# st.write(prediction_proba)

    