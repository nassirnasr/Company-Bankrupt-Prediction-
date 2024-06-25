import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the saved model
gbrt = joblib.load('bankrupt_model.pkl')

cleaned_data = pd.read_csv('cleaned_data.csv')

# Define the input features
input_features = cleaned_data.drop('Bankrupt?', axis=1).columns.tolist()

feature_names = {
    'Non-industry income and expenditure/revenue': 'Non-industry income',
    'Interest-bearing debt interest rate': 'Interest debt rate',
    'Net Value Per Share (A)': 'Net value per share',
    'Persistent EPS in the Last Four Seasons': 'EPS last 4 seasons',
    'Net Value Growth Rate': 'Net value growth rate',
    'Interest Expense Ratio': 'Interest expense ratio',
    'Total debt/Total net worth': 'Debt/net worth',
    'Borrowing dependency': 'Borrowing dependency',
    'Net profit before tax/Paid-in capital': 'Net profit before tax',
    'Fixed Assets Turnover Frequency': 'Fixed assets turnover',
    'Cash/Total Assets': 'Cash/Total assets',
    'Net Income to Total Assets': 'Net income/Total assets',
    'Net Income to Stockholder\'s Equity': 'Net income/Equity',
    'Degree of Financial Leverage (DFL)': 'Financial leverage',
    'Equity to Liability': 'Equity to liability'
}

# Title of the app
st.title("Company Bankruptcy Prediction")

# Sidebar for input features
st.sidebar.header("Input Features")
inputs = {}
for feature in input_features:
    display_name = feature_names.get(feature, feature)
    inputs[feature] = st.sidebar.number_input(display_name, value=0.0, step=0.000001, format="%.6f")

# Convert user input to array
user_input = np.array([inputs[feature] for feature in input_features]).reshape(1, -1)

# Predict bankruptcy
if st.sidebar.button("Predict"):
    prediction = gbrt.predict(user_input)
    prediction_proba = gbrt.predict_proba(user_input)

    if prediction[0] == 1:
        st.subheader("The company is predicted to be Bankrupt")
    else:
        st.subheader("The company is predicted to be Alive")

    st.write("Probability of being Bankrupt: {:.2f}%".format(prediction_proba[0][1] * 100))
    st.write("Probability of being Alive: {:.2f}%".format(prediction_proba[0][0] * 100))

    # Plot the results
    st.subheader("Prediction Probability")
    st.bar_chart({"Alive": [prediction_proba[0][0]], "Bankrupt": [prediction_proba[0][1]]})
