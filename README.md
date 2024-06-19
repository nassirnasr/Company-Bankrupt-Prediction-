 Company Bankruptcy Prediction
This project predicts the likelihood of a company's bankruptcy using a machine learning model trained on financial data. It includes data cleaning, visualization, model training, and a user interface built with Streamlit.

Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/nassirnasr/Company-Bankruptcy-Prediction-.git
    cd CompanyBankruptcyPrediction
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows: `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure Jupyter Lab and Streamlit are installed:
    ```sh
    pip install jupyterlab streamlit
    ```

 Data Cleaning and Model Training

1. Open and run the Jupyter Notebook:
    ```sh
    jupyter lab CompanyBankruptPrediction.ipynb
    ```

2. Follow the notebook instructions to clean data, visualize it, train the model, and save `bankrupt_model.pkl` and `cleaned_data.csv`.

Using the Streamlit App

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Input the financial features in the sidebar to get bankruptcy predictions.

## Streamlit App Code (app.py)

```python
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
inputs = {feature: st.sidebar.number_input(feature_names.get(feature, feature), value=0.0, step=0.000001, format="%.6f") for feature in input_features}

# Convert user input to array
user_input = np.array([inputs[feature] for feature in input_features]).reshape(1, -1)

# Predict bankruptcy
if st.sidebar.button("Predict"):
    prediction = gbrt.predict(user_input)
    prediction_proba = gbrt.predict_proba(user_input)
    
    st.subheader("The company is predicted to be " + ("Bankrupt" if prediction[0] == 1 else "Alive"))
    st.write(f"Probability of being Bankrupt: {prediction_proba[0][1] * 100:.2f}%")
    st.write(f"Probability of being Alive: {prediction_proba[0][0] * 100:.2f}%")

    # Plot the results
    st.subheader("Prediction Probability")
    st.bar_chart({"Alive": [prediction_proba[0][0]], "Bankrupt": [prediction_proba[0][1]]})
