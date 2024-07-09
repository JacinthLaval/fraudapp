import streamlit as st
import pandas as pd
import numpy as np
pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load data (you'll need to replace this with your actual data loading)
@st.cache_data
def load_data():
    # This is a placeholder. Replace with your actual data loading code.
    data = pd.DataFrame({
        'claim_amount': np.random.randint(100, 10000, 1000),
        'provider_years_experience': np.random.randint(1, 40, 1000),
        'num_procedures': np.random.randint(1, 10, 1000),
        'num_medications': np.random.randint(0, 20, 1000),
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    })
    return data

# Preprocess data
def preprocess_data(df):
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Streamlit app
def main():
    st.title('Medicare Fraud Prediction App')

    # Load data
    data = load_data()

    # Preprocess data
    X_scaled, y, scaler = preprocess_data(data)

    # Train model
    model = train_model(X_scaled, y)

    # User input
    st.sidebar.header('Enter Claim Details')
    claim_amount = st.sidebar.number_input('Claim Amount', min_value=100, max_value=10000, value=1000)
    provider_years = st.sidebar.number_input('Provider Years of Experience', min_value=1, max_value=40, value=10)
    num_procedures = st.sidebar.number_input('Number of Procedures', min_value=1, max_value=10, value=3)
    num_medications = st.sidebar.number_input('Number of Medications', min_value=0, max_value=20, value=5)

    # Make prediction
    if st.sidebar.button('Predict Fraud'):
        user_input = np.array([[claim_amount, provider_years, num_procedures, num_medications]])
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)
        probability = model.predict_proba(user_input_scaled)

        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.warning('This claim is predicted to be fraudulent.')
        else:
            st.success('This claim is predicted to be legitimate.')

        st.write(f'Probability of fraud: {probability[0][1]:.2%}')

    # Display some basic stats
    st.subheader('Data Overview')
    st.write(data.describe())

    # Display a sample of the data
    st.subheader('Sample Data')
    st.write(data.head())

if __name__ == '__main__':
    main()
