# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 06:45:24 2024

@author: sujan
"""

# Import libraries
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import joblib
import pymysql  # Added pymysql import

# Load the saved model and preprocessing objects
model = joblib.load('ab.pkl')  # Change the filename if needed
impute_data = joblib.load('impute.pkl')
scale = joblib.load('scale.pkl')
winzor = joblib.load('winzor.pkl')


def predict(data):
    # Apply preprocessing pipelines
    newprocessed1 = pd.DataFrame(impute_data.transform(data), columns=data.columns)
    newprocessed2 = pd.DataFrame(scale.transform(newprocessed1), columns=newprocessed1.columns)
    newprocessed3 = pd.DataFrame(winzor.transform(newprocessed2), columns=newprocessed2.columns)
    # Make predictions
    predictions = pd.DataFrame(model.predict(newprocessed3), columns=['Engine_Condition'])

    return predictions


def main():  
    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Vehicle Breakdown Prediction </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")
    st.write('Enter the following values to predict Engine Condition:')

    # Input fields for user to enter values
    engine_rpm = st.text_input('Engine rpm', 'Enter value')
    lub_oil_pressure = st.text_input('Lub oil pressure', 'Enter value')
    fuel_pressure = st.text_input('Fuel pressure', 'Enter value')
    coolant_pressure = st.text_input('Coolant pressure', 'Enter value')
    lub_oil_temp = st.text_input('Lub oil temp', 'Enter value')
    coolant_temp = st.text_input('Coolant temp', 'Enter value')
    
    if st.button('Predict'):
        # Check for empty inputs
        if engine_rpm and lub_oil_pressure and fuel_pressure and coolant_pressure and lub_oil_temp and coolant_temp:
            # Convert user input to float
            engine_rpm = float(engine_rpm)
            lub_oil_pressure = float(lub_oil_pressure)
            fuel_pressure = float(fuel_pressure)
            coolant_pressure = float(coolant_pressure)
            lub_oil_temp = float(lub_oil_temp)
            coolant_temp = float(coolant_temp)

            # Create DataFrame from user input
            data = pd.DataFrame({
                'Engine_rpm': [engine_rpm],
                'Lub_oil_pressure': [lub_oil_pressure],
                'Fuel_pressure': [fuel_pressure],
                'Coolant_pressure': [coolant_pressure],
                'lub_oil_temp': [lub_oil_temp],
                'Coolant_temp': [coolant_temp]
            })

            # Predict using the model
            prediction = predict(data)
            
            # Interpret prediction
            prediction_text = "Breakdown" if prediction.iloc[0, 0] == 1 else "Not breakdown"

            # Display prediction
            st.write(f'Predicted Engine Condition: {prediction_text}')


if __name__ == '__main__':
    main()
