import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the saved model and scaler
@st.cache
def load_saved_model():
    with open('diabetes_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# User input form for clinical features
def user_input_features():
    pregnancies = st.slider('Pregnancies', 0, 20, 1)
    glucose = st.slider('Glucose', 50, 200, 120)
    blood_pressure = st.slider('Blood Pressure', 30, 150, 70)
    skin_thickness = st.slider('Skin Thickness', 10, 50, 20)
    insulin = st.slider('Insulin', 0, 850, 180)
    bmi = st.slider('BMI', 10, 50, 25)
    diabetes_pedigree_function = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    age = st.slider('Age', 21, 100, 30)

    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    return features

# Prediction function
def predict(features, model, scaler):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0]

# Main Streamlit app logic
def main():
    st.title("Diabetes Prediction App")
    
    # Get user input
    features = user_input_features()

    # Load trained model and scaler
    model, scaler = load_saved_model()

    # Make prediction
    if st.button('Predict Diabetes'):
        prediction = predict(features, model, scaler)
        if prediction == 1:
            st.write("The person is predicted to have Diabetes.")
        else:
            st.write("The person is predicted to be non-diabetic.")
    
    # Show information about the model
    st.write("""
    This app predicts if a person is diabetic based on their clinical features.
    The model was trained on the Pima Indians Diabetes dataset.
    """)

if __name__ == '__main__':
    main()
