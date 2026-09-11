import streamlit as st
import joblib
import pandas as pd

st.title("Prédiction du prix d'un logement")

# Loading Model
model = joblib.load("regression.joblib")

# Form
size = st.number_input("Surface (en m²)", min_value=0.0, value=100.0, step=1.0)
nb_rooms = st.number_input("Nombre de chambres", min_value=0, value=2, step=1)
garden = st.number_input("Jardin (1 = Oui, 0 = Non)", min_value=0, max_value=1, value=0, step=1)

# Prediction
features = pd.DataFrame([[size, nb_rooms, garden]], columns=['size', 'nb_rooms', 'garden'])
prediction = model.predict(features)

st.write(f"**Prix estimé :** {prediction[0]:.2f} €")