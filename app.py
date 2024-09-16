import pandas as pd
import numpy as np
import streamlit as st
import pickle 
import tensorflow as tf


with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

model = tf.keras.models.load_model('smart.h5')

st.title('Smart Watch Purchase Propensity')

age = st.number_input('Age',0,100)
overall_recency = st.number_input('Overall Recency',0,2000)
overall_ATS = st.number_input('Overall ATS',0,100000)
overall_frequency = st.number_input('Overall Frequency',0,20)
smart_recency = st.number_input('Smart Recency',0,2000)
smart_ATS = st.number_input('Smart ATS',0,100000)
smart_frequency = st.number_input('Smart Frequency',0,20)

input_data = pd.DataFrame({
    'age' : [age],
    'overall_recency' : [overall_recency],
    'overall_ats' : [overall_ATS],
    'overall_frequency' : [overall_frequency],
    'smart_recency' : [smart_recency],
    'smart_ats' : [smart_ATS],
    'smart_frequency' : [smart_frequency]
})

scaled_input = scaler.transform(input_data)

prediction = model.predict(scaled_input)
prediction_proba = prediction[0][0]

st.write(f'Smart Product Purchase Propensity: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to purchase a smart product.')
else:
    st.write('The customer is not likely to purchase a smart product.')
