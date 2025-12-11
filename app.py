import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Memuat model, scaler, dan label encoder yang sudah disimpan
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Kolom yang digunakan saat pelatihan
model_columns = ['age', 'bmi', 'daily_steps', 'sleep_hours', 'smoker', 'alcohol', 
                 'systolic_bp', 'diastolic_bp', 'gender', 'calories_consumed', 
                 'cholesterol', 'family_history', 'resting_hr', 'water_intake_l', 
                 'bp_ratio', 'pulse_pressure', 'is_obese', 'low_sleep', 'risk_score']

# Membuat form input
st.title("Prediksi Risiko Penyakit Berdasarkan Gaya Hidup")

# Input data dari pengguna
age = st.number_input('Age', min_value=0, max_value=120, value=30)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
daily_steps = st.number_input('Daily Steps', min_value=0, max_value=50000, value=5000)
sleep_hours = st.number_input('Sleep Hours', min_value=0.0, max_value=24.0, value=7.0)
smoker = st.selectbox('Smoker', ['No', 'Yes'])
alcohol = st.selectbox('Alcohol Consumption', ['No', 'Yes'])
systolic_bp = st.number_input('Systolic Blood Pressure', min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=60, max_value=120, value=80)
gender = st.selectbox('Gender', ['Male', 'Female'])

# Konversi input ke format yang sesuai
smoker = 1 if smoker == 'Yes' else 0
alcohol = 1 if alcohol == 'Yes' else 0
gender_encoded = label_encoder.transform([gender])[0]

# Menangani input kosong dan menambahkan fitur yang hilang
input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'daily_steps': [daily_steps],
    'sleep_hours': [sleep_hours],
    'smoker': [smoker],
    'alcohol': [alcohol],
    'systolic_bp': [systolic_bp],
    'diastolic_bp': [diastolic_bp],
    'gender': [gender_encoded],
    'calories_consumed': [0],  # Nilai default
    'cholesterol': [0],  # Nilai default
    'family_history': [0],  # Nilai default
    'resting_hr': [0],  # Nilai default
    'water_intake_l': [0]  # Nilai default
})

# Verifikasi input_data
st.write(f"Input Data:\n{input_data}")

# Feature Engineering: Sama seperti yang dilakukan pada data training
input_data['bp_ratio'] = input_data['systolic_bp'] / input_data['diastolic_bp']
input_data['pulse_pressure'] = input_data['systolic_bp'] - input_data['diastolic_bp']
input_data['is_obese'] = (input_data['bmi'] >= 30).astype(int)
input_data['low_sleep'] = (input_data['sleep_hours'] < 6).astype(int)
input_data['risk_score'] = input_data['smoker'] + input_data['alcohol'] + input_data['is_obese'] + input_data['low_sleep']

# Pastikan input_data memiliki fitur yang sama dengan data pelatihan (menggunakan kolom yang sama)
input_data = input_data[model_columns]

# Scaling data dengan scaler yang sudah dilatih
input_scaled = scaler.transform(input_data)

# Prediksi dengan model yang sudah dilatih
prediction = model.predict(input_scaled)

# Tampilkan hasil prediksi
if prediction[0] == 1:
    st.write("**Hasil Prediksi: High Risk**")
else:
    st.write("**Hasil Prediksi: Low Risk**")

# Menyediakan visualisasi koefisien model jika diinginkan
if st.checkbox("Tampilkan Koefisien Model"):
    coef_df = pd.DataFrame({
        'Feature': model_columns,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)

    st.write("\nTop 5 Faktor Peningkat Risiko (Koefisien Positif):")
    st.write(coef_df.head(5))
    st.write("\nTop 5 Faktor Penurun Risiko (Koefisien Negatif):")
    st.write(coef_df.tail(5))
