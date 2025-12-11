import streamlit as st
import joblib

# Memuat model, scaler, dan label encoder yang sudah disimpan
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

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

# Membuat DataFrame dari input
input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'daily_steps': [daily_steps],
    'sleep_hours': [sleep_hours],
    'smoker': [smoker],
    'alcohol': [alcohol],
    'systolic_bp': [systolic_bp],
    'diastolic_bp': [diastolic_bp],
    'gender': [gender_encoded]
})

# Feature Engineering
input_data['bp_ratio'] = input_data['systolic_bp'] / input_data['diastolic_bp']
input_data['pulse_pressure'] = input_data['systolic_bp'] - input_data['diastolic_bp']
input_data['is_obese'] = (input_data['bmi'] >= 30).astype(int)
input_data['low_sleep'] = (input_data['sleep_hours'] < 6).astype(int)
input_data['risk_score'] = input_data['smoker'] + input_data['alcohol'] + input_data['is_obese'] + input_data['low_sleep']

# Scaling data with the pre-fitted scaler
input_scaled = scaler.transform(input_data)

# Prediksi dengan model yang sudah dilatih
prediction = model.predict(input_scaled)

# Tampilkan hasil prediksi
if prediction[0] == 1:
    st.write("**Hasil Prediksi: High Risk**")
else:
    st.write("**Hasil Prediksi: Low Risk**")
