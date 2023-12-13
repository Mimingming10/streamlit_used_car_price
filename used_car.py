import pickle
import streamlit as st
import matplotlib.pyplot as plt

model = pickle.load(open('estimasi_mobil.sav', 'rb'))

st.title("Estimasi Harga Mobil Bekas")

# buat form dengan label "Input Data"
with st.form("Input Data"):
  # buat input widget untuk setiap variabel
  year = st.number_input("Input Tahun Mobil")
  mileage = st.number_input("Input Km Mobil")
  tax = st.number_input("Input Pajak Mobil")
  mpg = st.number_input("Input Konsumsi BBM Mobil")
  engineSize = st.number_input("Input Engine Size")
  # buat tombol submit dengan label "Estimasi Harga"
  submitted = st.form_submit_button("Estimasi Harga")

# jika tombol submit ditekan
if submitted:
  # buat tuple dari input data
  input_data = (year, mileage, tax, mpg, engineSize)
  # prediksi harga mobil bekas dengan model
  predict = model.predict([[year, mileage, tax, mpg, engineSize]])
  # tampilkan hasil prediksi dalam EUR dan Rupiah
  st.write("Estimasi harga mobil bekas dalam EUR : ", predict)
  st.write("Estimasi harga mobil bekas dalam Rupiah : ", predict*16000)
  # buat figure dan axes
  fig, ax = plt.subplots()
  # plot histogram untuk prediksi
  label = f"Input data: {input_data}"
  ax.hist(predict, bins=10, alpha=0.5, label=label)
  # tambahkan legend ke axes
  ax.legend()
  # tampilkan figure dengan st.pyplot
  st.pyplot(fig)
