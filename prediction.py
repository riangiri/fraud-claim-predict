import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import json

# Load Model
with open('dt_model.pkl', 'rb') as file_1: # rb = read binary
  dtgrid = pickle.load(file_1)

def runpred():
    #buat form inputan
    with st.form('Form_Data_Classifikasi_fraud'):
        MonthClaimed = st.slider('MonthClaimed', min_value = -1.555, max_value = 1.616, help = 'terakhir claimed')
        Bulan = st.slider('Month', min_value = -1.555, max_value = 1.616, help = 'kodebulan')
        Pernah_claim_sebelumnya = st.slider('PastNumberOfClaims', min_value = -1.552, max_value = 1.627, help = 'terendah=belum pernah, tertinggi=pernah')
        Pihak_Penyebab_Kecelakaan = st.slider('Fault', min_value = -1.5549, max_value = 1.616, help = 'terendah=Pihak lain, tertinggi=Pemilik asuransi')
        TipePolis = st.slider('Type_Polis', min_value = 0.0, max_value = 3.0, help = 'terendah-tertinggi')
        st.markdown('---')
        Biaya_asuransi = st.slider('Deductible', min_value = 0.0, max_value = 1.0, help = 'Tingkatan biaya asuransi')
        Tipe_asuransi = st.slider('TypeInsurance', min_value = 0.0, max_value = 7.0, help = 'Tingkatan paket asuransi')
        VehiclePrice = st.slider('VehiclePrice', min_value = 0.0, max_value = 1.62, help = 'Semakin besar nilai, semakin mahal harga kendaraan')
        Frekuensi_pindah_rumah = st.slider('AddressChange_Claim', min_value = 0.0, max_value = 3.0, help = 'Semakin besar nilai semakin sering pindah rumah lebih dari 2x')
        Umur_kendaraan = st.slider('AgeOfVehicle', min_value = 0.0, max_value = 5.0, help = 'Semakin besar nilai semakin tua umur kendaraan')
        Umur_pemegang_polis = st.slider('AgeOfPolicyHolder', min_value = 0.0, max_value = 4.0, help = 'Semakin besar nilai semakin tua')
        Berapa_kali_laporan_ke_Polisi = st.slider('PoliceReportFiled', min_value = 0.0, max_value = 7.0, help = 'Semakin besar nilai semakin sering')
        Kategori_Kendaraan = st.slider('VehicleCategory', min_value = -1.552, max_value = 1.62, help = 'Category Kendaraan')
        Brand = st.slider('Brand', min_value = -1.552, max_value = 1.62, help = 'Brand Merk berdasarkan (code)')
        Lokasi_Kejadian = st.slider('AccidentArea', min_value = 0.0, max_value = 3.0, help = 'terendah = pedesaan, tertinggi = perkotaan')
        
        #submit button
        submitted = st.form_submit_button('Predict')

    data_inf_final = {
        'MonthClaimed': 1.55,
        'Month': 1.6,
        'PastNumberOfClaims':1.6,
        'Fault': 1.6,
        'Type_Polis' : 3.0,
        'Deductible': 1.0,
        'TypeInsurance': 7.0,
        'VehiclePrice': 3.0,
        'AddressChange_Claim': 2.0,
        'AgeOfVehicle': 5.0,
        'AgeOfPolicyHolder' : 4.0,
        'PoliceReportFiled' : 7.0,
        'VehicleCategory' : 1.6,
        'Brand' : 1.61,
        'AccidentArea' : 3.0
}

    data_inf_final = pd.DataFrame([data_inf_final])
    st.dataframe(data_inf_final)
    if submitted:
        y_pred_inf = dtgrid.predict(data_inf_final)

        st.write('## Clasifikasi : ', str(int(y_pred_inf)))

if __name__ == '__main__':
   runpred()

