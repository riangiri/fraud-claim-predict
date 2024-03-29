import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Buat judul (page title) dipaling atas tab browser
st.set_page_config(
    page_title= 'Permodelan Deteksi Fraud claim pada insurance',
    layout="wide",
    initial_sidebar_state="auto"
)

st.set_option('deprecation.showPyplotGlobalUse', False)

def runEDA():
    st.title('Model Machine Learning deteksi Fraud dalam claim asuransi kendaraan')
    # buat deskripsi
    st.write('##### Nama : Raden Rian Girianom')
    st.write('##### Batch : RMT-028')
    st.write('##### Link Dataset : [Click](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection/data)')
    st.write('##### Latar Belakang : Peningkatan tren kasus fraud pada industri keuangan pasca pandemi Covid19, khususnya financial insurance terdapat temuan beberapa tipe fraud salah satunya adalah customer fraud yang menyumbang hampir 44 % penyebab kerugian di industri keuangan dan pelaku external terbanyak menyebabkan kerugian adalah 26 % berasal dari customer, 24 % Serangan hacker dan sisanya pihak ketiga (vendor) 19%. Penggunaan strategi, teknologi detection fraud dan KYC sangat penting untuk mengurangi potensi kerugian perusahaan insurance. ([Berdasarkan riset dari PWC Global economic crime and fraud survey tahun 2022](https://www.pwc.com/gx/en/services/forensics/economic-crime-survey.html))')
    st.write('##### Projek ini bertujuan mengklasifikasi model detection apakah claim insurance dinyatakan credibel atau fraud menggunakan Machine learning dengan memilih salah satu algoritma terbaik klasifikasi decisiontree, randomforest, svc, atau logreg berdasrkan nilai cross validation dengan metrik f1_score')

    st.markdown('---') # membuat garis pemisah ---
    st.subheader('Tahap EDA : Exploratory Data Analysis')
    
    #Buat show dataframe
    st.write('#### Informasi dataset yang telah diolah')
    st.write('##### Dataset berisi : 15400 row/data dan 33 fitur.')
    df2 = pd.read_csv('fraud_oracle.csv')
    st.dataframe(df2)

    # Buat 1 EDA bar Bulan
    st.write('#### EDA 1 = Bulan apa ditemukan tingkat Fraud tertinggi?')
    # Fraud 1 = Yes, 0 = tidak. Sehingga kita gunakan filter 1 untuk menemukan fraud
    fraud_df = df2[df2['FraudFound_P'] == 1]

    # Group by berdasarkan bulan dan FraudFound_P
    fraud_by_month = fraud_df.groupby('Month')['FraudFound_P'].count().reset_index()

    # Visualisasi berdasarkan data diatas
    plt.figure(figsize=(8, 4))
    plt.bar(fraud_by_month['Month'], fraud_by_month['FraudFound_P'], color='skyblue')
    plt.xlabel('Bulan')
    plt.ylabel('Fraud Rate : Yes')
    plt.title('Fraud Rate berdasarkan Bulan')
    plt.show()
    st.pyplot()

    # Buat EDA 2 Persentase claim terdeteksi fraud
    st.write('#### EDA 2 = Persentase claim terdeteksi fraud dengan keseluruhan data')
    # Cek persentase claim terdeteksi fraud
    percentage_distribution = df2['FraudFound_P'].value_counts(normalize=True) * 100

    plt.figure(figsize=(8, 4))
    plt.pie(percentage_distribution, labels=percentage_distribution.index, autopct='%1.1f%%', colors=['skyblue', 'orange'])
    plt.title('Persentase temuan fraud')
    plt.show()
    print(percentage_distribution)
    st.pyplot()

    # Buat EDA 3 Persebaran area dengan tingkat fraud claim tinggi
    st.write('#### EDA 3 = Area mana yang mempunyai tingkat fraud tertinggi pada bulan Maret?')
    # Selanjutnya adalah pada bulan maret area mana yang mendominasi tingkat fraud dalam claim asuransi
    maret_fraud = df2[(df2['Month'] == 'Mar') & (df2['FraudFound_P'] == 1)]

    fraud_by_area = maret_fraud['AccidentArea'].value_counts()
    # Plotting bar
    plt.figure(figsize=(7, 4))
    fraud_by_area.plot(kind='bar', color='skyblue')
    plt.xlabel('Lokasi Urban=Kota, Rural=Pedesaan')
    plt.ylabel('Total Temuan fraud (Yes)')
    plt.title('Jumlah Fraud berdasarkan lokasi accident')
    plt.xticks(rotation=45)
    plt.show()
    st.pyplot()

    # Buat 4 EDA Tipe agen yang terlibat dalam fraud
    st.write('#### EDA 4 = Berapa tipe agen asuransi terlibat pengurusan claim yang terindikasi fraud?')
    # Menentukan kateegori fraud=1 yang artinya Yes, terlebih dahulu
    fraud_data = df2[df2['FraudFound_P'] == 1]
    # Menentukan agent type yang terdeteksi / terlibat dalam fraud berdasarkan data
    agent_type = fraud_data['AgentType'].value_counts()

    # Plot visualisasi
    plt.figure(figsize=(5, 4))
    plt.pie(agent_type, labels=agent_type.index, autopct='%1.1f%%', colors=['skyblue', 'orange'])
    plt.title('Type agent mengurus claim terindikasi fraud')
    plt.show()
    print("Type agent mengurus claim terindikasi fraud:")
    print(agent_type)
    st.pyplot()

    # Buat 5 EDA 
    st.write ('#### EDA 5 = Apakah marital status mempengaruhi probabilitas terjadinya fraud claim?')
    marital_status_count = df2.groupby(['MaritalStatus', 'FraudFound_P']).size().unstack()
    plt.bar(marital_status_count.index, marital_status_count[1], color='red', label='Fraud claim')
    plt.bar(marital_status_count.index, marital_status_count[0], bottom=marital_status_count[1], color='blue', label='Not fraud')

    # Labeling and styling
    plt.xlabel('Marital Status')
    plt.ylabel('Total Claim')
    plt.title('Marital Status vs Fraud Claim')
    plt.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    st.pyplot()

if __name__ == '__main__':
    runEDA()