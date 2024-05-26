# tempat pemrosesan machine learning ke app.py
import streamlit as st
import numpy as np
from sklearn.preprocessing import RobustScaler
import pandas as pd
from datetime import date, datetime

# import ml package
import joblib
import os

attribute_info = """
                Penjelasan untuk tiap-tiap kolom :
                - ID SPK : ID Transaksi, Nomor identifikasi unik untuk setiap transaksi yang masuk.
                - Tipe Order : Menunjukkan jenis pesanan yang dilakukan oleh pelanggan.
                    - Order : Transaksi pertama dari pelanggan.
                    - Re-Order : Transaksi kedua dari pelanggan untuk judul yang berbeda.
                    - Cetak Ulang : Transaksi kedua dari pelanggan untuk judul yang sama.
                    - Retur: Transaksi di mana nilai transaksi menjadi 0 karena pelanggan kecewa sehingga buku dikembalikan kepada penerbit. Jika pelanggan mengembalikan barang dan menginginkan barang baru karena kesalahan internal, transaksi akan diganti dengan nilai 100%.
                - Tanggal Masuk Naskah : Tanggal ketika naskah pertama kali diterima atau dimasukkan ke dalam sistem.
                - Tanggal Mulai Proses : Tanggal ketika proses pra-cetak dimulai untuk naskah yang diterima.
                - Judul Naskah : Judul naskah buku
                - Tanggal Deal : Tanggal ketika penulis menyetujui hasil layout buku.
                - Sales Order: Nominal yang dibayar oleh pelanggan untuk transaksi tersebut.
                - Kuantitas: Jumlah eksemplar buku yang dicetak dalam transaksi tersebut.
                """


def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model


def run_ml_app():
    st.subheader("Machine Learning Section")
    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    with st.form("my_data"):
        id_spk = st.number_input("ID SPK", step=1)
        tipe_order = st.selectbox("Tipe Order", [
                                  'order', 'reorder', 'cetak ulang', 'order ebook', 'cancel', 'retur'])
        id_cust = st.number_input("ID Konsumen", step=1)
        tgl_naskah = st.date_input("Tgl Masuk Naskah", value=date.today())
        tgl_proses = st.date_input("Tgl Mulai Proses", value=date.today(), help="Mohon input tanggal lebih dari atau sama dengan Tgl Masuk Naskah")
        judul_naskah = st.text_input(
            "Judul Naskah", value="", max_chars=None, type="default", placeholder=None)
        tgl_deal = st.date_input("Tanggal Deal", value=date.today(), help="Mohon input tanggal lebih dari atau sama dengan Tgl Mulai Proses")
        sales_order = st.number_input("Sales Order", step=1)
        kuantitas = st.number_input("Kuantitas", step=1)

        submitted = st.form_submit_button("Submit")

    if submitted:
        with st.expander("Your Selected Options"):
            result = {
                "ID SPK": id_spk,
                "Tipe Order": tipe_order,
                "ID Konsumen": id_cust,
                "Tgl Masuk Naskah": tgl_naskah,
                "Tgl Mulai Proses": tgl_proses,
                "Judul Naskah": judul_naskah,
                "Tanggal Deal": tgl_deal,
                "Sales Order": sales_order,
                "Kuantitas": kuantitas
            }

        # st.write(result)

        feature = pd.read_csv(os.path.join('feature.csv'))
        df1 = pd.read_csv(os.path.join('df1.csv'))
        data_baru = result
        df_baru = pd.DataFrame(data_baru, index=[0])
        st.write("Your Selected Options :")
        st.table(df_baru)
        df_baru['Tgl Masuk Naskah'] = pd.to_datetime(df_baru['Tgl Masuk Naskah']).dt.strftime('%Y-%m-%d')
        df_baru['Tgl Mulai Proses'] = pd.to_datetime(df_baru['Tgl Mulai Proses']).dt.strftime('%Y-%m-%d')
        df_baru['Tanggal Deal'] = pd.to_datetime(df_baru['Tanggal Deal']).dt.strftime('%Y-%m-%d')
        df_lama = df1[df1['ID Konsumen'] == df_baru.loc[0, 'ID Konsumen']]
        df_id = pd.concat([df_lama, df_baru], ignore_index=True).sort_values(by=['Tanggal Deal']).reset_index(drop=True)

        # feature engineering
        idx = df_id.loc[df_id['ID SPK'] == df_baru.loc[0, 'ID SPK']].index

        if str(df_baru.loc[0, 'Tipe Order']) == 'order':
            df_baru.loc[0, 'First Order'] = 1
        else:
            df_baru.loc[0, 'First Order'] = 0

        df_baru.loc[0, 'Reorder Count'] = ((df_id['Tanggal Deal'] < df_baru.loc[0, 'Tanggal Deal']) & (df_id['Tipe Order'] == 'reorder')).sum()
        df_baru.loc[0, 'Cetak Ulang Count'] = ((df_id['Tanggal Deal'] < df_baru.loc[0, 'Tanggal Deal']) & (df_id['Tipe Order'] == 'cetak ulang')).sum()
        df_baru.loc[0, 'Order Ebook Count'] = ((df_id['Tanggal Deal'] < df_baru.loc[0, 'Tanggal Deal']) & (df_id['Tipe Order'] == 'order ebook')).sum()
        df_baru.loc[0, 'Cancel Count'] = ((df_id['Tanggal Deal'] < df_baru.loc[0, 'Tanggal Deal']) & (df_id['Tipe Order'] == 'cancel')).sum()
        df_baru.loc[0, 'Retur Count'] = ((df_id['Tanggal Deal'] < df_baru.loc[0, 'Tanggal Deal']) & (df_id['Tipe Order'] == 'retur')).sum()

        df_baru.loc[0, 'Day Diff Proses to Deal'] = (pd.to_datetime(df_baru.loc[0, 'Tanggal Deal']) - pd.to_datetime(df_baru.loc[0, 'Tgl Mulai Proses'])).days

        if idx == 0:
            df_baru.loc[0, 'Day Diff to Prev Trx'] = -1
        else:
            df_baru.loc[0, 'Day Diff to Prev Trx'] = (pd.to_datetime(df_baru.loc[0, 'Tanggal Deal']) - pd.to_datetime(df_id.loc[idx-1, 'Tanggal Deal']).max()).days

        if (str(df_baru.loc[0, 'Tipe Order']) == 'cancel' or 'retur') and (idx == 0):
            df_baru.loc[0, 'Day Diff to First Order'] = -1
        else:
            df_baru.loc[0, 'Day Diff to First Order'] = (pd.to_datetime(df_baru.loc[0, 'Tanggal Deal']) - pd.to_datetime(df_id.loc[df_id['First Order'] == 1, 'Tanggal Deal']).min()).days

        df_baru.loc[0, 'Day Diff to Last Trx'] = (pd.to_datetime(df_baru.loc[0, 'Tanggal Deal']) - pd.to_datetime(df_id.loc[df_id.index[-1], 'Tanggal Deal'])).days

        df_baru.drop(columns=['ID SPK', 'Tgl Masuk Naskah', 'Tgl Mulai Proses',
                     'ID Konsumen', 'Tanggal Deal', 'Judul Naskah'], inplace=True)

        # encoding data kategorikal
        df_baru = pd.get_dummies(df_baru)
        for kolom in feature.columns:
            if kolom not in df_baru.columns:
                df_baru[kolom] = 0

        df_baru = df_baru[feature.columns]  # match column

        # scaling
        scaler = RobustScaler()
        df_scaled = scaler.fit_transform(feature)
        df_baru_scaled = scaler.transform(df_baru)

        # prediction section
        st.subheader('Prediction Result')
        single_array = np.array(df_baru_scaled).reshape(1, -1)

        model = load_model("model_lgbm.pkl")

        prediction = model.predict(single_array)

        if prediction == 0:
            st.info("""
                Hasil Prediksi Churn : 
                Konsumen tidak churn
                """)
        elif prediction == 1:
            st.info("""
                Hasil Prediksi Churn : 
                Konsumen churn
                """)
