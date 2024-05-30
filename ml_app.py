# tempat pemrosesan machine learning ke app.py
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
import pandas as pd
from datetime import date, datetime

# import ml package
import joblib
import os

attribute_info = """
                Penjelasan untuk tiap-tiap kolom :
                - customerID : ID yang bernilai unik untuk tiap customer
                - gender : male atau female
                - SeniorCitizen : No atau Yes
                - Partner : jika customer memiliki partner atau tidak
                - Dependents : jika customer memiliki tanggungan atau tidak
                - PhoneService
                - MultipleLines : 'No phone service', 'No', 'Yes'
                - InternetService : 'DSL', 'Fiber optic', 'No'
                - OnlineSecurity : 'No', 'Yes', 'No internet service'
                - OnlineBackup : 'Yes', 'No', 'No internet service'
                - DeviceProtection : 'No', 'Yes', 'No internet service'
                - TechSupport : 'No', 'Yes', 'No internet service'
                - StreamingTV : 'No', 'Yes', 'No internet service'
                - StreamingMovies : 'No', 'Yes', 'No internet service'
                - tenure : seberapa lama (bulan) menjadi customer
                - Contract : jangka waktu kontrak bernilai 'Month-to-month', 'One year', atau 'Two year'
                - PaperlessBilling : jika customer memilih billing dengan paperless atau tidak
                - PaymentMethod : 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)
                - MonthlyCharges : tagihan bulanan
                - TotalCharges : total tagihan
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
        customerID = st.text_input("Customer ID", value="", max_chars=10, type="default", placeholder=None)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        SeniorCitizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
        Partner = st.selectbox("Partner", ['Yes', 'No'])
        Dependents = st.selectbox("Dependents", ['Yes', 'No'])
        tenure = st.number_input("Tenure", step=1)
        placeholder_for_PhoneService = st.empty()
        placeholder_for_addPhoneService = st.empty()
        placeholder_for_InternetService = st.empty()
        placeholder_for_addInternetService = st.empty()
        Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
        PaymentMethod = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        MonthlyCharges = st.number_input("Monthly Charges")   
        TotalCharges = st.number_input("Total Charges")
      
        submitted = st.form_submit_button("Submit")

    with placeholder_for_PhoneService:
      PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])

    with placeholder_for_addPhoneService:
      if PhoneService == 'No':
          MultipleLines = st.selectbox("Multiple Lines", options=['Yes', 'No'], disabled=True)
          MultipleLines = 'No Phone Service'
        else:
          MultipleLines = st.selectbox("Multiple Lines", options=['Yes', 'No'], disabled=False)

    with placeholder_for_InternetService:
      InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

    with placeholder_for_addInternetService:
      if InternetService == 'No':
          OnlineSecurity = st.selectbox("Online Security", options=['Yes', 'No'], disabled=True)
          OnlineSecurity = 'No internet service'
          OnlineBackup = st.selectbox("Online Backup", options=['Yes', 'No'], disabled=True)
          OnlineBackup = 'No internet service'
          DeviceProtection = st.selectbox("Device Protection", options=['Yes', 'No'], disabled=True)
          DeviceProtection = 'No internet service'
          TechSupport = st.selectbox("Tech Support", options=['Yes', 'No'], disabled=True)
          TechSupport = 'No internet service'
          StreamingTV = st.selectbox("Streaming TV", options=['Yes', 'No'], disabled=True)
          StreamingTV = 'No internet service'
          StreamingMovies = st.selectbox("Streaming Movies", options=['Yes', 'No'], disabled=True)
          StreamingMovies = 'No internet service'
        else:
          OnlineSecurity = st.selectbox("Online Security", options=['Yes', 'No'], disabled=False)
          OnlineBackup = st.selectbox("Online Backup", options=['Yes', 'No'], disabled=False)
          DeviceProtection = st.selectbox("Device Protection", options=['Yes', 'No'], disabled=False)
          TechSupport = st.selectbox("Tech Support", options=['Yes', 'No'], disabled=False)
          StreamingTV = st.selectbox("Streaming TV", options=['Yes', 'No'], disabled=False)
          StreamingMovies = st.selectbox("Streaming Movies", options=['Yes', 'No'], disabled=False)
    
    if submitted:
        with st.expander("Your Selected Options"):
            result = {
                "customerID": customerID,
                "gender": gender,
                "SeniorCitizen": SeniorCitizen,
                "Partner": Partner,
                "Dependents": Dependents,
                "tenure": tenure,
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges
            }

        # st.write(result)

        feature = pd.read_csv(os.path.join('feature_churn.csv'))
        df1 = pd.read_csv(os.path.join('df1_churn.csv'))
        df_baru = pd.DataFrame(result, index=[0])
        st.write("Your Selected Options :")
        st.table(df_baru)
        df_baru.drop(columns='customerID', inplace=True)
      
        # data cleaning
        addService = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        for service in addService:
          df_baru[service] = df_baru[service].apply(lambda x: 'Yes' if x=='Yes' else 'No')

        # label encoding
        for col in df1.columns:
          if df1[col].dtype == 'object':
              if len(list(df1[col].unique())) <= 2:
                le = LabelEncoder()
                df1[col] = le.fit_transform(df1[col])
                df_baru[col] = le.transform(df_baru[col])

        # one-hot encoding
        df_baru = pd.get_dummies(df_baru)
        for col in feature.columns:
            if col not in df_baru.columns:
                df_baru[col] = 0
      
        # feature engineering
        services = ['PhoneService', 'MultipleLines', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        df_baru['Sum_of_Services'] = df_baru[services].sum(axis=1)
        
        df_baru = df_baru[feature.columns]  # match column

        # scaling
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(feature)
        df_baru_scaled = scaler.transform(df_baru)

        # prediction section
        st.subheader('Prediction Result')
        single_array = np.array(df_baru_scaled).reshape(1, -1)

        model = load_model("model_lr.pkl")

        prediction = model.predict(single_array)

        if prediction == 0:
            st.info("""
                Hasil Prediksi Churn : 
                Customer tidak churn
                """)
        elif prediction == 1:
            st.info("""
                Hasil Prediksi Churn : 
                Customer churn
                """)
