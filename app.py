import streamlit as st
import streamlit.components.v1 as stc

from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#F1B300;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;"> Customer Churn Prediction App </h1>
		    <h2 style="color:white;text-align:center;"> Telecom Company</h2>
	        </div>
            """

desc = """
            #### App Content
            - Exploratory Data Analysis
            - Machine Learning Section
         """


def main():
    stc.html(html_temp)
    menu = ['Home', 'Machine Learning']
    with st.sidebar:
        stc.html("""
                    <style>
                        .circle-image {
                            width: 200px;
                            height: 200px;
                            border-radius: 50%;
                            overflow: hidden;
                            box-shadow: 0 0 10px rgba(1, 1, 1, 1);
                        }
                        
                        .circle-image img {
                            width: 100%;
                            height: 100%;
                            object-fit: cover;
			    object-position: 100% 1%;
			}
                    </style>
                    <div class="circle-image">
                        <img src="https://www.shutterstock.com/image-vector/logo-inspiration-telecom-business-260nw-1889984209.jpg" />                 
                    </div>
                  """, 
		 height=215)
        st.subheader('TELECOM COMPANY')
        st.write("---")
        choice = st.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Welcome to Homepage")
        st.write("---")
        st.subheader("Citizens Income Prediction App")
        st.write("This app will be used to predict whether a customer would be churn or not based on the customer profile.")
        st.subheader("Data Source")
        st.write("Kaggle : https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data")
        st.markdown(desc)
    elif choice == "Machine Learning":
        # st.subheader("Welcome to Machine learning")
        run_ml_app()


if __name__ == '__main__':
    main()
