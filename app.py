import streamlit as st
import streamlit.components.v1 as stc

from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#6F8FAF;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;"> Customer Churn Prediction App </h1>
		    <h2 style="color:white;text-align:center;"> Telecom Company</h2>
	        </div>
            """

desc_temp = """
            ### Customer Churn Prediction App
            This app will be used to predict whether a customer would be churn or not based on the customer profile.
            #### App Content
            - Exploratory Data Analysis
            - Machine Learning Section
            """


def main():
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
			}
   
   			.position-100-70 {
      			    object-position: 100% 70%;
			}
                    </style>
                    <div class="circle-image">
                        <img src="https://www.shutterstock.com/image-vector/logo-inspiration-telecom-business-260nw-1889984209.jpg" />                 
                    </div>
                  """
        st.subheader('TELECOM COMPANY')
        st.write("---")
        choice = st.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Welcome to Homepage")
        st.write("---")
        st.markdown(desc_temp)
    elif choice == "Machine Learning":
        # st.subheader("Welcome to Machine learning")
        run_ml_app()


if __name__ == '__main__':
    main()
