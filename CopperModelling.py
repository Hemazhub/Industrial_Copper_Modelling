import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler

# Page Configuration
st.set_page_config(page_title="Copper Modelling",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Creating Page
option = option_menu(menu_title=None,
                     options=["Home","Price Prediction","Status"],
                     icons=['house','coin','graph-up'],
                     default_index=0,
                     orientation='horizontal',
                     styles={'backgroundColor':'#eccfa5',
                             'secondaryBackgroundColor':'#fffdfd',
                             'primaryColor':'#bb542a',
                             'textColor':'#0c0c0c'})

#Home Menu
if option == 'Home':
    st.title("Industrial Copper Modelling")
    st.write("The industrial copper market is dynamic and subject to various factors that influence its selling price. "
                 "Accurately predicting the selling price of copper is crucial for manufacturers, traders, and stakeholders to make informed business decisions and plan their operations effectively. "
                 "Additionally, categorizing the status of copper transactions helps in monitoring and streamlining the transaction lifecycle.")
    st.write("The goal of this project is to develop a machine learning solution that predicts the selling price of copper based on historical transaction data and classifies the status of each transaction. "
                 "By leveraging features such as item date, quantity (in tons), customer information, country, item type, application, dimensions (thickness, width), and material/product references, "
                 "the model aims to provide reliable price predictions and status classifications.")

elif option == "Price Prediction":
    st.title('Industrial Copper Modelling')
    # tab1, tab2 = st.tabs(['Regression', 'Classification'])
    product_refs = [1670798778, 1668701718, 628377, 640665, 611993,
                    1668701376, 164141591, 1671863738, 1332077137, 640405,
                    1693867550, 1665572374, 1282007633, 1668701698, 628117,
                    1690738206, 628112, 640400, 1671876026, 164336407,
                    164337175, 1668701725, 1665572032, 611728, 1721130331,
                    1693867563, 611733, 1690738219, 1722207579, 929423819,
                    1665584320, 1665584662, 1665584642]
    countries = [28, 25, 30, 32, 38, 78, 27, 77, 113, 79, 26, 39, 40, 84, 80, 107, 89]

    # with tab1:
    with st.form('form'):
        col1, col2 = st.columns(2, gap='large')
        # ['quantity tons','customer','country','application','thickness','width','product_ref','item_type','status']
        with col1:
            quantity = st.slider("Quantity range:", min_value= 0.00001, max_value = 1000000000.0)
            customer = st.slider("Customer:" ,min_value= 12458.0, max_value= 30408185.0)
            country = st.selectbox('country', options=countries)
            application = st.slider("Application range:",min_value= 2 ,max_value= 99)
            thickness = st.slider("Thickness range:", min_value= 0.18, max_value= 400.0)
        with col2:
            width = st.slider("width:",min_value= 1, max_value = 2990)
            product_ref = st.selectbox('product_ref', options=product_refs)
            item_type = st.radio('item_type', options=['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'],
                                 horizontal=True)
            status = st.radio('status', options=['Won', 'Lost'], horizontal=True)
            predict = st.form_submit_button('predict selling price')
            try:
                if predict:
                    with st.spinner('Getting Price'):
                        with open("C:/Users/Guvi_projects/Copper Modelling/copper_rfr.pkl", 'rb') as f:
                            model = pickle.load(f)
                        with open("C:/Users/Guvi_projects/Copper Modelling/cat1.pkl", 'rb') as f:
                            ohe1 = pickle.load(f)
                        with open("C:/Users/Guvi_projects/Copper Modelling/cat2.pkl", 'rb') as f:
                            ohe2 = pickle.load(f)
                        with open("C:/Users/Guvi_projects/Copper Modelling/scaler.pkl", 'rb') as f:
                            scaler = pickle.load(f)

                        sample = np.array([[np.log(float(quantity)), customer, country, application, np.log(float(thickness)),
                                            width, product_ref, item_type, status]])
                        nums = [quantity, customer, country, application, thickness, width, product_ref]
                        if all(nums) > 0:
                            a = sample[:, :7]
                            b = cat1.transform(sample[:, [7]]).toarray()
                            c = cat2.transform(sample[:, [8]]).toarray()
                            x = np.concatenate([a, b, c], axis=1)
                            y = scaler.transform(x)
                            z = model.predict(y)
                            prediction = np.exp(z)[0]
                            st.markdown(f'### The Selling Price is :green[{prediction}]')
            except:
                st.error("Enter valid details")

elif option == "Status":
    st.title('Industrial Copper Modelling')
    # tab1, tab2 = st.tabs(['Regression', 'Classification'])
    product_refs = [1670798778, 1668701718, 628377, 640665, 611993,
                    1668701376, 164141591, 1671863738, 1332077137, 640405,
                    1693867550, 1665572374, 1282007633, 1668701698, 628117,
                    1690738206, 628112, 640400, 1671876026, 164336407,
                    164337175, 1668701725, 1665572032, 611728, 1721130331,
                    1693867563, 611733, 1690738219, 1722207579, 929423819,
                    1665584320, 1665584662, 1665584642]
    countries = [28, 25, 30, 32, 38, 78, 27, 77, 113, 79, 26, 39, 40, 84, 80, 107, 89]

    # with tab2:
    with st.form('form2'):
        col1, col2 = st.columns(2, gap='large')
        # ['quantity tons','customer','country','application','thickness','width','product_ref','item_type','status']
        with col1:
            quantity = st.slider("Quantity range:", min_value= 0.00001, max_value = 1000000000.0)
            customer = st.slider("Customer:", min_value=12458.0, max_value=30408185.0)
            country = st.selectbox('country', options=countries)
            application = st.slider("Application range:", min_value=2, max_value=99)
            thickness = st.slider("Thickness range:", min_value=0.18, max_value=400.0)
        with col2:
            width = st.slider("width:", min_value=1, max_value=2990)
            product_ref = st.selectbox('product_ref', options=product_refs)
            item_type = st.radio('item_type', options=['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'],
                                 horizontal=True)
            selling_price = st.slider('selling_price:', min_value=0.1, max_value=100001015.0)
            predict = st.form_submit_button('predict status')
            try:
                if predict:
                    with st.spinner('Getting Status'):
                        with open("C:/Users/Guvi_projects/Copper Modelling/copper_rfc.pkl", 'rb') as f:
                            model = pickle.load(f)
                        with open("C:/Users/Guvi_projects/Copper Modelling/ohe3.pkl", 'rb') as f:
                            ohe3 = pickle.load(f)
                        with open("C:/Users/Guvi_projects/Copper Modelling/le.pkl", 'rb') as f:
                            le = pickle.load(f)
                        with open("C:/Users/Guvi_projects/Copper Modelling/scaler1.pkl", 'rb') as f:
                            scaler1 = pickle.load(f)

                        sample = np.array([[np.log(float(quantity)), customer, country, application,
                                            np.log(float(thickness)), width, product_ref,
                                            np.log(float(selling_price)), item_type]])
                        nums = [quantity, customer, country, application, thickness, width, product_ref,
                                selling_price]
                        if all(nums) > 0:
                            a = sample[:, :8]
                        b = ohe3.transform(sample[:, [8]]).toarray()
                        x = np.concatenate([a, b], axis=1)
                        y = scaler1.transform(x)
                        z = model.predict(y)
                        prediction = le.inverse_transform(z)[0]
                        st.markdown(f'### The Status is :green[{prediction}]')
            except:
                st.warning('Enter valid Details')


