import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

@st.cache_resource
# Load models
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)
    
def set_background_image_local(image_path):
    with open(image_path, "rb") as file:
        data = file.read()
    base64_image = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: contain;
            background-position: fit;
            background-repeat: repeat;
            background-attachment: fixed;
        }}     
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image_local(r"headlights-car.jpg")
    
model_car=load_model("RandomForestRegressor.pkl")

encoder_city=load_model("encoder_city.pkl")
encoder_model=load_model("encoder_model.pkl")
encoder_insurance=load_model("encoder_insurance.pkl")
encoder_fuel_type=load_model("encoder_fuel_type.pkl")
encoder_transmission=load_model("encoder_transmission.pkl")
encoder_ownership=load_model("encoder_ownership.pkl")

st.title("Car Price Prediction App")
df=pd.read_csv("Final_UsedCars_Data.csv")
categorical_features = ["city", "model", "insurance", "fuel_type", "transmission", "ownership"]
dropdown_options = {feature: df[feature].unique().tolist() for feature in categorical_features}

tab1, tab2, tab3 = st.tabs(["Home","Predict","ChatBot"])

with tab1:
    st.markdown("""
    **Welcome to the Car Price Prediction App!**
    
    This tool helps estimate car prices based on various attributes such as city, model, fuel type, year of manufacture, and more.
    
    **How it works:**
    - Enter car details in the "Predict" tab.
    - Click "Predict" to get an estimated price.
    """)

with tab2:
    a1, a2, a3 = st.columns(3)
    a4, a5, a6 = st.columns(3)
    a7, a8, a9 = st.columns(3)
    a10, a11, a12 = st.columns(3)

    with a1:
        city_select = st.selectbox("Select City", dropdown_options["city"])
        city=encoder_city.transform([[city_select]])[0][0]
    with a2:
        model_select = st.selectbox("Select Car Model", dropdown_options["model"])
        model=encoder_model.transform([[model_select]])[0][0]
    with a3:
        insurance_select = st.selectbox("Insurance Type", dropdown_options["insurance"])
        insurance=encoder_insurance.transform([[insurance_select]])[0][0]
    with a4:
        fuel_type_select = st.selectbox("Fuel Type", dropdown_options["fuel_type"])
        fuel_type=encoder_fuel_type.transform([[fuel_type_select]])[0][0]
    with a5:
        transmission_select = st.selectbox("Transmission Type", dropdown_options["transmission"])
        transmission=encoder_transmission.transform([[transmission_select]])[0][0]
    with a6:
        ownership_select = st.selectbox("Ownership Count", dropdown_options["ownership"])
        ownership=encoder_ownership.transform([[ownership_select]])[0][0]
    with a7:
        seats = st.number_input("Enter Seat Capacity", min_value=2, max_value=10, value=5)
    with a8:
        kms_driven = st.number_input("Enter KM Driven", min_value=1000, value=10000)
    with a9:
        year_of_manufacture = st.number_input("Manufacturing Year", min_value=1900, value=2015)
    with a10:
        engine = st.number_input("Enter Engine CC", min_value=500, value=1200)
    with a11:
        power = st.number_input("Enter Power (HP)", min_value=10.0, value=100.0)
    with a12:
        mileage = st.number_input("Enter Mileage (kmpl)", min_value=5.0, value=15.0)

    if st.button("Predict"):
        input_data = {"city":city, "model":model,"insurance": insurance,"fuel_type" :fuel_type,"seats": seats,"kms_driven" :kms_driven, 
                                "transmission":transmission,"year_of_manufacture": year_of_manufacture, "engine":engine, "power":power,"mileage": mileage,"ownership": ownership}
        input_df=pd.DataFrame([input_data])
        # Call prediction function
        predicted_price = model_car.predict(input_df)
with tab3:
    import pickle
    import numpy as np
    
    # Load the trained model
    with open("RandomForestRegressor.pkl", "rb") as file:
        model = pickle.load(file)
    
    # Chatbot function
    def chatbot_response(user_input):
        try:
            # Expecting user input as comma-separated values
            features = np.array([float(x) for x in user_input.split(",")]).reshape(1, -1)
            prediction = model.predict(features)[0]
            return f"Estimated Car Price: ₹{prediction:,.2f}"
        except:
            return "Invalid input! Please enter numeric values separated by commas."
    
    # Streamlit UI
    st.title("🚗 Car Price Prediction Chatbot")
    st.write("Enter car details as comma-separated values (e.g., Year, Mileage, Engine Size, etc.)")
    
    # Chatbot interaction
    user_input = st.text_input("You: ", "")
    if user_input:
        response = chatbot_response(user_input)
        st.text_area("Chatbot:", response, height=100)
    
        
            
            st.subheader("Predicted Car Price")
            st.markdown(f"### :green[₹ {predicted_price[0]:,.2f}]")   
