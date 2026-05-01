import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Fraud Detection System", layout="wide", initial_sidebar_state="expanded")

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        color: #FF6B6B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1em;
    }
    .sub-header {
        font-size: 1.5em;
        color: #4ECDC4;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .fraud-box {
        background-color: #FFE5E5;
        border-left: 5px solid #FF6B6B;
    }
    .safe-box {
        background-color: #E5F9F7;
        border-left: 5px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔍 Insurance Fraud Detection System</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## Navigation")
option = st.sidebar.radio("Choose an option:", [ "Test from Dataset","Manual Input"])

# ------- OPTION 1: MANUAL INPUT -------
if option == "Manual Input":
    st.markdown('<div class="sub-header">📝 Enter Insurance Claim Details</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        week_of_month = st.number_input("Week of Month", min_value=1, max_value=5, value=1)
        day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", 
                                                    "Thursday", "Friday", "Saturday", "Sunday"])
        make = st.selectbox("Vehicle Make", ["Honda", "Toyota", "Ford", "Chevrolet", 
                                             "Pontiac", "Mazda", "BMW", "Mercedes"])
        accident_area = st.selectbox("Accident Area", ["Urban", "Rural"])
    
    with col2:
        day_of_week_claimed = st.selectbox("Day of Week Claimed", ["Monday", "Tuesday", "Wednesday", 
                                                                    "Thursday", "Friday", "Saturday", "Sunday"])
        month_claimed = st.selectbox("Month Claimed", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        week_of_month_claimed = st.number_input("Week of Month Claimed", min_value=1, max_value=5, value=1)
        sex = st.selectbox("Sex", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    
    with col3:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        fault = st.selectbox("Fault", ["Policy Holder", "Third Party"])
        policy_type = st.selectbox("Policy Type", ["Sport - Liability", "Sport - Collision", 
                                                    "Sedan - Liability", "Sedan - Collision",
                                                    "Utility - All Perils", "Sedan - All Perils"])
        vehicle_category = st.selectbox("Vehicle Category", ["Sport", "Sedan", "Utility"])
        vehicle_price = st.selectbox("Vehicle Price", ["20000 to 29000", "30000 to 39000", 
                                                       "40000 to 59000", "more than 69000"])
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        deductible = st.number_input("Deductible", min_value=100, max_value=1000, value=400, step=100)
        driver_rating = st.number_input("Driver Rating", min_value=1, max_value=4, value=1)
        days_policy_accident = st.selectbox("Days Policy Accident", ["0-7", "8-15", "16-30", "more than 30"])
        days_policy_claim = st.selectbox("Days Policy Claim", ["0-7", "8-15", "16-30", "more than 30"])
        past_number_of_claims = st.selectbox("Past Number of Claims", ["none", "1", "2 to 4", "more than 4"])
    
    with col5:
        age_of_vehicle = st.selectbox("Age of Vehicle", ["new", "1 year", "2 years", "3 years", 
                                                         "4 years", "5 years", "6 years", "7 years", "more than 7"])
        age_of_policy_holder = st.selectbox("Age of Policy Holder", ["16 to 17", "18 to 21", "21 to 25",
                                                                     "26 to 30", "31 to 35", "36 to 40",
                                                                     "41 to 50", "51 to 65", "over 65"])
        police_report_filed = st.selectbox("Police Report Filed", ["Yes", "No"])
        witness_present = st.selectbox("Witness Present", ["Yes", "No"])
        agent_type = st.selectbox("Agent Type", ["External", "Internal"])
    
    with col6:
        number_of_suppliments = st.selectbox("Number of Suppliments", ["none", "1 to 2", "3 to 5", "more than 5"])
        address_change_claim = st.selectbox("Address Change Claim", ["no change", "1 year", "2 years", "3 years", "4 years"])
        number_of_cars = st.selectbox("Number of Cars", ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"])
        year = st.number_input("Year", min_value=1990, max_value=2024, value=1994)
        base_policy = st.selectbox("Base Policy", ["Liability", "Collision", "All Perils"])
    
    # Prediction button
    if st.button("🔮 Predict Fraud", key="predict_manual", use_container_width=True):
        st.info("⚠️ Note: Model integration coming soon. Train the model in the notebook first.")
        
        # Create input dataframe
        input_dict = {
            'Month': month,
            'WeekOfMonth': week_of_month,
            'DayOfWeek': day_of_week,
            'Make': make,
            'AccidentArea': accident_area,
            'DayOfWeekClaimed': day_of_week_claimed,
            'MonthClaimed': month_claimed,
            'WeekOfMonthClaimed': week_of_month_claimed,
            'Sex': sex,
            'MaritalStatus': marital_status,
            'Age': age,
            'Fault': fault,
            'PolicyType': policy_type,
            'VehicleCategory': vehicle_category,
            'VehiclePrice': vehicle_price,
            'Deductible': deductible,
            'DriverRating': driver_rating,
            'Days_Policy_Accident': days_policy_accident,
            'Days_Policy_Claim': days_policy_claim,
            'PastNumberOfClaims': past_number_of_claims,
            'AgeOfVehicle': age_of_vehicle,
            'AgeOfPolicyHolder': age_of_policy_holder,
            'PoliceReportFiled': police_report_filed,
            'WitnessPresent': witness_present,
            'AgentType': agent_type,
            'NumberOfSuppliments': number_of_suppliments,
            'AddressChange_Claim': address_change_claim,
            'NumberOfCars': number_of_cars,
            'Year': year,
            'BasePolicy': base_policy
        }
        
        input_df = pd.DataFrame([input_dict])
        
        st.success("✅ Input data processed successfully!")
        st.dataframe(input_df, use_container_width=True)

# ------- OPTION 2: TEST FROM DATASET -------
elif option == "Test from Dataset":
    st.markdown('<div class="sub-header">📊 Test with Dataset Samples</div>', unsafe_allow_html=True)
    
    st.info("📌 Please upload the fraud_oracle.csv file to use this feature, or connect to a data source.")
    
    st.markdown("""
    ### Dataset Information
    - **Total Records:** 15,420
    - **Fraud Cases:** ~923 (5.99%)
    - **Non-Fraud Cases:** ~14,497 (94.01%)
    - **Features:** 33 columns
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", "15,420")
        st.metric("Fraud Cases", "923")
    
    with col2:
        st.metric("Non-Fraud Cases", "14,497")
        st.metric("Fraud Percentage", "5.99%")
    
    st.markdown("---")
    st.write("Use the Manual Input option above to test the model with custom data.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; margin-top: 2em;'>
    <p><strong>Insurance Fraud Detection System</strong></p>
    <p>Built with Python, Scikit-Learn, and Streamlit</p>
    <p>Dataset: Fraud Oracle - Vehicle Insurance Claims</p>
</div>
""", unsafe_allow_html=True)
