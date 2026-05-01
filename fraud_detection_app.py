import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
option = st.sidebar.radio("Choose an option:", ["Test from Dataset","Manual Input"])

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model_and_data():
    # These will be created from your notebook
    model_data = {
        'selected_features': None,
        'le_dict': None,
        'scaler': None,
        'cat_cols': None,
    }
    return model_data

# Function to preprocess input
def preprocess_input(input_data, selected_features, le_dict, cat_cols):
    # Encode categorical variables
    for col in cat_cols:
        if col in input_data.columns:
            le = le_dict[col]
            input_data[col] = le.transform(input_data[col].astype(str))
    
    # Select only the features used in training
    input_data = input_data[selected_features]
    
    return input_data

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
        st.info("⚠️ Note: Model training and predictions will be available once you run all cells in the notebook")
        
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
    
    # Load dataset
    try:
        df = pd.read_csv('fraud_oracle.csv')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total Records in Dataset:** {len(df)}")
            st.write(f"**Fraud Cases:** {df['FraudFound_P'].sum()}")
            st.write(f"**Non-Fraud Cases:** {len(df) - df['FraudFound_P'].sum()}")
        
        with col2:
            fraud_percentage = (df['FraudFound_P'].sum() / len(df)) * 100
            st.write(f"**Fraud Percentage:** {fraud_percentage:.2f}%")
            st.write(f"**Non-Fraud Percentage:** {100 - fraud_percentage:.2f}%")
        
        st.markdown("---")
        
        # Choose record type
        record_type = st.radio("Select Record Type:", 
                              ["Random Record", "Fraud Record", "Non-Fraud Record"])
        
        if st.button("🎲 Load Sample Record", use_container_width=True):
            if record_type == "Random Record":
                sample = df.sample(1)
            elif record_type == "Fraud Record":
                fraud_df = df[df['FraudFound_P'] == 1]
                if len(fraud_df) > 0:
                    sample = fraud_df.sample(1)
                else:
                    st.error("No fraud records found!")
                    sample = None
            else:  # Non-Fraud Record
                non_fraud_df = df[df['FraudFound_P'] == 0]
                if len(non_fraud_df) > 0:
                    sample = non_fraud_df.sample(1)
                else:
                    st.error("No non-fraud records found!")
                    sample = None
            
            if sample is not None:
                st.markdown("#### Selected Record Details:")
                
                # Display in columns
                col1, col2, col3 = st.columns(3)
                
                sample_dict = sample.to_dict(orient='records')[0]
                
                display_data = pd.DataFrame(list(sample_dict.items()), 
                                          columns=['Feature', 'Value'])
                
                st.dataframe(display_data, use_container_width=True, hide_index=True)
                
                # Show actual label
                actual_label = sample['FraudFound_P'].values[0]
                st.markdown("---")
                
                if actual_label == 1:
                    st.markdown('<div class="prediction-box fraud-box"><h3>🚨 Actual Label: FRAUD</h3></div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-box safe-box"><h3>✅ Actual Label: NOT FRAUD</h3></div>', 
                               unsafe_allow_html=True)
                
                # Predict button
                if st.button("🔮 Predict This Record", use_container_width=True):
                    st.info("⚠️ Model prediction will be available once you train the model in the notebook")
    
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; margin-top: 2em;'>
    <p><strong>Insurance Fraud Detection System</strong></p>
    <p>Built with Python, Scikit-Learn, and Streamlit</p>
    <p>Dataset: Fraud Oracle - Vehicle Insurance Claims</p>
</div>
""", unsafe_allow_html=True)