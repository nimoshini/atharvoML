import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        color: #FF6347;
        text-align: center;
        font-weight: bold;
    }
    .header {
        font-size: 24px;
        color: #4682B4;
        font-weight: bold;
        margin-top: 20px;
    }
    .report {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #ADD8E6;
    }
    .input-section {
        background-color: #FFF8DC;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #F5DEB3;
        margin-top: 20px;
    }
    .predict-button {
        background-color: #32CD32;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        border: none;
    }
    .predict-button:hover {
        background-color: #28A745;
    }
    </style>
    """, unsafe_allow_html=True)

# Step 2: Load the dataset
df = pd.read_csv("E:/Bengaluru_House_Data.csv")

# Sidebar for navigation
st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Choose an option:", ["Home", "Visualizations"])

# Variables to store model performance metrics
rmse = None

if option == "Home":
    # Display dataset and input features
    st.write("<div class='title'>House Price Prediction App</div>", unsafe_allow_html=True)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Data Preprocessing
    df = df.dropna()
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=categorical_cols)
    df = df.sample(frac=0.3, random_state=42)

    X = df.drop('price', axis=1)
    y = df['price']

    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    gbr = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)

    ensemble_model = VotingRegressor(estimators=[('rf', rf), ('gbr', gbr)])
    ensemble_model.fit(X_train, y_train)

    y_pred = ensemble_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Store RMSE for use in other sections
    st.write(f"<div class='header'>Optimized Model RMSE: {rmse}</div>", unsafe_allow_html=True)

    # Streamlit user input for report generation and prediction
    st.header("House Price Prediction")

    # Input section styling
    with st.container():
        st.markdown("<div class='input-section'>", unsafe_allow_html=True)

        total_sqft = st.number_input("Enter Total Square Feet", value=0.0)
        size = st.selectbox("Select Size", options=["1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK"])
        area_type = st.selectbox("Select Area Type", options=["Built-up Area", "Super Built-up Area", "Carpet Area"])
        availability = st.selectbox("Select Availability", options=["Ready to Move", "Under Construction"])
        bath = st.number_input("Enter Number of Baths", min_value=0, max_value=10, value=0)
        balcony = st.number_input("Enter Number of Balconies", min_value=0, max_value=10, value=0)

        if st.button('Generate Report', key='generate_report'):
            st.markdown("<div class='report'>", unsafe_allow_html=True)
            st.write("Generated Report:")
            st.write(f"Total Square Feet: {total_sqft}")
            st.write(f"Size: {size}")
            st.write(f"Area Type: {area_type}")
            st.write(f"Availability: {availability}")
            st.write(f"Number of Baths: {bath}")
            st.write(f"Number of Balconies: {balcony}")
            st.markdown("</div>", unsafe_allow_html=True)

            user_input = pd.DataFrame({
                'total_sqft': [total_sqft],
                'size': [size],
                'area_type': [area_type],
                'availability': [availability],
                'bath': [bath],
                'balcony': [balcony]
            })

            user_input_encoded = pd.get_dummies(user_input)
            user_input_encoded = user_input_encoded.reindex(columns=X.columns, fill_value=0)

            user_input_scaled = scaler.transform(user_input_encoded)
            prediction = ensemble_model.predict(user_input_scaled)
            st.write(f"Predicted House Price: {prediction[0]:.2f}")

        st.markdown("</div>", unsafe_allow_html=True)

elif option == "Visualizations":
    st.header("Data Visualizations")

    # Visualization of the data
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['price'], kde=True, ax=ax, color='skyblue')
    ax.set_title("Distribution of House Prices")
    st.pyplot(fig)

    st.subheader("Total Sqft vs Price")
    fig, ax = plt.subplots()
    sns.scatterplot(x='total_sqft', y='price', data=df, ax=ax, color='coral')
    ax.set_title("Total Square Feet vs Price")
    st.pyplot(fig)


