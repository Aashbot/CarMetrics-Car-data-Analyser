import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from PIL import Image
import os
from streamlit_option_menu import option_menu

########################################################

def load_preloaded_data():
    # Load the pre-loaded dataset
    df = pd.read_csv("indian_cars.csv")

    # Handle missing values
    df.replace("?", np.nan, inplace=True)
    df = df.astype({"horsepower": float})  # Convert 'horsepower' to numeric

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    df['horsepower'] = imputer.fit_transform(df[['horsepower']])

    # Drop rows with missing price values
    df.dropna(subset=['price'], axis=0, inplace=True)

    return df

@st.cache_data
def preprocess_user_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.replace("?", np.nan, inplace=True)
    df = df.astype({"horsepower": float})
    imputer = SimpleImputer(strategy='mean')
    df['horsepower'] = imputer.fit_transform(df[['horsepower']])
    df.dropna(subset=['price'], axis=0, inplace=True)
    return df

@st.cache_data
def train_model(df):
    X = df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

#######################################################

st.title("CarMetrics: Car Data Analyser")

selected = option_menu(
    menu_title=None,
    options=["Home", "Analysis", "Buy", "Sell"],
    icons=["house", "bar-chart", "basket", "shop"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

company_names = [
    "Hyundai", "Toyota", "MG", "Tata", "Volkswagen", 
    "Ford", "Honda", "Kia", "Maruti Suzuki", "Skoda", 
    "Mahindra", "Renault"
    ]

if selected == "Home":
    logo_path = 'Screenshot 2024-06-23 003628-Photoroom.png'
    st.image(logo_path, use_column_width=True)
    st.write("Welcome to the CarMetrics: Car Data Analyser app. Navigate to the Analysis, Buy, or Sell section using the menu above.")

elif selected == "Analysis":

    st.write("### Car Analysis")

    lower_price = st.number_input("Enter the lower price limit", min_value=0, value=None)
    upper_price = st.number_input("Enter the upper price limit", min_value=0, value=None)
    company_name = st.selectbox("Select a company name", company_names,index=None)

    if st.button("Start Analysis"):
        # Read the dataset
        df = pd.read_csv("indian_cars_200_unique.csv")

        # Filter the dataset based on the price range and company name
        df = df[(df['price'] >= lower_price) & (df['price'] <= upper_price) & (df['company'].str.lower() == company_name.lower())]


        if df.empty:
            st.write("No cars found matching the criteria.")
        else:
            # Display the first few rows of the dataset
            st.write("### Dataset Preview")
            st.write(df.head())

        # Step 2: Exploratory Data Analysis (EDA)
        st.write("### Exploratory Data Analysis (EDA)")

        # Summary statistics
        st.write("#### Summary Statistics")
        st.write(df.describe())

        # Visualize the distribution of numerical features
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        fig, ax = plt.subplots(len(numerical_features), figsize=(10, 20))
        for i, col in enumerate(numerical_features):
            sns.histplot(df[col], kde=True, ax=ax[i])

        # Visualize correlations
        st.write("#### Correlation Matrix")
        plt.figure(figsize=(12, 8))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt.gcf())  # Use plt.gcf() to get the current figure
        st.write("### How to Read the Correlation Graph")
        st.write("#### What Is This Graph?")
        st.write("This graph is called a **correlation heatmap**. It helps us understand how different car features (like weight, horsepower, and price) are related to each other.")
        
        st.write("#### What Do the Colors Mean?")
        st.write("- **Red/Orange**: Strong relationship.")
        st.write("  - **Dark Red**: Very strong.")
        st.write("  - **Light Orange**: Moderate.")
        st.write("- **Blue**: Weak or no relationship.")
        st.write("  - **Dark Blue**: Very weak or negative relationship.")
        st.write("  - **Light Blue**: Slightly weak.")
        
        st.write("#### What Do the Numbers Mean?")
        st.write("- The numbers in each box tell us how two features are related.")
        st.write("- **Positive Numbers**: When one feature goes up, the other feature also goes up.")
        st.write("  - **Example**: If the number is 0.95, it means these two features go up together almost all the time.")
        st.write("- **Negative Numbers**: When one feature goes up, the other feature goes down.")
        st.write("  - **Example**: If the number is -0.78, it means if one feature goes up, the other feature goes down a lot.")
        st.write("- **Zero**: No relationship. The features don’t affect each other.")
        
        st.write("#### How to Use This Graph")
        st.write("- **Find the two features you are interested in.**")
        st.write("  - Look at the row for one feature.")
        st.write("  - Look at the column for the other feature.")
        st.write("  - The box where they meet shows how they are related.")
        
        st.write("#### Examples")
        st.write("1. **mpg and weight (-0.83)**:")
        st.write("   - Dark blue box.")
        st.write("   - **Means**: When the car’s weight goes up, the mpg goes down a lot.")
        st.write("2. **cylinders and displacement (0.95)**:")
        st.write("   - Dark red box.")
        st.write("   - **Means**: When the number of cylinders goes up, the engine displacement also goes up a lot.")
        st.write("3. **horsepower and price (0.48)**:")
        st.write("   - Light red box.")
        st.write("   - **Means**: When horsepower goes up, the price usually goes up, but not always.")
        
        st.write("### Conclusion")
        st.write("- **Strong Positive**: Dark red/orange numbers close to 1.")
        st.write("- **Strong Negative**: Dark blue numbers close to -1.")
        st.write("- **No Relationship**: Numbers close to 0.")
        
        st.write("This graph helps you understand which car features are connected, making it easier to decide what kind of car you might want based on what’s important to you!")

        # Step 3: Handling Missing Values
        # Impute missing values with mean for numerical columns
        imputer = SimpleImputer(strategy='mean')
        df[numerical_features] = imputer.fit_transform(df[numerical_features])

        # Step 4: Outlier Detection and Removal
        # Only apply IQR method to numeric columns
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1

        # Removing outliers
        df_outliers_removed = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

        # Step 5: Feature Engineering
        # Encode categorical variables
        df_encoded = pd.get_dummies(df_outliers_removed, drop_first=True)

        # Check if 'price' column exists
        #and 'image_path' in df.columns and 'url' in df.columns
        if 'price' in df_encoded.columns and 'image_path' in df.columns and 'url' in df.columns :
            # Extract the car names (assuming 'name' column exists)
            car_names = df_outliers_removed['name']

            # Split the data into features and target variable
            X = df_encoded.drop('price', axis=1)
            y = df_encoded['price']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Step 6: Model Building
            #st.write("### Model Building")
            # Random Forest Regressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)

            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)

            # Step 7: Model Evaluation
            #st.write("### Model Evaluation")

            # Random Forest evaluation
            mse_rf = mean_squared_error(y_test, y_pred_rf)
            r2_rf = r2_score(y_test, y_pred_rf)

            # Linear Regression evaluation
            mse_lr = mean_squared_error(y_test, y_pred_lr)
            r2_lr = r2_score(y_test, y_pred_lr)

            # Step 9: Best Car Recommendations
            st.write("### Best Car Recommendations")

            # Get the top 5 cars with the highest prices from the test set
            top_5_indices = np.argsort(y_test)[-5:][::-1]
            top_5_cars = df_outliers_removed.iloc[top_5_indices]

            # Save the top 5 cars to session state
            st.session_state['top_5_cars'] = top_5_cars

            st.write("#### The top best cars to buy based on the analysis:")
            for idx, car in top_5_cars.iterrows():
                cols = st.columns([1, 3])  # Adjust the column ratio as needed
                with cols[0]:
                    if os.path.exists(car['image_path']):
                        image = Image.open(car['image_path'])
                        st.image(image, caption=car['name'], use_column_width=True)
                    else:
                        st.write("Image not found.")
                with cols[1]:
                    st.write(f"**Car:** {car['name']}")
                    st.write(car[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']])
                    st.write("Price:", car['price'])

        else:
            st.write("The dataset does not contain 'price', 'image_path', or 'url' columns.")

elif selected == "Buy":
    st.write("### Buy Your Car Here")
    df = pd.read_csv("indian_cars_200_unique.csv")
    st.write("### Compare Cars")
    company_name = st.selectbox("Select a company name", company_names,index=None)
    filtered_cars = df[df['company'].str.lower() == company_name.lower()]
    car_names = filtered_cars['name'].unique()
    compare = st.multiselect("Select cars to compare", car_names)
    if len(compare) > 1:
        compare_df = filtered_cars[filtered_cars['name'].isin(compare)]
        st.write(compare_df[['name', 'price', 'mpg', 'horsepower', 'weight', 'acceleration', 'displacement', 'cylinders']])

    st.write("### Finance Calculator")
    loan_amount = st.number_input("Loan Amount", min_value=0, value=None)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1, value=None)
    loan_term = st.number_input("Loan Term (years)", min_value=1, max_value=30, value=None)

    if st.button("Calculate Monthly Payment"):
        monthly_rate = interest_rate / 100 / 12
        num_payments = loan_term * 12
        if monthly_rate > 0:
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        else:
            monthly_payment = loan_amount / num_payments
        st.write(f"Estimated Monthly Payment: ₹{monthly_payment:.2f}")


    if 'top_5_cars' in st.session_state:
        top_5_cars = st.session_state['top_5_cars']
        if not top_5_cars.empty:
            # Display top 5 cars
            st.write("### Top 5 Cars")
            rows = [st.columns(2), st.columns(2), st.columns([1, 2, 1])]  # Define rows for displaying cars
            for idx, car in top_5_cars.iterrows():
                if idx < 4:
                    row = rows[2]
                    col = row[1]
                else:
                    row = rows[2]
                    col = row[1]
                with col:
                    if os.path.exists(car['image_path']):
                        image = Image.open(car['image_path'])
                        st.image(image, use_column_width=True)
                    st.markdown(f'<center><a href="{car["url"]}" target="_blank">{car["name"]}</a></center>', unsafe_allow_html=True)
                    st.markdown(f'<center><a href="https://www.olx.in/cars/" target="_blank">Buy Second Hand {car["name"]}</a></center>', unsafe_allow_html=True)
        else:
            st.write("No car recommendations available. Please perform the analysis first.")
    else:
        st.write("No car recommendations available. Please perform the analysis first.")

elif selected == "Sell":
    st.write("## Sell Your Car")
    st.write("""
    Use this section to get an estimated price for your car. Provide detailed information about your car to get an accurate estimate.
    """)

    df = load_preloaded_data()
    rf_model = train_model(df)

    # Form for user inputs
    #st.write("### Car Information")
    with st.form("sell_form"):
        mpg = st.number_input("Miles Per Gallon (MPG)", min_value=0.0,value=None)
        cylinders = st.number_input("Cylinders", min_value=1, max_value=16,value=None)
        displacement = st.number_input("Engine Displacement (cc)", min_value=0.0,value=None)
        horsepower = st.number_input("Horsepower", min_value=0.0,value=None)
        weight = st.number_input("Weight (lbs)", min_value=0.0,value=None)
        acceleration = st.number_input("Acceleration (0-60 mph in seconds)", min_value=0.0,value=None)

        submitted = st.form_submit_button("Get Estimated Price")

        if submitted:
            # Prepare the input features for prediction
            features = np.array([[mpg, cylinders, displacement, horsepower, weight, acceleration]])

            # Predict the price using the trained model
            estimated_price = rf_model.predict(features)[0]

            price_range_lower = estimated_price * 0.9
            price_range_upper = estimated_price * 1.1

            st.write("### Estimated Price")
            st.write(f"The estimated price for your car is between ₹{price_range_lower:,.2f} and ₹{price_range_upper:,.2f}")

            st.write("### List Your Car for Sale")
            st.write("Here are some platforms where you can list your car for sale:")
            st.markdown("""
            - [Cars24](https://www.cars24.com/)
            - [OLX](https://www.olx.in/cars/)
            - [CarDekho](https://www.cardekho.com/)
            - [Zigwheels](https://www.zigwheels.com/)
            - [CarTrade](https://www.cartrade.com/)
            - [Droom](https://droom.in/)
            - [Spinny](https://www.spinny.com/)
            """)
