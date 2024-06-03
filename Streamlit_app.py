import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained pipeline model
pipeline = joblib.load('churn_model_pipeline.pkl')

# Load the dataset
data = pd.read_csv('telco.csv')

# Drop columns that can cause data leakage
leakage_columns = ['Churn Score', 'Customer Status', 
                   'Churn Category', 'Churn Reason', 'Churn Label']
data.drop(columns=leakage_columns, inplace=True)

# Identify features for training
features = [col for col in data.columns if col not in leakage_columns + ['Customer ID']]

# Identify categorical and numerical features
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Filter out non-feature columns from categorical and numerical features
categorical_features = [col for col in categorical_features if col in features]
numerical_features = [col for col in numerical_features if col in features]

# Convert Latitude and Longitude to numeric types
data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')

# Filter out rows with invalid latitude or longitude values
data = data.dropna(subset=['Latitude', 'Longitude'])

# Apply the model to the entire DataFrame to predict churn
data['Predicted Churn'] = pipeline.predict(data[features])
data['Churn Probability'] = pipeline.predict_proba(data[features])[:, 1]
data['Predicted Churn'] = data['Predicted Churn'].map({1: 'Yes', 0: 'No'})

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Filter and Display Data", "Insights and Visualizations", "Predict Churn for a New Customer"])

# Section 1: Filter and Display Data
if page == "Filter and Display Data":
    st.title("Filter and Display Data")
    
    # Widget: Filter by predicted churn
    predicted_churn_filter = st.selectbox("Filter by Predicted Churn", ["All", "Yes", "No"])

    # Widget: Filter by churn probability
    churn_probability_range = st.slider("Churn Probability Range", 0.0, 1.0, (0.0, 1.0))

    # Filter data based on widget selections
    filtered_data = data.copy()
    if predicted_churn_filter != "All":
        filtered_data = filtered_data[filtered_data['Predicted Churn'] == predicted_churn_filter]
    filtered_data = filtered_data[(filtered_data['Churn Probability'] >= churn_probability_range[0]) & (filtered_data['Churn Probability'] <= churn_probability_range[1])]

    # Widget: Select number of rows to display
    rows_to_display = st.selectbox("Select number of rows to display", ["All rows", 25, 50, 100])

    # Widget: Select city filter
    cities = ["All cities"] + data['City'].unique().tolist()
    selected_city = st.selectbox("Select City", cities)

    # Widget: Select contract type filter
    contract_types = ["All contract types"] + data['Contract'].unique().tolist()
    selected_contract = st.selectbox("Select Contract Type", contract_types)

    # Widget: Select internet type filter
    internet_types = ["All internet types"] + data['Internet Type'].unique().tolist()
    selected_internet_type = st.selectbox("Select Internet Type", internet_types)

    # Widget: Select payment method filter
    payment_methods = ["All payment methods"] + data['Payment Method'].unique().tolist()
    selected_payment_method = st.selectbox("Select Payment Method", payment_methods)

    # Apply filters based on widget selections
    if selected_city != "All cities":
        filtered_data = filtered_data[filtered_data['City'] == selected_city]
    if selected_contract != "All contract types":
        filtered_data = filtered_data[filtered_data['Contract'] == selected_contract]
    if selected_internet_type != "All internet types":
        filtered_data = filtered_data[filtered_data['Internet Type'] == selected_internet_type]
    if selected_payment_method != "All payment methods":
        filtered_data = filtered_data[filtered_data['Payment Method'] == selected_payment_method]

    # Display the filtered DataFrame
    if rows_to_display != "All rows":
        st.dataframe(filtered_data[['Customer ID'] + features + ['Predicted Churn', 'Churn Probability']].head(rows_to_display))
    else:
        st.dataframe(filtered_data[['Customer ID'] + features + ['Predicted Churn', 'Churn Probability']])

    # Function to convert DataFrame to CSV
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    # Download button for filtered data
    csv = convert_df(filtered_data)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name='filtered_data.csv',
        mime='text/csv',
    )

    # Plot customers on a map with colors based on predicted churn label
    map_data = filtered_data[['Latitude', 'Longitude', 'Predicted Churn']]
    map_data['color'] = map_data['Predicted Churn'].apply(lambda x: [255, 0, 0] if x == 'Yes' else [0, 0, 255])

    # Create a pydeck layer for the map
    layer = pdk.Layer(
        'ScatterplotLayer',
        map_data,
        get_position='[Longitude, Latitude]',
        get_fill_color='color',
        get_radius=200,
        pickable=True,
        auto_highlight=True
    )

    # Determine the initial view state for the map
    initial_view_state = pdk.ViewState(
        latitude=map_data['Latitude'].mean(),
        longitude=map_data['Longitude'].mean(),
        zoom=10,
        pitch=50
    )

    # Create the deck.gl map with a lighter style
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=initial_view_state,
        tooltip={"text": "Churn: {Predicted Churn}\nLatitude: {Latitude}\nLongitude: {Longitude}"},
        map_style='mapbox://styles/mapbox/light-v10'
    )

    # Display the map
    st.pydeck_chart(r)

    # Search by Customer ID section
    st.subheader("Search by Customer ID")
    customer_id = st.text_input("Enter Customer ID to search", value='')

    # Display customer information and prediction if a valid ID is entered
    if customer_id:
        customer_data = data[data['Customer ID'] == customer_id]
        if not customer_data.empty:
            st.write("Customer Information")
            st.write(customer_data[['Customer ID', 'Age', 'City', 'State', 'Monthly Charge', 'Total Charges', 'Predicted Churn', 'Churn Probability']])

            # Display churn prediction
            churn_prediction = customer_data['Predicted Churn'].values[0]
            churn_probability = customer_data['Churn Probability'].values[0]
            st.write('Churn Prediction:', churn_prediction)
            st.write('Churn Probability:', churn_probability)
        else:
            st.write("Customer ID not found.")

# Section 2: Insights and Visualizations
elif page == "Insights and Visualizations":
    st.title("Insights and Visualizations")

    # Widget: Distribution of numerical features
    st.subheader("Distribution of Numerical Features")
    num_feature = st.selectbox("Select a numerical feature", numerical_features)
    fig, ax = plt.subplots()
    sns.histplot(data[num_feature], kde=True, ax=ax)
    st.pyplot(fig)

    # Widget: Relationship between features and churn
    st.subheader("Relationship Between Features and Churn")
    feature = st.selectbox("Select a feature", features)
    if feature in numerical_features:
        fig, ax = plt.subplots()
        sns.boxplot(x=data['Predicted Churn'], y=data[feature], ax=ax)
        st.pyplot(fig)
    elif feature in categorical_features:
        fig, ax = plt.subplots()
        churn_counts = data.groupby([feature, 'Predicted Churn']).size().unstack().fillna(0)
        churn_counts.plot(kind='bar', stacked=True, ax=ax)
        st.pyplot(fig)

    # Correlation matrix section
    st.subheader("Correlation Matrix")
    corr_matrix = data[numerical_features + ['Churn Probability']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Section 3: Predict Churn for a New Customer
elif page == "Predict Churn for a New Customer":
    st.title("Predict Churn for a New Customer")

    # Create input fields for all predictors except 'Customer ID'
    input_data = {}
    for feature in features:
        if feature in categorical_features:
            unique_values = data[feature].unique()
            if len(unique_values) == 2:  # Binary categorical feature
                input_data[feature] = st.radio(f"Select {feature}", options=unique_values)
            else:  # Non-binary categorical feature
                input_data[feature] = st.selectbox(f"Select {feature}", options=unique_values)
        elif feature in ['Zip Code', 'Latitude', 'Longitude']:
            input_data[feature] = st.number_input(f"Enter {feature}", value=float(data[feature].mean()))
        else:  # All other numeric features
            min_value = int(data[feature].min())
            max_value = int(data[feature].max())
            mean_value = int(data[feature].mean())
            input_data[feature] = st.slider(f"Select {feature}", min_value, max_value, mean_value)

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict button for new customer churn prediction
    if st.button("Predict Churn"):
        prediction = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df)[:, 1][0]
        st.write('Churn Prediction:', 'Yes' if prediction[0] == 1 else 'No')
        st.write('Churn Probability:', probability)
