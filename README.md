## Customer Churn Label Prediction

This project consists of a machine learning model for predicting customer churn and a Streamlit web application to interact with the model. The model is trained on a telecommunications dataset, and the app provides functionalities for filtering and displaying data, visualizing insights, and predicting churn for new customers.

### Features

- **Data Filtering and Display**: Filter data based on various criteria and display it in a tabular format.
- **Insights and Visualizations**: Visualize distributions, relationships between features and churn, and a correlation matrix.
- **Churn Prediction**: Predict churn for a new customer based on input features.
- **Map Visualization**: Display customers on a map, color-coded by predicted churn status.

### Setup and Configuration

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/Churn-Label-Predictor.git
    cd Churn-Label-Predictor
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r re.txt
    ```

3. **Prepare the Dataset**:
    - Place the `telco.csv` dataset in the root directory of the project.

4. **Train the Model**:
    - Run the model training script to train and save the model pipeline.
    ```bash
    python model.py
    ```

5. **Run the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```

### Usage

1. **Filtering and Displaying Data**:
    - Navigate to the "Filter and Display Data" section.
    - Apply filters using the sidebar widgets.
    - View the filtered data in a table and download it as a CSV file.
    - Visualize customer locations on a map.

2. **Insights and Visualizations**:
    - Navigate to the "Insights and Visualizations" section.
    - Select numerical or categorical features to visualize their distribution and relationship with churn.
    - View the correlation matrix of numerical features.

3. **Predict Churn for a New Customer**:
    - Navigate to the "Predict Churn for a New Customer" section.
    - Enter feature values for the new customer.
    - Click "Predict Churn" to see the prediction and probability.

### File Structure

- `app.py`: Main script for running the Streamlit application.
- `train_model.py`: Script for training the machine learning model.
- `requirements.txt`: List of dependencies required for the project.
- `telco.csv`: Dataset for training and evaluating the model.
- `churn_model_pipeline.pkl`: Saved model pipeline for churn prediction.

### Dependencies

- `streamlit`
- `pandas`
- `joblib`
- `pydeck`
- `matplotlib`
- `seaborn`
- `scikit-learn`
