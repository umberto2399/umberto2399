# Prototypes Repository

Welcome to the Prototypes repository! This repository serves as a collection of prototype projects that I am currently working on, primarily using Python as the coding language. These projects leverage two different libraries for building interactive web applications: Streamlit and Dash. Each project in this repository is designed to explore different use cases and functionalities, ranging from data analysis and machine learning to natural language processing (GPT 4o) and computer vision (Google Vision).

## Projects

1. **NBA Trade Advisor**
   - **Description:** An interactive web application built with Streamlit that provides NBA player statistics and allows users to analyze potential trade scenarios. The app uses OpenAI's GPT-4 API to generate insights about the pros and cons of proposed trades.
   - **Features:** 
     - View and filter player stats
     - Add and save custom playing style descriptions
     - Analyze potential trades with AI-generated insights

2. **Grocery Tracker**
   - **Description:** A web application designed to track grocery purchases by uploading pictures of receipts. The app extracts information from the receipts and stores it in a CSV file. There is also a beta feature for extracting information from audio inputs.
   - **Features:**
     - Upload and process receipt images
     - Store product information in a CSV file
     - Beta: Extract information from audio inputs

3. **Predicting Churn Label**
   - **Description:** A Streamlit app aimed at providing telco managers with insights into customer churn. The app includes a pipeline for training a machine learning model to predict churn and offers various data visualizations and insights.
   - **Features:**
     - Train and evaluate churn prediction models
     - Visualize customer data and churn insights
     - Provide actionable insights for reducing churn

4. **Customer Care Chatbot**
   - **Description:** A chatbot application designed to simulate customer care interactions for a company. The chatbot can handle common customer inquiries and provide relevant responses.
   - **Features:**
     - Simulate customer care interactions
     - Provide automated responses to common inquiries
     - Enhance customer service experience

## Technologies and Libraries

- **Python:** The primary programming language used for all projects.
- **Streamlit:** Used for building interactive web applications for the NBA Trade Advisor and Predicting Churn Label projects.
- **Dash:** Used for other interactive applications that may require more advanced functionalities and customizations.

## Installation and Usage

To get started with any of the projects, follow the steps below:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/prototypes.git
   cd prototypes
   ```

2. **Navigate to the project directory:**
   ```bash
   cd <project_name>
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py  # For Streamlit apps
   dash run app.py       # For Dash apps (assuming appropriate command)
   ```

## Contribution

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.
