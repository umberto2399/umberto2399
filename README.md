# Receipt and Audio Analyzer

## Overview

The **Receipt and Audio Analyzer** application is designed to help users analyze their grocery receipts and audio recordings to track their expenses. This tool allows users to upload images or PDFs of their receipts, as well as audio files, and automatically extracts and saves the data. Users can visualize their expenses over time and get detailed insights into their spending habits.

## Features

- **Analyze Receipts**: Upload images, PDFs, or audio recordings of receipts to extract and save the data.
- **Expenses Over Time**: Visualize spending trends and categorize expenses.
- **Ask a Question**: Get insights based on spending data through a question-answer interface.

## Requirements

- Python 3.x
- Streamlit
- pandas
- plotly
- google-cloud-speech
- google-cloud-vision
- openai
- PyMuPDF (fitz)

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/receipt-audio-analyzer.git
    cd receipt-audio-analyzer
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your Google Cloud Vision and Speech service key:
    - Download your Google Cloud service account key JSON file and save it in a secure location.
    - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your JSON key file.
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google-cloud-service-key.json"
    ```

4. Configure your OpenAI API key:
    - Set your OpenAI API key in the `OpenAI` client initialization.

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the tabs to navigate through the functionalities:
    - **Welcome**: Introduction and instructions on how to use the app.
    - **Analyze Receipts**: Upload and analyze your receipts.
    - **Expenses Over Time**: Visualize your spending trends.
    - **Ask a Question**: Get insights based on your spending data.

## Code Structure

- `app.py`: Main Streamlit application file.
- `requirements.txt`: List of required Python packages.
- `data/`: Directory to store CSV files and other data.
- `assets/`: Directory for images and other static assets.

## Example Questions

- **General Spending Questions**:
    - "What is my total spending for the current month?"
    - "How much did I spend last month?"
    - "What is the total amount spent on meat this month?"

- **Comparison Questions**:
    - "How does my spending this month compare to last month?"
    - "What is my average daily spending for the current month?"

- **Most Frequent Purchases**:
    - "Which product did I buy the most last month?"
    - "Which supermarket do I visit most often?"

- **Specific Date Range Analysis**:
    - "What is my total spending from 1st to 15th of this month?"
    - "How much did I spend on dairy products between 10th and 20th of last month?"

- **Payment Method Insights**:
    - "What is the total amount spent using credit cards this month?"
    - "How much did I spend using cash in the last month?"

- **Expense Breakdown**:
    - "Can you provide a detailed breakdown of my spending for the last week?"
    - "What are the top 5 most expensive items I bought this month?"

- **Visualization-Driven Questions**:
    - "Show me a graph of my spending over the last month."
    - "Can you display a bar chart of my spending by category for this month?"

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
