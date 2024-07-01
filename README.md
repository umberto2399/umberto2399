# Receipt and Audio Analyzer

## Overview

The Receipt and Audio Analyzer is a web application designed to help users extract and analyze data from receipts and audio files. The app leverages Google Cloud's Vision and Speech APIs to process images and audio, and OpenAI's GPT model to analyze and categorize the extracted data. Users can visualize expenses over time, filter data by various criteria, and ask questions about their expenses.

## Features

1. **Analyze Receipts**:
   - Upload an image or PDF of a receipt.
   - Extract text using Google Cloud Vision API.
   - Analyze the extracted text to categorize products and save data to a CSV file.

2. **Audio Transcription**:
   - Upload an audio file in .wav format.
   - Transcribe audio using Google Cloud Speech-to-Text API.
   - Analyze the transcribed text to categorize products and save data to a CSV file.

3. **Visualization of Expenses Over Time**:
   - Visualize expenses over time with interactive charts.
   - Filter data by date range and category.
   - Display product purchase frequency with bar charts.

4. **Ask a Question**:
   - Generate summary statistics from the data.
   - Use OpenAI's GPT model to answer user questions about their expenses.
   - Perform advanced searches based on user-defined criteria.

## Files in the Repository

- `app.py`: The main Streamlit application script.
- `csv_app_eng_final.csv`: The file to store extracted receipt data.
- `requirements-3.txt`: A list of Python dependencies required to run the app.
- `Landing page.jpeg`: The landing page image displayed in the app.

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements-3.txt
   ```

4. **Configure API keys**:
   - Open `app.py` and replace `INSERT YOUR OPENAI API KEY HERE` with your actual OpenAI API key.
   - Set the path to your Google Cloud Vision and Speech service key in the `os.environ["GOOGLE_APPLICATION_CREDENTIALS"]` line.

5. **Run the app**:
   ```bash
   streamlit run app.py
   ```

6. **Access the app**:
   - Open your web browser and navigate to `http://localhost:8501`.

## How to Use the App

### Welcome Tab

1. **Introduction**:
   - Provides an overview of the application.
   - Displays an image related to the app.

### Analyze Receipts

1. **Upload an Image or PDF of a Receipt**:
   - Use the file uploader to upload a receipt image or PDF.
   - The extracted text from the receipt will be displayed.

2. **Analyze the Extracted Text**:
   - Click the "Analyze Text" button to analyze the extracted text using OpenAI GPT.
   - The analyzed data will be displayed.

3. **Confirm and Save Data**:
   - The analyzed data is automatically saved to a CSV file.
   - Download the updated CSV file if needed.

4. **Audio Transcription**:
   - Upload an audio file and the app will transcribe it.
   - Click the "Analyze Transcription" button to analyze the transcribed text.
   - The analyzed data is automatically saved to a CSV file.

### Expenses Over Time

1. **Visualize Expenses**:
   - Select a time range from the dropdown menu to filter the data.
   - View the expenses over time with interactive charts.

2. **Category and Product Analysis**:
   - Filter data by category and visualize the purchase frequency of products within that category.

### Ask a Question

1. **Ask a Question About Your Expenses**:
   - Enter a question in the text input box.
   - The app will generate a response based on the summary statistics and the last receipt details.

2. **Advanced Search**:
   - If the initial answer does not satisfy the query, perform an advanced search by specifying date range and category.


## Conclusion

The Receipt and Audio Analyzer provides a comprehensive tool for extracting and analyzing data from receipts and audio files. The app leverages GEN AI to provide detailed insights into expenses, helping users manage their finances more effectively. Further improvements and iterations will continue to enhance the user experience and address any existing challenges.

---
