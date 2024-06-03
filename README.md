# README

## Receipt and Audio Analyzer

This project is a Streamlit web application that allows users to upload receipts in image or PDF format and audio files for automatic text extraction, analysis, and data storage. The app leverages various APIs and libraries to achieve its functionalities, including Google Cloud Vision, Google Cloud Speech-to-Text, and OpenAI.

### Features

- **Upload Receipts**: Users can upload images or PDF files of receipts.
- **Text Extraction**: Extract text from the uploaded receipts using Google Cloud Vision API.
- **Audio Transcription**: Transcribe audio files using Google Cloud Speech-to-Text API.
- **Data Analysis**: Analyze extracted text and transcriptions to identify and structure relevant information.
- **Data Storage**: Save extracted data into a CSV file for easy access and download.
- **Visualization**: Display and filter expenses over time using interactive plots.

### Setup and Configuration

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/Grocery-Tracker.git
    cd Grocery-Tracker
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure API Keys**:
    - **OpenAI**: Set your OpenAI API key in the script.
      ```python
      client = OpenAI(api_key='YOUR API KEY HERE')
      ```
    - **Google Cloud Services**: Set the path to your Google Cloud Vision and Speech service key.
      ```python
      os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "PATH TO YOUR GOOGLE CLOUD SERVICE KEY HERE"
      ```

4. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

### Usage

1. **Uploading Receipts**:
    - Click the "Carica un'immagine o un file PDF dello scontrino" button.
    - Select the receipt file to upload.
    - Extracted text will be displayed in a text area.
    - Click "Analizza Testo" to process the extracted text and save the data to a CSV file.
    - Download the updated CSV file using the provided download button.

2. **Visualizing Expenses**:
    - View expenses over time in an interactive line chart.
    - Filter data by selecting a date range.
    - Download the filtered data as a CSV file.

3. **Uploading and Transcribing Audio**:
    - Click the "Carica un file audio" button.
    - Select the audio file to upload.
    - The transcribed text will be displayed in a text area.
    - Click "Analizza Trascrizione" to process the transcription and save the data to a CSV file.
    - Download the updated CSV file using the provided download button.

### File Structure

- `app.py`: Main script for running the Streamlit application.
- `requirements.txt`: List of dependencies required for the project.
- `new_receipt_data.csv`: CSV file to store extracted data from receipts and transcriptions.

### Dependencies

- `streamlit`: For building the web application.
- `pandas`: For data manipulation and storage.
- `plotly`: For data visualization.
- `google-cloud-speech`: For speech-to-text capabilities.
- `google-cloud-vision`: For text extraction from images.
- `openai`: For text analysis and processing.
- `pydub`: For audio file handling.
