import streamlit as st
import os
import pandas as pd
import plotly.express as px
from google.cloud import speech
from google.cloud import vision
from openai import OpenAI
import fitz
from datetime import datetime, timedelta

# Configure your OpenAI API key
client = OpenAI(api_key='')

# Set the path to your Google Cloud Vision and Speech service key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

# Initialize the Google Cloud Speech client
speech_client = speech.SpeechClient()

# File to store new receipt data
NEW_DATA_FILE = "csv_app_eng_fin.csv"

def load_existing_data(file_path):
    """Load existing data from a CSV file."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        columns = ["Product", "Quantity", "Brand", "Item_Price", "Date", "Supermarket", "Payment_Method", "Product_Category"]
        return pd.DataFrame(columns=columns)

def save_data_to_file(data, file_path):
    """Save data to a CSV file."""
    data.to_csv(file_path, index=False)

def transcribe_audio(file_path):
    """Transcribe audio file using Google Cloud Speech-to-Text API."""
    with open(file_path, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code="en-US",
    )
    response = speech_client.recognize(config=config, audio=audio)
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript + " "
    return transcript

def extract_text_from_image(image):
    """Extract text from an uploaded image using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()
    content = image.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if response.error.message:
        raise Exception(f'{response.error.message}')
    return texts[0].description if texts else ""

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyMuPDF (Fitz)."""
    pdf_document = fitz.open(pdf_file)
    extracted_text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        extracted_text += page.get_text()
    return extracted_text

def analyze_text(text):
    prompt = (
        f"Extract the following data from the receipt in a format that makes it easy to create a csv with the corresponding columns to the data you have extracted, ignoring the total price. We are interested in the individual products. "
        f"If you cannot find some data, leave the field empty."
        f"If there are multiple products, separate each product with a semicolon (;)."
        f"These are the columns we are interested in:"
        f"[Product, Quantity, Brand, Item_Price, Date, Supermarket, Payment_Method, Category]."
        f"Here are some explanations for each column to help you understand what we are looking for:"
        f"Product: The name of the purchased product in lowercase, e.g., water."
        f"Quantity: The number of units of the purchased product, e.g., 2."
        f"Brand: The brand name of the product in lowercase, e.g., san benedetto."
        f"Item_Price: The price of a single item using the dot as the decimal separator, e.g., 1.50."
        f"Date: The date of the receipt in the format dd/mm/yy, e.g., 21/05/24."
        f"Supermarket: The name of the supermarket in lowercase, e.g., carrefour."
        f"Payment_Method: The payment method used in lowercase, e.g., credit card ***1234."
        f"Category: The category of the product must be one of these categories (lower case, choose one of the following values): meat, fish, vegetable, fruit, cheese, bread, bakery, beverage, snack, frozen, pantry, household, personal care,alcohol, other."
        f"Make sure not to change the name of the columns to avoid issues with concatenating new rows in the dataframe."
        f"The output should be similar to this to allow me to save the various products in different rows of my csv using the semicolon to separate the products:"
        f"Product: cookies, Quantity: 1, Brand: misura, Item_Price: 2.50, Date: 21/05/24, Supermarket: carrefour, Payment_Method: credit card ***1234, Category: snack; "
        f"Product: chicken, Quantity: 1, Brand: aia, Item_Price: 2.50, Date: 21/05/24, Supermarket: carrefour, Payment_Method: credit card ***1234, Category: meat"
        f"\n\nText:\n{text}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a grocery store cashier. A customer hands you a receipt. You need to extract the following data from the receipt: Product, Quantity, Brand, Item_Price, Date, Supermarket, Payment_Method."},
            {"role": "user", "content": prompt}
        ]
    )
    data = response.choices[0].message.content.strip()
    return data

def get_summary_stats(data):
    """Generate summary statistics from the data."""
    summary = {}

    # Total spending by different payment methods
    payment_summary = data.groupby('Payment_Method')['Item_Price'].sum().to_dict()
    summary['Total spent by payment methods'] = payment_summary

    # Most visited supermarket
    most_visited_supermarket = data['Supermarket'].mode()[0] if not data['Supermarket'].mode().empty else "N/A"
    summary['Most visited supermarket'] = most_visited_supermarket

    # Product bought the most
    most_bought_product = data['Product'].mode()[0] if not data['Product'].mode().empty else "N/A"
    summary['Most bought product'] = most_bought_product

    # Brand bought the most
    most_bought_brand = data['Brand'].mode()[0] if not data['Brand'].mode().empty else "N/A"
    summary['Most bought brand'] = most_bought_brand

    # Average spending per visit
    avg_spending = data['Item_Price'].mean()
    summary['Average spending per visit'] = avg_spending

    # Total number of visits
    total_visits = data.shape[0]
    summary['Total number of visits'] = total_visits

    # Total spending
    total_spending = data['Item_Price'].sum()

    # Total spending last day/week/month
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y', errors='coerce')
    now = datetime.now()
    last_day = data[data['Date'] > now - timedelta(days=1)]['Item_Price'].sum()
    last_week = data[data['Date'] > now - timedelta(days=7)]['Item_Price'].sum()
    last_month = data[data['Date'] > now - timedelta(days=30)]['Item_Price'].sum()
    summary['Total spent last day'] = last_day
    summary['Total spent last week'] = last_week
    summary['Total spent last month'] = last_month

    # Last receipt details
    last_receipt = data.iloc[-1].to_dict() if not data.empty else {}

    return summary, last_receipt

def generate_response(summary, last_receipt, question):
    """Generate response to the user's question based on summary statistics."""
    prompt = (
        f"You have the following summary statistics and the details of the last receipt from a CSV file of grocery purchases. "
        f"Use this information to answer the user's question."
        f"\n\nSummary statistics:\n{summary}"
        f"\n\nLast receipt details:\n{last_receipt}"
        f"\n\nQuestion:\n{question}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data analyst. A user asks you a question about their grocery purchases. Use the provided summary statistics and last receipt details to answer the question."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content.strip()
    return answer

def parse_analyzed_data(analyzed_text):
    """Parse the extracted data into a structured format."""
    data_list = []
    product_entries = analyzed_text.split(';')
    for entry in product_entries:
        fields = entry.split(', ')
        data = {}
        for field in fields:
            if ': ' in field:
                key, value = field.split(': ', 1)
                data[key.strip()] = value.strip()
        if data:
            data_list.append(data)
    return pd.DataFrame(data_list)

# Streamlit app
st.title("Receipt and Audio Analyzer")

# Analyze Receipts
st.header("Analyze Receipts")
uploaded_file = st.file_uploader("Upload an image or PDF of the receipt", type=["jpg", "jpeg", "png", "pdf"])

# Load existing data
existing_data = load_existing_data(NEW_DATA_FILE)

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_image(uploaded_file)
    st.text_area("Extracted Text", text, height=200)
    if st.button("Analyze Text"):
        analyzed_text = analyze_text(text)
        st.text_area("Extracted Data", analyzed_text, height=200)
        
        if st.button("Confirm and Save Data"):
            new_data = parse_analyzed_data(analyzed_text)
            
            # Concatenate the new data with existing data
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Save the updated data to the CSV file
            save_data_to_file(updated_data, NEW_DATA_FILE)
            
            st.write("Data saved successfully!")
            csv = updated_data.to_csv(index=False)
            st.download_button(label="Download data as CSV", data=csv, file_name='new_receipt_data.csv', mime='text/csv')

# Audio Transcription
st.header("Audio Transcription")
uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "m4a"])

if uploaded_audio is not None:
    st.write("Audio file uploaded. Extracting transcription...")
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_audio.getbuffer())
    audio_file = "uploaded_audio.wav"
    
    transcript = transcribe_audio(audio_file)
    st.text_area("Audio Transcription", transcript, height=200)
    if st.button("Analyze Transcription"):
        analyzed_text = analyze_text(transcript)
        st.text_area("Extracted Data from Transcription", analyzed_text, height=200)
        
        if st.button("Confirm and Save Transcription Data"):
            new_data = parse_analyzed_data(analyzed_text)
            
            # Concatenate the new data with existing data
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Save the updated data to the CSV file
            save_data_to_file(updated_data, NEW_DATA_FILE)
            
            st.write("Data saved successfully!")
            csv = updated_data.to_csv(index=False)
            st.download_button(label="Download data as CSV", data=csv, file_name='new_receipt_data.csv', mime='text/csv')

# Expenses Over Time
st.header("Expenses Over Time")
data_for_plotting = load_existing_data(NEW_DATA_FILE)

if not data_for_plotting.empty:
    if 'Date' in data_for_plotting.columns:
        data_for_plotting['Date'] = pd.to_datetime(data_for_plotting['Date'], dayfirst=True, errors='coerce')
    else:
        st.write("The 'Date' column is missing in the data.")
    
    if 'Item_Price' in data_for_plotting.columns:
        data_for_plotting['Item_Price'] = data_for_plotting['Item_Price'].astype(str).str.replace(',', '.').astype(float)
        
        st.subheader("Visualization of expenses over time")
        
        # Dropdown menu for time range selection
        time_filter = st.selectbox("Select Time Range", ["All", "Last Day", "Last Week", "Last Month"], index=0)
        
        now = datetime.now()
        if time_filter == "Last Day":
            filtered_data = data_for_plotting[data_for_plotting['Date'] > now - timedelta(days=1)]
        elif time_filter == "Last Week":
            filtered_data = data_for_plotting[data_for_plotting['Date'] > now - timedelta(days=7)]
        elif time_filter == "Last Month":
            filtered_data = data_for_plotting[data_for_plotting['Date'] > now - timedelta(days=30)]
        else:
            filtered_data = data_for_plotting
        
        fig = px.line(filtered_data, x='Date', y='Item_Price', title="Expenses Over Time",
                      labels={"Item_Price": "Item Price (€)", "Date": "Date"},
                      template="plotly_white")
        fig.update_traces(line=dict(color='royalblue', width=2))
        fig.update_layout(title_font=dict(size=20, color='darkblue'),
                          xaxis_title_font=dict(size=14, color='darkblue'),
                          yaxis_title_font=dict(size=14, color='darkblue'))
        
        st.plotly_chart(fig)
        
        st.subheader("Filter by Date")
        start_date = st.date_input("Start Date", value=filtered_data['Date'].min())
        end_date = st.date_input("End Date", value=filtered_data['Date'].max())
        filtered_data = filtered_data[(filtered_data['Date'] >= pd.to_datetime(start_date)) & (filtered_data['Date'] <= pd.to_datetime(end_date))]
        st.write("Filtered Data:")
        st.dataframe(filtered_data)
        filtered_csv = filtered_data.to_csv(index=False)
        st.download_button(label="Download filtered data as CSV", data=filtered_csv, file_name='filtered_data.csv', mime='text/csv')

        # Dropdown menu for supermarket selection
        st.subheader("Filter by Supermarket")
        supermarkets = ["All"] + sorted(data_for_plotting['Supermarket'].dropna().unique().tolist())
        selected_supermarket = st.selectbox("Select Supermarket", supermarkets, index=0)

        if selected_supermarket != "All":
            filtered_data = filtered_data[filtered_data['Supermarket'] == selected_supermarket]

        # Bar plot for product frequency
        st.subheader("Product Purchase Frequency")
        product_counts = filtered_data['Product'].value_counts().reset_index()
        product_counts.columns = ['Product', 'Count']
        bar_fig = px.bar(product_counts, x='Product', y='Count', title="Product Purchase Frequency",
                         labels={"Product": "Product", "Count": "Count"},
                         template="plotly_white")
        bar_fig.update_traces(marker_color='royalblue')
        bar_fig.update_layout(title_font=dict(size=20, color='darkblue'),
                              xaxis_title_font=dict(size=14, color='darkblue'),
                              yaxis_title_font=dict(size=14, color='darkblue'))

        st.plotly_chart(bar_fig)

# Ask a Question
st.header("Ask a Question")

if not existing_data.empty:
    summary, last_receipt = get_summary_stats(existing_data)

    question = st.text_input("Ask a question about your expenses")
    if question:
        answer = generate_response(summary, last_receipt, question)
        st.write(answer)
else:
    st.write("No data available to analyze. Please upload receipts in the 'Analyze Receipts' section.")
