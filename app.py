import streamlit as st
import os
import pandas as pd
import plotly.express as px
from google.cloud import speech
from google.cloud import vision
from openai import OpenAI
import fitz
from datetime import datetime, timedelta
import base64
import plotly.graph_objects as go
# Configure your OpenAI API key
client = OpenAI(api_key='')

# Set the path to your Google Cloud Vision and Speech service key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

# Initialize the Google Cloud Speech client
speech_client = speech.SpeechClient()

# File to store new receipt data
NEW_DATA_FILE = "csv_app_eng_final.csv"

def load_existing_data(file_path):
    """Load existing data from a CSV file."""
    if os.path.exists(file_path):
        st.write("Loading existing data...")
        return pd.read_csv(file_path)
    else:
        st.write("Creating a new data file...")
        columns = ["Product", "Quantity", "Brand", "Item_Price", "Date", "Supermarket", "Payment_Method", "Category"]
        return pd.DataFrame(columns=columns)
    

data_for_plotting = load_existing_data(NEW_DATA_FILE)

def save_data_to_file(data, file_path):
    """Save data to a CSV file."""
    data.to_csv(file_path, index=False)

def clean_data(data):
    """Clean the data to ensure all item prices are numeric and dates are in the correct format."""
    data['Item_Price'] = pd.to_numeric(data['Item_Price'], errors='coerce')
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y', errors='coerce')
    return data.dropna(subset=['Item_Price', 'Date'])

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
        f"Extract the following data from the receipt or from a recording in a format that makes it easy to create a csv with the corresponding columns to the data you have extracted, ignoring the total price. We are interested in the individual products. "
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
        f"Category: The category of the product must be one of these categories (lower case, choose one of the following values): meat, fish, vegetable, fruit, cheese, bread, bakery, beverage, snack, frozen, pantry, household, personal care, alcohol, other."
        f"Make sure not to change the name of the columns to avoid issues with concatenating new rows in the dataframe."
        f"The output should be similar to this to allow me to save the various products in different rows of my csv using the semicolon to separate the products:"
        f"Product: cookies, Quantity: 1, Brand: misura, Item_Price: 2.50, Date: 21/05/24, Supermarket: carrefour, Payment_Method: credit card ***1234, Category: snack; "
        f"Product: chicken, Quantity: 1, Brand: aia, Item_Price: 2.50, Date: 21/05/24, Supermarket: carrefour, Payment_Method: credit card ***1234, Category: meat"
        f"\n\nText:\n{text}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a grocery store cashier. A customer hands you a receipt. You need to extract the following data from the receipt: Product, Quantity, Brand, Item_Price, Date, Supermarket, Payment_Method, Category."},
            {"role": "user", "content": prompt}
        ]
    )
    data = response.choices[0].message.content.strip()
    return data

def get_summary_stats(data):
    """Generate summary statistics from the data."""
    summary = {}

    # Clean data
    data['Item_Price'] = pd.to_numeric(data['Item_Price'], errors='coerce')
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y', errors='coerce')
    data = data.dropna(subset=['Item_Price', 'Date'])

    # Total spending by different payment methods
    payment_summary = data.groupby('Payment_Method')['Item_Price'].sum().to_dict()
    summary['Total spent by payment methods'] = payment_summary

    # Total spending by supermarket
    supermarket_summary = data.groupby('Supermarket')['Item_Price'].sum().to_dict()
    summary['Total spent by supermarket'] = supermarket_summary

    # Total spending by category
    category_summary = data.groupby('Category')['Item_Price'].sum().to_dict()
    summary['Total spent by category'] = category_summary

    # Number of visits using date and supermarket
    num_visits = data.groupby(['Date', 'Supermarket']).size().to_dict()
    summary['Number of visits using date and supermarket'] = num_visits

    # Most visited supermarket
    most_visited_supermarket = data['Supermarket'].mode()[0] if not data['Supermarket'].mode().empty else "N/A"
    summary['Most visited supermarket'] = most_visited_supermarket

    # Product bought the most
    most_bought_product = data['Product'].mode()[0] if not data['Product'].mode().empty else "N/A"
    summary['Most bought product'] = most_bought_product

    # Brand bought the most
    most_bought_brand = data['Brand'].mode()[0] if not data['Brand'].mode().empty else "N/A"
    summary['Most bought brand'] = most_bought_brand

    # Average spending per visit using date and item price
    avg_spending_per_visit = data.groupby('Date')['Item_Price'].mean().to_dict()
    summary['Average spending per visit using date'] = avg_spending_per_visit

    # Total number of products purchased
    total_products = data['Product'].nunique()
    summary['Total number of products purchased'] = total_products

    # Most popular product category
    most_popular_category = data['Category'].mode()[0] if not data['Category'].mode().empty else "N/A"
    summary['Most popular product category'] = most_popular_category

    # Total number of products purchased in each category
    category_counts = data['Category'].value_counts().to_dict()
    summary['Total number of products purchased in each category'] = category_counts
    
    # Total spending
    total_spending = data['Item_Price'].sum()
    summary['Total spending'] = total_spending

    # Total spending last day/week/month
    now = datetime.now()
    last_day = data[data['Date'] > now - timedelta(days=1)]['Item_Price'].sum()
    last_week = data[data['Date'] > now - timedelta(days=7)]['Item_Price'].sum()
    last_month = data[data['Date'] > now - timedelta(days=30)]['Item_Price'].sum()
    summary['Total spent last day'] = last_day
    summary['Total spent last week'] = last_week
    summary['Total spent last month'] = last_month

    # Spending by category last month
    start_of_last_month = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
    end_of_last_month = now.replace(day=1) - timedelta(days=1)
    last_month_data = data[(data['Date'] >= start_of_last_month) & (data['Date'] <= end_of_last_month)]
    last_month_category_spending = last_month_data.groupby('Category')['Item_Price'].sum().to_dict()
    summary['Last month spent by category'] = last_month_category_spending

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
        f"If you do not have enough information to answer the question, the answer must be exactly like this: 'Sorry, I don't understand the question.'"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data analyst. A user asks you a question about their grocery purchases. Use the provided summary statistics and last receipt details to answer the question."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content.strip()
    if "Sorry, I don't understand the question" in answer:
        return "Sorry, I don't understand the question."
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


def get_spending_in_date_range(data, start_date, end_date):
    """Calculate total spending in a specific date range."""
    date_range_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    return date_range_data['Item_Price'].sum()

def get_spending_on_product(data, product_name):
    """Calculate total spending on a specific product."""
    product_data = data[data['Product'].str.contains(product_name, case=False, na=False)]
    return product_data['Item_Price'].sum()

def get_detailed_breakdown(data, start_date, end_date):
    """Get a detailed breakdown of spending in a specific date range."""
    breakdown_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    return breakdown_data

def compare_spending_current_vs_last_month(data):
    """Compare spending between the current month and the previous month."""
    now = datetime.now()
    current_month = now.month
    current_year = now.year
    previous_month = current_month - 1 if current_month > 1 else 12
    previous_month_year = current_year if current_month > 1 else current_year - 1

    current_month_spending = data[(data['Date'].dt.month == current_month) & (data['Date'].dt.year == current_year)]['Item_Price'].sum()
    previous_month_spending = data[(data['Date'].dt.month == previous_month) & (data['Date'].dt.year == previous_month_year)]['Item_Price'].sum()
    
    return current_month_spending, previous_month_spending

def get_average_daily_spending_current_month(data):
    """Calculate average daily spending for the current month."""
    now = datetime.now()
    current_month_data = data[(data['Date'].dt.month == now.month) & (data['Date'].dt.year == now.year)]
    days_in_month = now.day
    return current_month_data['Item_Price'].sum() / days_in_month

def get_average_daily_spending_last_month(data):
    """Calculate average daily spending for the last month."""
    now = datetime.now()
    first_day_last_month = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
    last_day_last_month = first_day_last_month + timedelta(days=32)
    last_month_data = data[(data['Date'] >= first_day_last_month) & (data['Date'] < last_day_last_month)]
    days_in_last_month = (last_day_last_month - first_day_last_month).days
    return last_month_data['Item_Price'].sum() / days_in_last_month


# Streamlit app
st.title("Receipt and Audio Analyzer")

tab0, tab1, tab2, tab3 = st.tabs(["Welcome", "Analyze Receipts", "Expenses Over Time", "Ask a Question"])


with tab0:
    st.header("Welcome to the Receipt and Audio Analyzer")
    
    # Display the image with a specified width
    st.image("/Users/umbertocirilli/Downloads/GROUP PRO/Landing page.jpeg", caption="Landing Page Image", width=600)
    
    # Center the text and add some styling
    st.markdown(
        """
        <style>
        .intro-text {
            text-align: center;
            margin-top: 20px;
            margin-left: auto;
            margin-right: auto;
            max-width: 800px;
        }
        .intro-text p, .intro-text ul {
            font-size: 1.2em;
            line-height: 1.6em;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="intro-text">
        <p>
        This application allows you to analyze your grocery receipts and audio recordings to track your expenses. 
        You can upload images or PDFs of your receipts, or even audio files, and the app will extract and save the data for you. 
        Additionally, you can visualize your expenses over time and get detailed insights into your spending habits. 
        Use the tabs to navigate through different functionalities:
        </p>
        <ul>
            <li><strong>Analyze Receipts:</strong> Upload and analyze your receipts.</li>
            <li><strong>Expenses Over Time:</strong> Visualize your spending trends and categorize your expenses.</li>
            <li><strong>Ask a Question:</strong> Get insights based on your spending data.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True
    )

with tab1:
    st.header("Analyze Receipts")
    
    # Image and PDF Upload
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
            
            # Save the extracted data immediately
            new_data = parse_analyzed_data(analyzed_text)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            save_data_to_file(updated_data, NEW_DATA_FILE)
            
            st.write("Data saved successfully!")
            csv = updated_data.to_csv(index=False)
            st.download_button(label="Download data as CSV", data=csv, file_name='new_receipt_data.csv', mime='text/csv')

    # Audio upload and transcription
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
            
            # Save the extracted data immediately
            new_data = parse_analyzed_data(analyzed_text)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            save_data_to_file(updated_data, NEW_DATA_FILE)
            
            st.write("Data saved successfully!")
            csv = updated_data.to_csv(index=False)
            st.download_button(label="Download data as CSV", data=csv, file_name='new_receipt_data.csv', mime='text/csv')

with tab2:
    st.header("Expenses Over Time")
    data_for_plotting = load_existing_data(NEW_DATA_FILE)

    if not data_for_plotting.empty:
        if 'Date' in data_for_plotting.columns:
            data_for_plotting['Date'] = pd.to_datetime(data_for_plotting['Date'], dayfirst=True, errors='coerce')
            # Filter out dates before 2024
            data_for_plotting = data_for_plotting[data_for_plotting['Date'] >= '2024-01-01']
        else:
            st.write("The 'Date' column is missing in the data.")
        
        if 'Item_Price' in data_for_plotting.columns:
            data_for_plotting['Item_Price'] = data_for_plotting['Item_Price'].astype(str).str.replace(',', '.').astype(float)

            st.subheader("Visualization of expenses over time")
            
            now = datetime.now()

            # Calculate total amount spent during the last month
            last_month_data = data_for_plotting[data_for_plotting['Date'] > now - timedelta(days=30)]
            total_spent_last_month = last_month_data['Item_Price'].sum()

            # Calculate total amount spent during the previous month
            previous_month_data = data_for_plotting[(data_for_plotting['Date'] > now - timedelta(days=60)) & 
                                                    (data_for_plotting['Date'] <= now - timedelta(days=30))]
            total_spent_previous_month = previous_month_data['Item_Price'].sum()

            # Calculate average spent per day during the last month
            if not last_month_data.empty:
                avg_spent_per_day = total_spent_last_month / last_month_data['Date'].nunique()
            else:
                avg_spent_per_day = 0

            # Calculate average spent per day during the previous month
            if not previous_month_data.empty:
                avg_spent_per_day_previous = total_spent_previous_month / previous_month_data['Date'].nunique()
            else:
                avg_spent_per_day_previous = 0

            # Determine arrow direction and color for delta
            delta_color_total = "red" if total_spent_last_month > total_spent_previous_month else "green"
            delta_symbol_total = "↑" if total_spent_last_month > total_spent_previous_month else "↓"

            delta_color_avg = "red" if avg_spent_per_day > avg_spent_per_day_previous else "green"
            delta_symbol_avg = "↑" if avg_spent_per_day > avg_spent_per_day_previous else "↓"

            # Create circular progress bar for total amount spent
            fig_total_spent = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = total_spent_last_month,
                delta = {'reference': total_spent_previous_month, 'position': "top", 'relative': False,
                         'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                title = {'text': "Total Spent Last Month (€)"},
                gauge = {'axis': {'range': [None, max(total_spent_last_month, total_spent_previous_month) * 1.5]},
                         'bar': {'color': "green"}},
                domain={'x': [0, 1], 'y': [0, 1]}))

            # Create circular progress bar for average spent per day
            fig_avg_spent = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = avg_spent_per_day,
                delta = {'reference': avg_spent_per_day_previous, 'position': "top", 'relative': False,
                         'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                title = {'text': "Average Spent Per Day (€)"},
                gauge = {'axis': {'range': [None, max(avg_spent_per_day, avg_spent_per_day_previous) * 1.5]},
                         'bar': {'color': "blue"}},
                domain={'x': [0, 1], 'y': [0, 1]}))

            # Display the two circular progress bars one after the other
            st.plotly_chart(fig_total_spent, use_container_width=True)
            st.plotly_chart(fig_avg_spent, use_container_width=True)

            # Dropdown menu for time range selection
            time_filter = st.selectbox("Select Time Range", ["All", "Last Day", "Last Week", "Last Month"], index=3)

            if time_filter == "Last Day":
                filtered_data = data_for_plotting[data_for_plotting['Date'] > now - timedelta(days=1)]
            elif time_filter == "Last Week":
                filtered_data = data_for_plotting[data_for_plotting['Date'] > now - timedelta(days=7)]
            elif time_filter == "Last Month":
                filtered_data = data_for_plotting[data_for_plotting['Date'] > now - timedelta(days=30)]
            else:
                filtered_data = data_for_plotting

            # Area chart for expenses over time
            fig = px.area(filtered_data, x='Date', y='Item_Price', title="Expenses Over Time",
                          labels={"Item_Price": "Item Price (€)", "Date": "Date"},
                          template="plotly_white")
            fig.update_traces(line=dict(color='royalblue', width=2), fillcolor='lightgreen')
            fig.update_layout(title_font=dict(size=20, color='darkblue'),
                              xaxis_title_font=dict(size=14, color='darkblue'),
                              yaxis_title_font=dict(size=14, color='darkblue'))
            
            st.plotly_chart(fig)

            # Bar chart for product categories
            st.subheader("Product Purchase Frequency by Category")
            
            def plot_category_bar_chart(data):
                category_counts = data['Category'].value_counts().reset_index()
                category_counts.columns = ['Category', 'Count']
                bar_fig = px.bar(category_counts, x='Category', y='Count', title="Product Purchase Frequency by Category",
                                 labels={"Category": "Category", "Count": "Count"},
                                 template="plotly_white")
                bar_fig.update_traces(marker_color='royalblue')
                bar_fig.update_layout(title_font=dict(size=20, color='darkblue'),
                                      xaxis_title_font=dict(size=14, color='darkblue'),
                                      yaxis_title_font=dict(size=14, color='darkblue'))
                return bar_fig
            
            def plot_product_bar_chart(data, category):
                product_counts = data[data['Category'] == category]['Product'].value_counts().reset_index()
                product_counts.columns = ['Product', 'Count']
                # Merge with price data to include price in the hover tooltip
                product_price_data = data[['Product', 'Item_Price']].drop_duplicates()
                product_counts = product_counts.merge(product_price_data, on='Product', how='left')
                bar_fig = px.bar(product_counts, x='Product', y='Count', 
                                 hover_data={'Item_Price': True}, 
                                 title=f"Product Purchase Frequency in {category} Category",
                                 labels={"Product": "Product", "Count": "Count", "Item_Price": "Price (€)"},
                                 template="plotly_white")
                bar_fig.update_traces(marker_color='royalblue')
                bar_fig.update_layout(title_font=dict(size=20, color='darkblue'),
                                      xaxis_title_font=dict(size=14, color='darkblue'),
                                      yaxis_title_font=dict(size=14, color='darkblue'))
                return bar_fig

            if 'Category' in filtered_data.columns:
                selected_category = st.selectbox("Select Category", ["All"] + sorted(filtered_data['Category'].dropna().unique().tolist()), index=0)

                if selected_category == "All":
                    category_fig = plot_category_bar_chart(filtered_data)
                    st.plotly_chart(category_fig)
                else:
                    product_fig = plot_product_bar_chart(filtered_data, selected_category)
                    st.plotly_chart(product_fig)
                    if st.button("Back to Category Bar Chart"):
                        category_fig = plot_category_bar_chart(filtered_data)
                        st.plotly_chart(category_fig)
# Tab 3: Ask a Question
with tab3:
    st.header("Ask a Question")

    # Load existing data
    existing_data = load_existing_data(NEW_DATA_FILE)
    existing_data = clean_data(existing_data)

    if not existing_data.empty:
        summary, last_receipt = get_summary_stats(existing_data)

        question = st.text_input("Ask a question about your expenses")
        answer = ""
        advanced_search_needed = False

        if question:
            answer = generate_response(summary, last_receipt, question)

            if answer == "Sorry, I don't understand the question.":
                advanced_search_needed = True

            st.write(answer)

            if advanced_search_needed:
                if st.button("Advanced Search"):
                    st.subheader("Advanced Search")
                    start_date = st.date_input("Start date")
                    end_date = st.date_input("End date")
                    category = st.selectbox("Category", options=["All"] + list(existing_data['Category'].unique()))

                    if st.button("Submit"):
                        if start_date and end_date:
                            start_date = pd.to_datetime(start_date)
                            end_date = pd.to_datetime(end_date)
                            if category == "All":
                                advanced_answer = get_spending_in_date_range(existing_data, start_date, end_date)
                            else:
                                filtered_data = existing_data[existing_data['Category'] == category]
                                advanced_answer = get_spending_in_date_range(filtered_data, start_date, end_date)
                            st.write(f"Total spending from {start_date.date()} to {end_date.date()} in {category} category: {advanced_answer}")
    else:
        st.write("No data available to analyze. Please upload receipts in the 'Analyze Receipts' tab.")
