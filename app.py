import streamlit as st
import os
import pandas as pd
import plotly.express as px
from google.cloud import speech
from google.cloud import vision
from openai import OpenAI
from pydub import AudioSegment

# Configure your OpenAI API key
client = OpenAI(api_key='YOUR API KEY HERE')

# Set the path to your Google Cloud Vision and Speech service key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "PATTERN TO YOUR GOOGLE CLOUD SERVICE KEY HERE"

# Initialize the Google Cloud Speech client
speech_client = speech.SpeechClient()

# File to store new receipt data
NEW_DATA_FILE = "new_receipt_data.csv"

def load_existing_data(file_path):
    """Load existing data from a CSV file."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=["Prodotto", "Quantità", "Marca", "Prezzo_Articolo", "Data", "Supermercato", "Metodo_Pagamento"])

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
        language_code="it-IT",
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

def analyze_text(text):
    prompt = (
        f"Estrai i seguenti dati dallo scontrino in un formato tale che sia facile creare un csv con le colonne corrispondenti ai dati che hai estratto ignorando il prezzo totale, siamo interessati ai singoli prodotti. "
        f"Se non dovessi trovare qualche dato, lascia il campo vuoto."
        f"Se ci sono più prodotti, separa ogni prodotto con un punto e virgola (;)."
        f"Queste sono le colonne che ci interessano:"
        f"[Prodotto, Quantità, Marca, Prezzo_Articolo, Data, Supermercato, Metodo_Pagamento]."
        f"Seguono delle spiegazioni per ogni colonna per aiutarti a capire cosa cerco:"
        f"Prodotto: Il nome del prodotto acquistato tutto minuscolo, esempio: acqua."
        f"Quantità: Il numero di unità del prodotto acquistate, esempio: 2."
        f"Marca: Il nome del prodotto tutto minuscolo, esempio: san benedetto."
        f"Prezzo_Articolo: Il prezzo di un singolo articolo utilizzando il punto come sepratore decimale, esempio: 1.50."
        f"Data: La data dello scontrino nel formato gg/mm/aa, esempio: 21/05/24."
        f"Supermercato: Il nome del supermercato tutto minuscolo, esempio: carrefour."
        f"Metodo_Pagamento: Il metodo di pagamento utilizzato tutto minuscolo, esempio: carta di credito ***1234."
        f"Fai attenzione a non modificare il nome delle colonne per non causare problemi con il concatenamento di nuove righe nel dataframe."
        f"L'output deve essere simile a questo per permettermi di salvare i vari prodotti in righe differenti del mio csv usando il punto e virgola per separare i prodotti:"
        f"Prodotto: biscotti, Quantità: 1, Marca: misura, Prezzo_Articolo: 2.50, Data: 21/05/24, Supermercato: carrefour, Metodo_Pagamento: carta di credito ***1234; "
        f"Prodotto: pollo, Quantità: 1, Marca: aia, Prezzo_Articolo: 2.50, Data: 21/05/24, Supermercato: carrefour, Metodo_Pagamento: carta di credito ***1234."
        f"\n\nTesto:\n{text}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a grocery store cashier. A customer hands you a receipt. You need to extract the following data from the receipt: Prodotto, Quantità, Marca, Prezzo_Articolo, Data, Supermercato, Metodo_Pagamento."},
            {"role": "user", "content": prompt}
        ]
    )
    data = response.choices[0].message.content.strip()
    return data

def analyze_text_a(text):
    prompt = (
        f"Estrai i seguenti dati testo in un formato tale che sia facile creare un csv con le colonne corrispondenti ai dati che hai estratto ignorando il prezzo totale, siamo interessati ai singoli prodotti. "
        f"Se non dovessi trovare qualche dato, lascia semplicemente il campo o i campi vuoti senza scusarti o dirmi che non sei riuscita trovare determinati dati."
        f"L'output non deve mai contenere spiegazioni o testo aggiuntivo, solo i dati estratti nel formato che ti verrà chiesto."
        f"Se ci sono più prodotti, separa ogni prodotto con un punto e virgola (;)."
        f"Queste sono le colonne che ci interessano:"
        f"[Prodotto, Quantità, Marca, Prezzo_Articolo, Data, Supermercato, Metodo_Pagamento]."
        f"Seguono delle spiegazioni per ogni colonna per aiutarti a capire cosa cerco:"
        f"Prodotto: Il nome del prodotto acquistato tutto minuscolo, esempio: acqua."
        f"Quantità: Il numero di unità del prodotto acquistate, esempio: 2."
        f"Marca: Il nome del prodotto tutto minuscolo, esempio: san benedetto."
        f"Prezzo_Articolo: Il prezzo di un singolo articolo utilizzando il punto come sepratore decimale, esempio: 1.50."
        f"Data: La data dello scontrino nel formato gg/mm/aa, esempio: 21/05/24."
        f"Supermercato: Il nome del supermercato tutto minuscolo, esempio: carrefour."
        f"Metodo_Pagamento: Il metodo di pagamento utilizzato tutto minuscolo, esempio: carta di credito ***1234."
        f"Fai attenzione a non modificare il nome delle colonne per non causare problemi con il concatenamento di nuove righe nel dataframe."
        f"L'output deve essere simile a questo per permettermi di salvare i vari prodotti in righe differenti del mio csv usando il punto e virgola per separare i prodotti:"
        f"Prodotto: biscotti, Quantità: 1, Marca: misura, Prezzo_Articolo: 2.50, Data: 21/05/24, Supermercato: carrefour, Metodo_Pagamento: carta di credito ***1234; "
        f"Prodotto: pollo, Quantità: 1, Marca:, Prezzo_Articolo: 2.50, Data: 21/05/24, Supermercato: carrefour, Metodo_Pagamento: carta di credito ***1234."
        f"\n\nTesto:\n{text}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a grocery store cashier. A customer hands you a receipt. You need to extract the following data from the receipt: Prodotto, Quantità, Marca, Prezzo_Articolo, Data, Supermercato, Metodo_Pagamento."},
            {"role": "user", "content": prompt}
        ]
    )
    data = response.choices[0].message.content.strip()
    return data

def parse_analyzed_data(analyzed_data):
    """Parse the extracted data into a structured format."""
    data_list = []
    product_entries = analyzed_data.split(';')
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
st.title("Analizzatore di Scontrini e Audio")

# Image and PDF Upload
uploaded_file = st.file_uploader("Carica un'immagine o un file PDF dello scontrino", type=["jpg", "jpeg", "png", "pdf"])

# Load existing data
existing_data = load_existing_data(NEW_DATA_FILE)

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_image(uploaded_file)
    st.text_area("Testo Estratto", text, height=200)
    if st.button("Analizza Testo"):
        analyzed_data = analyze_text(text)
        st.text_area("Dati Estratti", analyzed_data, height=200)
        new_data = parse_analyzed_data(analyzed_data)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        save_data_to_file(updated_data, NEW_DATA_FILE)
        csv = updated_data.to_csv(index=False)
        st.download_button(label="Scarica dati come CSV", data=csv, file_name='new_scontrino_dati.csv', mime='text/csv')

# Time series plot of expenses
st.title("Grafico delle Spese nel Tempo")
data_for_plotting = load_existing_data(NEW_DATA_FILE)
if not data_for_plotting.empty:
    data_for_plotting['Data'] = pd.to_datetime(data_for_plotting['Data'], dayfirst=True, errors='coerce')
    data_for_plotting['Prezzo_Articolo'] = data_for_plotting['Prezzo_Articolo'].astype(str).str.replace(',', '.').astype(float)
    st.subheader("Visualizzazione delle spese nel tempo")
    fig = px.line(data_for_plotting, x='Data', y="Prezzo_Articolo", title="Spese nel Tempo", labels={"Prezzo_Articolo": "Prezzo_Articolo", "Data": "Data"})
    st.plotly_chart(fig)
    st.subheader("Filtra per Data")
    start_date = st.date_input("Data di Inizio", value=data_for_plotting['Data'].min())
    end_date = st.date_input("Data di Fine", value=data_for_plotting['Data'].max())
    filtered_data = data_for_plotting[(data_for_plotting['Data'] >= pd.to_datetime(start_date)) & (data_for_plotting['Data'] <= pd.to_datetime(end_date))]
    st.write("Dati Filtrati:")
    st.dataframe(filtered_data)
    filtered_csv = filtered_data.to_csv(index=False)
    st.download_button(label="Scarica dati filtrati come CSV", data=filtered_csv, file_name='dati_filtrati.csv', mime='text/csv')

# Audio upload and transcription
st.title("Trascrizione Audio")
uploaded_audio = st.file_uploader("Carica un file audio", type=["wav", "m4a"])

if uploaded_audio is not None:
    st.write("File audio caricato. Estrazione trascrizione in corso...")
    if uploaded_audio.type == "audio/x-m4a":
        audio_file = convert_m4a_to_wav(uploaded_audio)
    else:
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_audio.getbuffer())
        audio_file = "uploaded_audio.wav"
    
    transcript = transcribe_audio(audio_file)
    st.text_area("Trascrizione Audio", transcript, height=200)
    if st.button("Analizza Trascrizione"):
        analyzed_data = analyze_text_a(transcript)
        st.text_area("Dati Estratti dalla Trascrizione", analyzed_data, height=200)
        new_data = parse_analyzed_data(analyzed_data)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        save_data_to_file(updated_data, NEW_DATA_FILE)
        csv = updated_data.to_csv(index=False)
        st.download_button(label="Scarica dati come CSV", data=csv, file_name='new_scontrino_dati.csv', mime='text/csv')
