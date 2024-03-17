import streamlit as st
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz
import numpy as np

# load the model
model = pickle.load(open('debat_sentiment.pkl', 'rb'))

# load data
data_path = "sentiment_debat.csv"
data = pd.read_csv(data_path)

# Sidebar
st.sidebar.title("Debat Sentiment Analysis")
nav = st.sidebar.radio("Navigation", ["Prediksi", "Visualisasi Sentimen"])

if nav == "Prediksi":
    st.title('Debat Sentiment Analysis')
    tweet = st.text_input('Masukkan Kata')
    submit = st.button('Prediksi')
    if submit:
        start = time.time()
        prediction = model.predict([tweet])
        end = time.time()
        st.write('Waktu prediksi yang dibutuhkan: ', round(end-start, 2), 'detik')
        
        # Menampilkan peringatan
        st.warning('**Catatan:** Harap dicatat bahwa akurasi prediksi model ini hanya sekitar 67%, Oleh karena itu, prediksi yang dihasilkan mungkin tidak selalu akurat.')
        
        print(prediction[0])
        st.write(prediction[0])

elif nav == "Visualisasi Sentimen":
    st.title("Visualisasi Data")

    # Menampilkan distribusi sentimen
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    data['sentiment'].value_counts().plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Menampilkan visualisasi distribusi sentimen untuk kata kunci tertentu
    st.subheader("Sentiment Distribution For Capres")

    # Daftar kata kunci
    keywords = ["ganjar", "anies", "prabowo"]

    # Selectbox untuk memilih kata kunci
    selected_keyword = st.selectbox("Select a keyword", keywords)

    # Fungsi untuk mencari typo
    def find_typo(data, column, word_to_match, threshold=80):
        typo_list = []
        for index, row in data.iterrows():
            ratio = fuzz.ratio(row[column].lower(), word_to_match)
            if ratio >= threshold:
                typo_list.append(row[column])
        return typo_list

    # Mencari typo dalam kolom 'text'
    typo_list = find_typo(data, 'text', selected_keyword)

    # Menghitung distribusi sentimen untuk kata kunci dan kata-kata yang mirip
    sentiment_counts = data[data['text'].str.contains('|'.join([selected_keyword] + typo_list), case=False)].groupby('sentiment').size()

    # Memvisualisasikan distribusi sentimen untuk kata kunci menggunakan pie chart
    fig, ax = plt.subplots()
    sentiment_counts = sentiment_counts[sentiment_counts.index.isin(['positive', 'negative'])]
    ax.pie(sentiment_counts, labels=[f'{label} ({count})' for label, count in zip(sentiment_counts.index, sentiment_counts)], autopct='%1.1f%%')
    ax.set_title(f'Sentiment Distribution for "{selected_keyword}"')
    st.pyplot(fig)