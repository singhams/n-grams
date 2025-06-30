import streamlit as st
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.util import ngrams

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Function to process text and get common bigrams, trigrams, and quadgrams
def process_text_for_ngrams(text, remove_stopwords=True, top_n=20):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = word_tokenize(text)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    
    if len(words) < 2:
        raise ValueError("Not enough words for bigrams.")
    if len(words) < 3:
        raise ValueError("Not enough words for trigrams.")
    if len(words) < 4:
        raise ValueError("Not enough words for quadgrams.")
    
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, top_n)
    
    trigram_finder = TrigramCollocationFinder.from_words(words)
    trigrams = trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio, top_n)
    
    quadgrams = list(ngrams(words, 4))
    quadgram_freq = nltk.FreqDist(quadgrams)
    common_quadgrams = quadgram_freq.most_common(top_n)
    
    return bigrams, trigrams, common_quadgrams

# Function to process all data together for n-grams
def process_data_together_for_ngrams(data, remove_stopwords=True, top_n=20):
    data = [str(item) for item in data]
    combined_text = ' '.join(data)
    combined_text = combined_text.lower()
    combined_text = re.sub(r'[^a-zA-Z0-9\s]', '', combined_text)
    words = word_tokenize(combined_text)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigram_freq = bigram_finder.ngram_fd.items()
    
    trigram_finder = TrigramCollocationFinder.from_words(words)
    trigram_freq = trigram_finder.ngram_fd.items()
    
    quadgrams = list(ngrams(words, 4))
    quadgram_freq = nltk.FreqDist(quadgrams).items()
    
    return {
        'bigrams': sorted(bigram_freq, key=lambda x: x[1], reverse=True)[:top_n],
        'trigrams': sorted(trigram_freq, key=lambda x: x[1], reverse=True)[:top_n],
        'quadgrams': sorted(quadgram_freq, key=lambda x: x[1], reverse=True)[:top_n]
    }

# Streamlit app
st.title("N-Gram Frequency Analyzer")

# File uploader for Excel or text input
uploaded_file = st.file_uploader("Choose an Excel or text file", type=["xlsx", "txt"])

# Initialize data
data = []
file_type = None

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        file_type = "excel"
        df = pd.read_excel(uploaded_file, header=None)
        if not df.empty:
            data = df.iloc[:, 0].dropna().astype(str).tolist()
            st.success(f"Uploaded Excel file with {len(data)} rows.")
        else:
            st.error("The uploaded Excel file is empty.")
    elif uploaded_file.name.endswith('.txt'):
        file_type = "text"
        content = uploaded_file.read().decode('utf-8').strip()
        if content:
            data = content
            st.success(f"Uploaded text file with {len(content)} characters.")
        else:
            st.error("The uploaded text file is empty.")

# Workflow for Excel files
if file_type == "excel" and data:
    top_n = st.number_input("Enter the number of top n-grams to display", min_value=1, value=20)

    if st.button("Start Analysis"):
        combined_result_ngrams = process_data_together_for_ngrams(data, top_n=top_n)
        combined_results_df = pd.DataFrame({
            'bigrams': [' '.join(map(str, bigram)) for bigram, _ in combined_result_ngrams['bigrams']],
            'bigram_freq': [freq for _, freq in combined_result_ngrams['bigrams']],
            'trigrams': [' '.join(map(str, trigram)) for trigram, _ in combined_result_ngrams['trigrams']],
            'trigram_freq': [freq for _, freq in combined_result_ngrams['trigrams']],
            'quadgrams': [' '.join(map(str, quadgram)) for quadgram, _ in combined_result_ngrams['quadgrams']],
            'quadgram_freq': [freq for _, freq in combined_result_ngrams['quadgrams']]
        })
        st.write("Combined N-Gram Results", combined_results_df)
        combined_results_df.to_excel("combined_ngrams_results.xlsx", index=False)
        st.download_button(label="Download Combined Results as Excel", data=open("combined_ngrams_results.xlsx", 'rb'), file_name="combined_ngrams_results.xlsx")

# Workflow for text files
elif file_type == "text" and data:
    top_n = st.number_input("Enter the number of top n-grams to display", min_value=1, value=20)

    if st.button("Start Analysis"):
        combined_result_ngrams = process_data_together_for_ngrams([data], top_n=top_n)
        combined_results_df = pd.DataFrame({
            'bigrams': [' '.join(map(str, bigram)) for bigram, _ in combined_result_ngrams['bigrams']],
            'bigram_freq': [freq for _, freq in combined_result_ngrams['bigrams']],
            'trigrams': [' '.join(map(str, trigram)) for trigram, _ in combined_result_ngrams['trigrams']],
            'trigram_freq': [freq for _, freq in combined_result_ngrams['trigrams']],
            'quadgrams': [' '.join(map(str, quadgram)) for quadgram, _ in combined_result_ngrams['quadgrams']],
            'quadgram_freq': [freq for _, freq in combined_result_ngrams['quadgrams']]
        })
        st.write("N-Gram Results", combined_results_df)
        combined_results_df.to_excel("text_ngrams_results.xlsx", index=False)
        st.download_button(label="Download Results as Excel", data=open("text_ngrams_results.xlsx", 'rb'), file_name="text_ngrams_results.xlsx")
