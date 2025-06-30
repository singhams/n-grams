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
    
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, top_n)
    
    trigram_finder = TrigramCollocationFinder.from_words(words)
    trigrams = trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio, top_n)
    
    quadgrams = list(ngrams(words, 4))
    quadgram_freq = nltk.FreqDist(quadgrams)
    common_quadgrams = quadgram_freq.most_common(top_n)
    
    return bigrams, trigrams, common_quadgrams

# Function to process each row separately for n-grams
def process_data_separately_for_ngrams(data, remove_stopwords=True, top_n=20):
    results = []
    for row in data:
        bigrams, trigrams, quadgrams = process_text_for_ngrams(row, remove_stopwords, top_n)
        results.append({
            'text': row,
            'bigrams': ', '.join([' '.join(bigram) for bigram in bigrams]),
            'trigrams': ', '.join([' '.join(trigram) for trigram in trigrams]),
            'quadgrams': ', '.join([' '.join(quadgram) for quadgram, _ in quadgrams])
        })
    return results

# Function to process all data together for n-grams
def process_data_together_for_ngrams(data, remove_stopwords=True, top_n=20):
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

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, header=None)  # Read without assuming a header
        if not df.empty:
            data = df.iloc[:, 0].dropna().tolist()  # Select the first column and drop NaN values
        else:
            st.error("The uploaded Excel file is empty.")
    elif uploaded_file.name.endswith('.txt'):
        content = uploaded_file.read().decode('utf-8').strip()
        if content:
            data = content.splitlines()
        else:
            st.error("The uploaded text file is empty.")

# N-Gram selection
ngram_type = st.radio("Select N-Gram Type", ("Bigrams", "Trigrams", "Quadgrams", "All Three"))

# Processing mode selection
processing_mode = st.radio("Select Processing Mode", ("Process all text together", "Process each line/row separately"))

# Top-N selection (conditionally displayed)
top_n = st.number_input("Enter the number of top n-grams to display", min_value=1, value=20)

# Process button
if st.button("Start Analysis"):
    if data:
        if processing_mode == "Process all text together":
            combined_result_ngrams = process_data_together_for_ngrams(data, top_n=top_n)
            combined_results_df = pd.DataFrame({
                'bigrams': [' '.join(bigram) for bigram, _ in combined_result_ngrams['bigrams']],
                'bigram_freq': [freq for _, freq in combined_result_ngrams['bigrams']],
                'trigrams': [' '.join(trigram) for trigram, _ in combined_result_ngrams['trigrams']],
                'trigram_freq': [freq for _, freq in combined_result_ngrams['trigrams']],
                'quadgrams': [' '.join(quadgram) for quadgram, _ in combined_result_ngrams['quadgrams']],
                'quadgram_freq': [freq for _, freq in combined_result_ngrams['quadgrams']]
            })
            st.write("Combined N-Gram Results", combined_results_df)
            combined_output_file = 'combined_ngrams_results.xlsx'
            combined_results_df.to_excel(combined_output_file, index=False)
            st.download_button(label="Download Combined Results as Excel", data=open(combined_output_file, 'rb'), file_name=combined_output_file)
        else:
            separate_results_ngrams = process_data_separately_for_ngrams(data, top_n=top_n)
            separate_results_df = pd.DataFrame(separate_results_ngrams)
            st.write("Separate N-Gram Results", separate_results_df)
            separate_output_file = 'separate_ngrams_results.xlsx'
            separate_results_df.to_excel(separate_output_file, index=False)
            st.download_button(label="Download Separate Results as Excel", data=open(separate_output_file, 'rb'), file_name=separate_output_file)
    else:
        st.error("No valid data found in the uploaded file.")
