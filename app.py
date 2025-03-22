import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Ensure necessary NLTK datasets are downloaded
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")

# Append to nltk data path
nltk.data.path.append(nltk_data_path)

# Check if 'punkt' is available; if not, download it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

# Check if 'stopwords' is available; if not, download it
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

ps = PorterStemmer()

# Load pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("📩 Email/SMS Spam Classifier 🚀")

# User input
sms = st.text_area("Enter the message")

def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize text

    # Remove non-alphanumeric characters
    words = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    words = [word for word in words if word not in stopwords.words('english') and word not in string.punctuation]

    # Apply stemming
    words = [ps.stem(word) for word in words]

    return " ".join(words)

if st.button('Predict'):  
    transformed_sms = transform_text(sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        st.header("🚨 SPAM 🚨")
    else:
        st.header("✅ NOT SPAM ✅")
