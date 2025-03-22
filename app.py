import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

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

if st.button('Predict'):  # Corrected st.Buttom to st.button
    transformed_sms = transform_text(sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        st.header("🚨 SPAM 🚨")
    else:
        st.header("✅ NOT SPAM ✅")
