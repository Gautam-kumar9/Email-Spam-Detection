import streamlit as st
import nltk
import string
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 🛠 Fix: Ensure 'punkt' is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load trained model & vectorizer (Ensure these files exist in your project)
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Text preprocessing function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuations
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(tokens)

# Streamlit UI
st.title("📩 Email Spam Detection App")

# User input
sms = st.text_area("Enter the message/email:")

if st.button("Predict"):
    if sms:
        transformed_sms = transform_text(sms)
        vector_input = vectorizer.transform([transformed_sms])  # Convert text to vector
        result = model.predict(vector_input)[0]  # Predict spam or not

        # Display result
        if result == 1:
            st.error("⚠️ This message is **Spam**!")
        else:
            st.success("✅ This message is **Not Spam**.")
    else:
        st.warning("Please enter a message for prediction.")
