import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re

# ✅ Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ✅ Load the trained model
model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# ✅ Text Preprocessing Function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize words
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(filtered_tokens)

# ✅ Streamlit UI
st.title("Email Spam Detection App 🚀")

sms = st.text_area("Enter the message:", "")

if st.button("Check Spam"):
    if sms:
        transformed_sms = transform_text(sms)
        vectorized_input = vectorizer.transform([transformed_sms])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.error("🚨 Spam Message Detected!")
        else:
            st.success("✅ This message is NOT spam.")
    else:
        st.warning("⚠️ Please enter a message to check.")

