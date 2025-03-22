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
        st.markdown(
            "<h1 style='text-align: center; color: red;'>🚨 SPAM ALERT! 🚨</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='text-align: center; background-color: #ffcccc; padding: 10px; border-radius: 10px;'>"
            "<h3 style='color: red;'>Be Careful! This message looks like SPAM.</h3>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h1 style='text-align: center; color: green;'>✅ NOT SPAM ✅</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='text-align: center; background-color: #ccffcc; padding: 10px; border-radius: 10px;'>"
            "<h3 style='color: green;'>You're Safe! This message seems genuine. 👍</h3>"
            "</div>",
            unsafe_allow_html=True
        )

