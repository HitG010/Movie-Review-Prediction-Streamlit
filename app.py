import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the model
model = load_model('rnn_model.h5')
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
maxlen = 500

# helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review

## Prediction function
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    return sentiment, prediction[0][0]

import streamlit as st

st.title('Sentiment Analysis with RNN')
st.write('This is a simple web app to predict sentiment using a trained RNN model.')
st.write('Enter a review and press the button to predict the sentiment.')

review = st.text_area('Enter a review:')

if st.button('Predict Sentiment'):
    sentiment, prediction = predict_sentiment(review)
    
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Probability: {prediction:.4f}')
else:
    st.write('Enter a review and press the button to predict the sentiment.')
