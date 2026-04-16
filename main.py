# importing module

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb

## Loading imdb datasets

word_index= imdb.get_word_index()
reverse_word_index= {value: key for key, value in word_index.items()}

##loading pre-trained models
model= load_model('simple_rnn.keras')

#decoding reviews
def decoding(text):
    return ' '.join(reverse_word_index.get(i-3, '?') for i in text)

#preprocess user input

def preprocessing(text):
    words= text.lower().split()
    encoded_review= [word_index.get(word,2)+3 for word in words]
    padded_review= sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

#streamlit setup
import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# ---------------- DEMO EXAMPLES ----------------
st.subheader("Try Demo Examples 👇")

example1 = "The movie was amazing and I loved every part of it"
example2 = "Worst movie ever, totally waste of time"
example3 = "The movie was okay, not great but not bad either"

col1, col2, col3 = st.columns(3)

if col1.button("Positive Example"):
    st.session_state.user_input = example1

if col2.button("Negative Example"):
    st.session_state.user_input = example2

if col3.button("Neutral Example"):
    st.session_state.user_input = example3

# ---------------- INPUT BOX ----------------
user_input = st.text_area(
    'Movie Review',
    value=st.session_state.get("user_input", "")
)

# ---------------- PREDICTION ----------------
if st.button('Classify'):
    if user_input.strip() != "":
        processed_input = preprocessing(user_input)
        prediction = model.predict(processed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]}')
    else:
        st.write('Please enter a movie review.')

