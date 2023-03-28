import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import keras
import pickle

tokenizer = Tokenizer(filters = [])
# Load pickled vocabulary
with open("D:\question answering\word_dict.pkl", "rb") as f:
    word_index = pickle.load(f)

# Load saved model
model = keras.models.load_model("D:\question answering\QandA.h5")
vocab_size = 38
def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=156 ,max_question_len=6):
    X = []
    Xq = []
    for story, query in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]
        X.append(x)
        Xq.append(xq)
    return (pad_sequences(X, maxlen=max_story_len),pad_sequences(Xq, maxlen=max_question_len))

# Define function to predict answer
def predict_answer(story, question):
    story = story.split()
    question = question.split()
    my_data = [(story, question)]
    my_story, my_ques= vectorize_stories(my_data, word_index)
    pred_results = model.predict(([my_story, my_ques]))
    val_max = np.argmax(pred_results[0])
    for key, val in word_index.items():
        if val == val_max:
            k = key
    return k, pred_results[0][val_max]

# Define Streamlit app
def app():
    st.title("Story QA")
    # Get inputs
    story = st.text_area("Enter the story:" )
    question = st.text_input("Enter the question:")
    # Predict answer and show result
    if st.button("Submit"):
        pred_ans = predict_answer(story, question)
        result = f"Predicted answer: {pred_ans}"
        st.write(result)

# Run Streamlit app
if __name__ == "__main__":
    app()