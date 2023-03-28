# Boolean-Question-Answering #

This is a boolean question answering project that uses an encoder and decoder architecture with LSTM layers inside of them to answer yes/no questions. It is built using Python and requires the following modules to be installed:


 * `numpy`
 * `pickle`
 * `tensorflow`
 * `streamlit `
 * `keras`
 
 
 ## Installation ##
 
 
 To install the necessary modules, run the following command:


* `pip install numpy`
* `pip install pickle`
* `pip install tensorflow`
* `pip install streamlit`
* `pip install keras`

## Usage ##

`streamlit run app.py`


This will launch a web application where you can enter a yes/no question and get the predicted answer. The web application is built using the Streamlit library.


## Training ##


The boolean question answering model is trained using an encoder and decoder architecture with LSTM layers inside of them. The training data is a corpus of questions and their corresponding answers labeled as either "yes" or "no". The text is preprocessed and then converted into numerical features using the Bag of Words (BoW) method. The resulting feature vectors are used to train the LSTM-based encoder-decoder model to predict the answer to yes/no questions.
