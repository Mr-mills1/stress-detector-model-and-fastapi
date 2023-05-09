# load the libraries
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# load the pickled model
cv = pickle.load(open("Xcv.pkl", "rb"))
model = pickle.load(open('stress_model.pkl','rb'))
# create the app
app = FastAPI()
# get app route or endpoint
@app.post("/predict")
async def predict(text: str):
    #transform the inptut to Tfidf vectors 
    
    text_cv = cv.transform([text]).toarray()
    
    #predict the class of the input text
    prediction = model.predict(text_cv)
    
    #map the predicted class to a string
    class_name = "Unstress" if prediction == 1 else "Stress"
    
    #Return the prediction in a JSON response
    return {
        "text":text,
        "class":class_name
    }
