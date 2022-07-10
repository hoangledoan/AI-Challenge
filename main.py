import json
import pickle

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open('ai-challenge.pkl', 'rb'))
ohe = pickle.load(open('ohe.pkl', 'rb'))
scaler = pickle.load(open('scl.pkl', 'rb'))


class Model_input(BaseModel):
    year: int
    month: str


def transform(cat1, cat2, num, cat3):
    category_input = [cat1, cat2, cat3]
    x = ohe.transform([category_input])
    y = scaler.transform([[num]])
    z = model.predict(np.concatenate((y, x), axis=1))
    return z[0]


@app.post('/predict')
def prediction(data: Model_input):
    input_data = data.json()
    input_dictionary = json.loads(input_data)
    category = 'Alkoholunfälle'
    type = 'insgesamt'
    year = input_dictionary['year']
    month = input_dictionary['month']
    prediction = transform(category, type, year, month)
    return {'prediction': prediction.tolist()}


# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
