from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI()

import pickle
import json
# class Item(BaseModel):
#     name: str
#     price: float
#     is_offer: Union[bool, None] = None

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}

model = pickle.load(open('ai-challenge.pkl', 'rb'))
ohe = pickle.load(open('ohe.pkl', 'rb'))
scaler = pickle.load(open('scl.pkl', 'rb'))


class Model_input(BaseModel):
    category: str
    type: str
    year: int
    month: str


data = Model_input(category="Alkoholunfälle",
                   type="insgesamt",
                   year=2021,
                   month="01")

# data = data.dict()
# categorical_columns = ['category', 'type', 'month']
# numerical_column = ['year']

# category_input = [category, type, month]
# x = ohe.transform([list_input])
# y = scaler.transform([[year]])
# z = np.concatenate((y, x), axis=1)

# print(model.predict(z))


@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.post('/predict')
def prediction(data: Model_input):
    input_data = data.json()
    input_dictionary = json.loads(input_data)
    category = input_dictionary['category']
    type = input_dictionary['type']
    year = input_dictionary['year']
    month = input_dictionary['month']
    category_input = [category, type, month]
    x = ohe.transform([category_input])
    y = scaler.transform([[year]])
    z = model.predict(np.concatenate((y, x), axis=1))
    prediction = z[0]
    return {'prediction': prediction}


print(prediction(data))
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
