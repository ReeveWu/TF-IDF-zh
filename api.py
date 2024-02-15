from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from utilities.TfidfVectorizer import TfidfVectorizer

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = TfidfVectorizer()

class Data(BaseModel):
    data: List[List[str]]

@app.post('/fit_transform')
async def fit_transform(payload: Data):
    model.reset()
    X = model.fit_transform(payload.data)
    message = {
        "idf": model.idf_,
        "vocabulary": model.vocabulary_,
        "value": X
    }

    return message