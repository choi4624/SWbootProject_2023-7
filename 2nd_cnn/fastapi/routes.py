from fastapi import FastAPI
from model import predict
import PIL.Image

app = FastAPI()

@app.post("/predict")
def predict_fault(image: bytes):
    image = PIL.Image.open(io.BytesIO(image))
    prediction = predict(image)
    return prediction
