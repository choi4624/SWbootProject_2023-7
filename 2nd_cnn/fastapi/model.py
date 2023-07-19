import torch

def load_model():
    model = torch.load("model.ckpt")
    return model

def predict(image):
    model = load_model()
    prediction = model(image)
    return prediction
