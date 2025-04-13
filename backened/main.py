from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
from model import f_forward, generate_wt, sigmoid, back_propagation, train, loss, one_hot_labels, letters_binary

app = FastAPI()

# Allow all origins (use with caution in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods, including OPTIONS
    allow_headers=["*"],  # Allows all headers
)

# One-time conversion of input characters to data and labels
x = [np.array(binary).reshape(1, 30) for binary in letters_binary.values()]
y = np.array([one_hot_labels[chr(i + 65)] for i in range(26)])

# Load weights and accuracy using joblib, or train and save
if os.path.exists("weights.pkl"):
    data = joblib.load("weights.pkl")
    if len(data) == 2:
        w1, w2 = data
        saved_accuracy = 0.0
    else:
        w1, w2, saved_accuracy = data
else:
    w1 = generate_wt(30, 5)
    w2 = generate_wt(5, 26)
    w1, w2, accuracy, losses = train(x, y, w1, w2, learning_rate=0.01, epochs=100000)
    saved_accuracy = accuracy
    joblib.dump((w1, w2, saved_accuracy), "weights.pkl")

# API input model
class BinaryInput(BaseModel):
    data: list  # list of 30 binary integers

@app.post("/predict")
def predict_letter(input: BinaryInput):
    if len(input.data) != 30:
        raise HTTPException(status_code=400, detail="Input must be 30 binary values")

    x_input = np.array(input.data).reshape(1, 30)

    # Feedforward prediction
    output = f_forward(x_input, w1, w2)
    predicted_index = int(np.argmax(output))
    predicted_char = chr(predicted_index + 65)

    return {
        "prediction": predicted_char,
        "probabilities": output.tolist()[0],
        "accuracy": saved_accuracy,
    }
