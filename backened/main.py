from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from model import f_forward, generate_wt, sigmoid, back_propagation, train, loss, one_hot_labels, letters_binary, predict as user_predict
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware

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

# Load weights and accuracy if they exist, else train, save them
if os.path.exists("weights.pkl"):
    with open("weights.pkl", "rb") as f:
        data = pickle.load(f)
        if len(data) == 2:
            # If only weights are loaded (older pickle file), assume no accuracy and set a default
            w1, w2 = data
            saved_accuracy = 0.0  # Default accuracy if it's not present in the pickle file
        elif len(data) == 3:
            # If weights and accuracy are loaded
            w1, w2, saved_accuracy = data
else:
    w1 = generate_wt(30, 5)
    w2 = generate_wt(5, 26)
    w1, w2, accuracy, losses = train(x, y, w1, w2, learning_rate=0.01, epochs=100000)
    saved_accuracy = accuracy
    with open("weights.pkl", "wb") as f:
        pickle.dump((w1, w2, saved_accuracy), f)

# API input model
class BinaryInput(BaseModel):
    data: list  # list of 30 binary integers

@app.post("/predict")
def predict_letter(input: BinaryInput):
    if len(input.data) != 30:
        raise HTTPException(status_code=400, detail="Input must be 30 binary values")

    x_input = np.array(input.data).reshape(1, 30)

    # Use user's custom predict function (modification to return result instead of print/show)
    output = f_forward(x_input, w1, w2)
    maxm = 0
    k = 0

    for i in range(len(output[0])):
        if output[0][i] > maxm:
            maxm = output[0][i]
            k = i

    predicted_char = chr(k + 65)

    return {
        "prediction": predicted_char,
        "probabilities": output.tolist()[0],
        "accuracy": saved_accuracy,  # Include the accuracy in the response
    }
