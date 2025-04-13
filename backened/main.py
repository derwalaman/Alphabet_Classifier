from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
from model import f_forward, generate_wt, sigmoid, back_propagation, train, loss, one_hot_labels, letters_binary

app = FastAPI()

# Allow all origins (CORS setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prepare training data
x = [np.array(binary).reshape(1, 30) for binary in letters_binary.values()]
y = np.array([one_hot_labels[chr(i + 65)] for i in range(26)])

# Load or train model weights
weights_file = "weights.pkl"
if os.path.exists(weights_file):
    try:
        data = joblib.load(weights_file)
        if len(data) == 2:
            w1, w2 = data
            saved_accuracy = 0.0
        else:
            w1, w2, saved_accuracy = data
    except Exception as e:
        print(f"Failed to load weights: {e}")
        w1 = generate_wt(30, 5)
        w2 = generate_wt(5, 26)
        w1, w2, accuracy, _ = train(x, y, w1, w2, learning_rate=0.01, epochs=100000)
        saved_accuracy = accuracy
        joblib.dump((w1, w2, saved_accuracy), weights_file)
else:
    w1 = generate_wt(30, 5)
    w2 = generate_wt(5, 26)
    w1, w2, accuracy, _ = train(x, y, w1, w2, learning_rate=0.01, epochs=100000)
    saved_accuracy = accuracy
    joblib.dump((w1, w2, saved_accuracy), weights_file)

# Request model
class BinaryInput(BaseModel):
    data: list  # Should contain 30 binary values

# Prediction endpoint
@app.post("/predict")
def predict_letter(input: BinaryInput):
    if len(input.data) != 30:
        raise HTTPException(status_code=400, detail="Input must be 30 binary values")

    x_input = np.array(input.data).reshape(1, 30)
    output = f_forward(x_input, w1, w2)

    predicted_index = int(np.argmax(output))
    predicted_char = chr(predicted_index + 65)

    return {
        "prediction": predicted_char,
        "probabilities": output.tolist()[0],
        "accuracy": saved_accuracy,
    }
