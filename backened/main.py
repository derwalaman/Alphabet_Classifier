from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
from model import f_forward, generate_wt, sigmoid, back_propagation, train, loss, one_hot_labels, letters_binary

app = FastAPI()

# CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/")
def root():
    return {"status": "Backend is up and running 🚀"}

# Prepare training data
x = [np.array(binary).reshape(1, 30) for binary in letters_binary.values()]
y = np.array([one_hot_labels[chr(i + 65)] for i in range(26)])

# Load or train weights
weights_file = "weights.pkl"
saved_accuracy = 0.0

try:
    if os.path.exists(weights_file):
        data = joblib.load(weights_file)
        if len(data) == 2:
            w1, w2 = data
        else:
            w1, w2, saved_accuracy = data
    else:
        raise FileNotFoundError("Weights file not found. Training new model.")
except Exception as e:
    print(f"Loading failed: {e} — training model...")
    w1 = generate_wt(30, 5)
    w2 = generate_wt(5, 26)
    w1, w2, saved_accuracy, _ = train(x, y, w1, w2, learning_rate=0.01, epochs=100000)
    joblib.dump((w1, w2, saved_accuracy), weights_file)

# Input schema
class BinaryInput(BaseModel):
    data: list  # List of 30 binary values

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

# For Railway to run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
