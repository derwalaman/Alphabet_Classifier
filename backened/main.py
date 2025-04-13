from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import model
from weights import w1, w2, saved_accuracy  # Import pre-trained weights

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BinaryInput(BaseModel):
    data: list

@app.post("/predict")
def predict_letter(input: BinaryInput):
    if len(input.data) != 30:
        raise HTTPException(status_code=400, detail="Input must be 30 binary values")
    
    x_input = np.array(input.data).reshape(1, 30)
    output = model.f_forward(x_input, np.array(w1), np.array(w2))

    predicted_index = int(np.argmax(output))
    predicted_char = chr(predicted_index + 65)

    return {
        "prediction": predicted_char,
        "probabilities": output.tolist()[0],
        "accuracy": saved_accuracy,
    }
