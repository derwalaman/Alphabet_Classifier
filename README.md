# Alphabet Recognition with Binary Encoding (A-Z)

This project demonstrates a simple neural network or ML model that recognizes alphabets (A–Z) using their binary visual representations. Each alphabet is encoded as a 5x6 binary matrix (flattened into 30 features), and mapped to a one-hot encoded label.

## 🌐 **Live Demo**
[https://alphabet-classifier.vercel.app](https://alphabet-classifier.vercel.app)

## 🧠 Project Structure

- **letters_binary**: A dictionary containing each letter (A–Z) as a 5x6 binary image.
- **x**: Input feature list where each element is a `(1, 30)` NumPy array representing a letter.
- **y**: Output labels using one-hot encoding, e.g., `A = [1, 0, 0, ..., 0]`, `B = [0, 1, 0, ..., 0]`, etc.

## 🧠 Backend Overview (FastAPI + ML Model)

- Built using **FastAPI**
- Hosted on [Railway](https://railway.app)
- Accepts a 5×6 binary grid (30 values) via POST request to the `/predict` endpoint
- Returns the predicted letter from A to Z

## 🗂 Data Format
Each letter is represented by a binary pattern (5 rows × 6 columns = 30 bits):

```python
# Example:
letters_binary["A"] = [
    0,1,1,1,1,0,
    1,0,0,0,0,1,
    1,1,1,1,1,1,
    1,0,0,0,0,1,
    1,0,0,0,0,1
]
```

## 🎯 Labels (One-Hot Encoding)
Each label is represented as a one-hot encoded vector:
```python
# A to Z:
y = [
    [1, 0, 0, ..., 0],  # A
    [0, 1, 0, ..., 0],  # B
    ...
    [0, 0, 0, ..., 1]   # Z
]
```

## 📦 Backend Requirements
- Install the required Python packages:
```bash
pip install fastapi uvicorn scikit-learn numpy
```

- ▶️ Running Backend Locally
```bash
uvicorn app:app --reload
```

## 💻 Frontend Overview (Next.js + Tailwind CSS)
- Built using Next.js App Router
- Styled with Tailwind CSS
- Hosted on Vercel
- Lets users interact with a 5×6 binary grid to draw a letter
- Sends the grid data to the backend and displays the predicted alphabet

- 🧪 Features
- - ✅ Interactive 5×6 drawing grid
- - ✅ Predict button to send input to ML model
- - ✅ Clear/reset grid option
- - ✅ Displays prediction instantly
- - ✅ Responsive layout (Mobile Friendly)

- 🛠 Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🚀 Usage
- Prepare the letters_binary dictionary for A to Z.

- Create the input and output datasets:

```python
x = [
    np.array(letters_binary["A"]).reshape(1, 30),
    np.array(letters_binary["B"]).reshape(1, 30),
    ...
    np.array(letters_binary["Z"]).reshape(1, 30),
]

y = [
    [1, 0, 0, ..., 0],  # A
    [0, 1, 0, ..., 0],  # B
    ...
    [0, 0, 0, ..., 1]   # Z
]
```
- Feed x and y into your model (MLP, logistic regression, etc.)

## 🧪 Example Applications
- Simple handwritten character recognition.

- Basic neural network classifier testing.

- Educational use to understand image-to-vector conversion and one-hot labels
