from model import train, generate_wt, one_hot_labels, letters_binary, f_forward, sigmoid, back_propagation, loss
import numpy as np

x = [np.array(binary).reshape(1, 30) for binary in letters_binary.values()]
y = np.array([one_hot_labels[chr(i + 65)] for i in range(26)])

w1 = generate_wt(30, 5)
w2 = generate_wt(5, 26)

print("Training the model...")
w1, w2, accuracy, _ = train(x, y, w1, w2, learning_rate=0.01, epochs=100000)

print("Model trained successfully.")
print(f"Final accuracy: {accuracy}")
print("Saving weights to file...")
print(w1)
print(w2)

# Write weights to weights.py file
try:
    with open("weights.py", "w") as f:
        f.write(f"w1 = {w1.tolist()}\n")  # Convert numpy arrays to lists
        f.write(f"w2 = {w2.tolist()}\n")
        f.write(f"saved_accuracy = {accuracy}\n")
    print("Weights saved successfully to 'weights.py'")
except Exception as e:
    print(f"Failed to write to weights.py: {e}")
