import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# datasets
letters_binary = {
    'A': [0,1,1,1,1,0,
          1,0,0,0,0,1,
          1,1,1,1,1,1,
          1,0,0,0,0,1,
          1,0,0,0,0,1],

    'B': [1,1,1,1,1,0,
          1,0,0,0,0,1,
          1,1,1,1,1,0,
          1,0,0,0,0,1,
          1,1,1,1,1,0],

    'C': [0,1,1,1,1,1,
          1,0,0,0,0,0,
          1,0,0,0,0,0,
          1,0,0,0,0,0,
          0,1,1,1,1,1],

    'D': [1,1,1,1,0,0,
          1,0,0,0,1,0,
          1,0,0,0,0,1,
          1,0,0,0,1,0,
          1,1,1,1,0,0],

    'E': [1,1,1,1,1,1,
          1,0,0,0,0,0,
          1,1,1,1,0,0,
          1,0,0,0,0,0,
          1,1,1,1,1,1],

    'F': [1,1,1,1,1,1,
          1,0,0,0,0,0,
          1,1,1,1,0,0,
          1,0,0,0,0,0,
          1,0,0,0,0,0],

    'G': [0,1,1,1,1,1,
          1,0,0,0,0,0,
          1,0,0,1,1,1,
          1,0,0,0,0,1,
          0,1,1,1,1,1],

    'H': [1,0,0,0,0,1,
          1,0,0,0,0,1,
          1,1,1,1,1,1,
          1,0,0,0,0,1,
          1,0,0,0,0,1],

    'I': [1,1,1,1,1,1,
          0,0,0,1,0,0,
          0,0,0,1,0,0,
          0,0,0,1,0,0,
          1,1,1,1,1,1],

    'J': [0,0,0,0,1,1,
          0,0,0,0,0,1,
          0,0,0,0,0,1,
          1,0,0,0,0,1,
          1,1,1,1,1,1],

    'K': [1,0,0,1,0,0,
          1,0,1,0,0,0,
          1,1,0,0,0,0,
          1,0,1,0,0,0,
          1,0,0,1,0,0],

    'L': [1,0,0,0,0,0,
          1,0,0,0,0,0,
          1,0,0,0,0,0,
          1,0,0,0,0,0,
          1,1,1,1,1,1],

    'M': [1,0,0,0,0,1,
          1,1,0,0,1,1,
          1,0,1,1,0,1,
          1,0,0,0,0,1,
          1,0,0,0,0,1],

    'N': [1,0,0,0,0,1,
          1,1,0,0,0,1,
          1,0,1,0,0,1,
          1,0,0,1,0,1,
          1,0,0,0,1,1],

    'O': [0,1,1,1,1,0,
          1,0,0,0,0,1,
          1,0,0,0,0,1,
          1,0,0,0,0,1,
          0,1,1,1,1,0],

    'P': [1,1,1,1,1,0,
          1,0,0,0,0,1,
          1,1,1,1,1,0,
          1,0,0,0,0,0,
          1,0,0,0,0,0],

    'Q': [0,1,1,1,1,0,
          1,0,0,0,0,1,
          1,0,0,0,0,1,
          1,0,0,1,0,1,
          0,1,1,1,1,1],

    'R': [1,1,1,1,1,0,
          1,0,0,0,0,1,
          1,1,1,1,1,0,
          1,0,0,1,0,0,
          1,0,0,0,1,0],

    'S': [0,1,1,1,1,1,
          1,0,0,0,0,0,
          0,1,1,1,1,0,
          0,0,0,0,0,1,
          1,1,1,1,1,0],

    'T': [1,1,1,1,1,1,
          0,0,0,1,0,0,
          0,0,0,1,0,0,
          0,0,0,1,0,0,
          0,0,0,1,0,0],

    'U': [1,0,0,0,0,1,
          1,0,0,0,0,1,
          1,0,0,0,0,1,
          1,0,0,0,0,1,
          0,1,1,1,1,0],

    'V': [1,0,0,0,0,1,
          1,0,0,0,0,1,
          0,1,0,0,1,0,
          0,1,0,0,1,0,
          0,0,1,1,0,0],

    'W': [1,0,0,0,0,1,
          1,0,0,0,0,1,
          1,0,0,1,0,1,
          1,0,1,0,1,1,
          0,1,0,0,1,0],

    'X': [1,0,0,0,0,1,
          0,1,0,0,1,0,
          0,0,1,1,0,0,
          0,1,0,0,1,0,
          1,0,0,0,0,1],

    'Y': [1,0,0,0,0,1,
          0,1,0,0,1,0,
          0,0,1,1,0,0,
          0,0,0,1,0,0,
          0,0,0,1,0,0],

    'Z': [1,1,1,1,1,1,
          0,0,0,0,1,0,
          0,0,0,1,0,0,
          0,0,1,0,0,0,
          1,1,1,1,1,1]
}

#one hot label - y
one_hot_labels = {
    chr(i + 65): [1 if j == i else 0 for j in range(26)]
    for i in range(26)
}


# activation function - retune the value between 0 and 1 for any real number
# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# creating feed forward NN function
def f_forward(x, w1, w2):
    # hidden layer
    z1 = x @ w1 
    a1 = sigmoid(z1)

    # output layer 
    z2 = a1 @ w2
    a2 = sigmoid(z2)

    return a2

# initializing weights
def generate_wt(x, y):
    l = []
    for i in range(x*y):
        l.append(np.random.randn())
    return np.array(l).reshape(x, y)

# calculating the loss - mean squared error
def loss(output, y):
    s = (np.square(output - y))
    s = np.sum(s)/len(y)
    return s

# backpropagation
def back_propagation(x, y, w1, w2, learning_rate):
    # hidden layer
    z1 = x @ w1 
    a1 = sigmoid(z1)

    # output layer 
    z2 = a1 @ w2
    a2 = sigmoid(z2)

    # calculating the loss
    d2 = (a2 - y)
    d1 = np.multiply((d2 @ w2.T), np.multiply(a1, (1 - a1)))

    # updating the weights
    w1_adj = x.T @ d1
    w2_adj = a1.T @ d2

    w1 = w1 - (learning_rate * w1_adj)
    w2 = w2 - (learning_rate * w2_adj)

    return w1, w2

# training the model 
def train(x, y, w1, w2, learning_rate, epochs=10):
    accuracy = []
    losses = []
    for j in range(epochs):
        l = []
        for i in range(len(x)):
            output = f_forward(x[i], w1, w2)
            l.append(loss(output, y[i]))
            w1, w2 = back_propagation(x[i], y[i], w1, w2, learning_rate)
        # print("Epoch : ", j+1, " Accuracy : ", (1 - (sum(l)/len(x)))*100, "%")
        accuracy.append((1 - (sum(l)/len(x)))*100)
        losses.append(sum(l)/len(x))
    return w1, w2, accuracy, losses

# predict the model 
def predict(x, w1, w2):
    output = f_forward(x, w1, w2)

    maxm = 0
    k = 0

    for i in range(len(output[0])):
        if output[0][i] > maxm:
            maxm = output[0][i]
            k = i

    print("Predicted character : ", chr(k + 65))
    plt.imshow(x.reshape(5, 6))
    plt.title(f'Character: {chr(k + 65)}')
    plt.show()