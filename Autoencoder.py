import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def initialize_weights(n_in, n_out):
    #       Initializes a weight matrix.

    r = np.sqrt(6) / np.sqrt(n_out + n_in)
    return np.random.uniform(-r, r, [n_out, n_in])

def mean_squared_error(Y, X):
    #    Computes the mean squared error.
    L = np.mean(np.sum((X-Y)**2, axis=0))
    return L

def relu(A):
    #   Computes the rectified linear activation.

    Y = np.maximum(A, 0)
    return Y

def linear(X, W):
    #    Computes the activities for a fully connected layer.

    A = np.dot(W,X)
    return A

def backward(X, A1, H, Y, W2):
    #    Computes the backward pass for a two-layer network with sigmoid and softmax units, and cross-entropy loss.

    delta_2 = Y - X # P N
    delta_1 = (A1 > 0).astype('float') * np.dot(W2.T, delta_2) # Q N * Q N = Q N

    dW2 = np.dot(delta_2, H.T) # P N x N Q = P Q
    dW1 = np.dot(delta_1, X.T) # Q N x N P = Q P

    return dW1, dW2


def train_network(X_train, X_val, n_hidden=30, n_epochs=2000, eta=10 ** -5):
    #    Performs the training procedure for a two-layer MLP with ReLU hidden units and MSE.


    P = X_train.shape[0]
    Q = n_hidden
    # Initialize W1 and W2 (use initialize_weights())
    W1 = initialize_weights(P, Q)
    W2 = initialize_weights(Q, P)

    # Loop over epochs
    train_loss = np.zeros((n_epochs))
    val_loss = np.zeros((n_epochs))
    for i_epoch in range(n_epochs):

        # Forward pass
        A1, H, Y_train = forward(X_train, W1, W2)
        _, _, Y_val = forward(X_val, W1, W2)

        # Backward pass
        dW1, dW2 = backward(X_train, A1, H, Y_train, W2)

        # Parameter update
        W1 -= dW1 * eta
        W2 -= dW2 * eta

        # Save loss
        train_loss[i_epoch] = mean_squared_error(Y_train, X_train)
        val_loss[i_epoch] = mean_squared_error(Y_val, X_val)

        # Print progress and loss
        if i_epoch % 50 == 0:
            print("Epoch {}/{}. Train loss: {:.3f}. Validation loss: {:.3f}.".format(
                1 + i_epoch, n_epochs, train_loss[i_epoch], val_loss[i_epoch]))

    return W1, W2, train_loss, val_loss

# Read dataset
maxsz = [32, 32]
X = []
i = 0
for file_name in os.listdir(os.path.join(os.getcwd(), "yalefaces")):
    if file_name[:7] != "subject":
        continue
    im = Image.open(os.path.join(os.getcwd(), "yalefaces", file_name))
    im.thumbnail(maxsz, Image.ANTIALIAS)
    data = np.asarray(im)
    if i == 0:
        sz = data.shape
    X.append(np.ndarray.flatten(data))
    i += 1
X = np.array(X).astype("float32")
X = X.T

# Print dimensions
print("Dimensions of face data set: {}.".format(X.shape))

# Plot some examples
fig, ax = plt.subplots(2, 5, figsize=(12, 4))
ax = ax.flatten()
for i_example in range(len(ax)):
    ax[i_example].imshow(X[:, i_example*10].reshape((sz)), cmap="gray")
    ax[i_example].set_xticks([], [])
    ax[i_example].set_yticks([], [])

# Standardise data to zero mean and unit variance. We are doing this manually here as we need
# mu and sigma to revert this operation (add mu, multiply by sigma) when
# visualizing the resulting reconstructed faces.

mu = X.mean(axis=1)
sigma = X.std(axis=1)

X -= np.expand_dims(mu, 1)
X /= np.expand_dims(sigma, 1)

# Split dataset in training, validation, and testing split
X_train = X[:, :100]
X_val = X[:, 100:140]
X_test = X[:, 140:]

# Print dimensions
print("Dimensions training inputs: {}".format(X_train.shape))
print("Dimensions validation inputs: {}".format(X_val.shape))
print("Dimensions testing inputs: {}".format(X_test.shape))

# Train the network
w1, w2, train_loss, val_loss = train_network(X_train, X_val)

# Plot the training and validation losses
plt.plot(np.arange(1,2001),train_loss)
plt.plot(np.arange(1,2001),val_loss)
plt.xscale('log')
plt.xlabel('n_epochs')
plt.ylabel('Loss')
plt.legend(['train','validation'])

# Open a 5-by-2 figure with plt.subplots
fig, ax = plt.subplots(5, 2, figsize=(4, 8))
ax = ax.flatten()

# Loop over 5
for i_example in range(0, 10, 2):
    # Take a new example from X (the autoencoder input)
    ind = np.random.choice(X_test.shape[1])
    test = X_test[:, ind]

    # Forward pass of X through the network, obtain the reconstruction y
    _, _, y_test = forward(test, w1, w2)

    # Plot the original X
    test *= sigma
    test += mu
    ax[i_example].imshow(test.reshape((sz)), cmap="gray")
    ax[i_example].set_xticks([], [])
    ax[i_example].set_yticks([], [])

    # Plot the reconstructed X (output Y)
    y_test *= sigma
    y_test += mu
    ax[i_example + 1].imshow(y_test.reshape((sz)), cmap="gray")
    ax[i_example + 1].set_xticks([], [])
    ax[i_example + 1].set_yticks([], [])

ax[0].set_title('Original image')
ax[1].set_title('Reconstructed image')

# Visualize W1
n_hidden = w1.shape[0]

plt.figure(figsize=([8,6]))
for i in range(n_hidden):
    plt.subplot(5,6,i+1)
    fig = plt.imshow(w1[i, :].reshape((sz)), cmap='gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
plt.suptitle('W1 representation')


# Visualize W2
plt.figure(figsize=([8,6]))
for i in range(n_hidden):
    plt.subplot(5,6,i+1)
    fig = plt.imshow(w2[:, i].reshape((sz)), cmap='gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
plt.suptitle('W2 representation')