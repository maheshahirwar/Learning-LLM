# Multi-head Attention Implementation from scratch using NumPy

import numpy as np

# Step 1: Input embeddings (3 tokens, embedding dim = 4)
X = np.array([
    [1.0, 0.0, 1.0, 0.0],  # I
    [0.0, 2.0, 0.0, 2.0],  # love
    [1.0, 1.0, 1.0, 1.0]   # AI
])

# Step 5: Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def multi_head_attention(X, num_heads=2):
    head_dim = X.shape[1] // num_heads
    outputs = []

    for i in range(num_heads):
        Wq = np.random.randn(X.shape[1], head_dim)
        Wk = np.random.randn(X.shape[1], head_dim)
        Wv = np.random.randn(X.shape[1], head_dim)

        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv

        scores = Q @ K.T / np.sqrt(head_dim)
        weights = softmax(scores)
        out = weights @ V

        outputs.append(out)

    return np.concatenate(outputs, axis=-1)

multi_output = multi_head_attention(X)
print("\nMulti-head output:\n", multi_output)
