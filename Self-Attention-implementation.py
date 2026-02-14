# Implementing self-attention from scratch using NumPy

import numpy as np

# Step 1: Input embeddings (3 tokens, embedding dim = 4)
X = np.array([
    [1.0, 0.0, 1.0, 0.0],  # I
    [0.0, 2.0, 0.0, 2.0],  # love
    [1.0, 1.0, 1.0, 1.0]   # AI
])

# Step 2: Random weight matrices (4x4)
np.random.seed(42)
Wq = np.random.randn(4, 4)
Wk = np.random.randn(4, 4)
Wv = np.random.randn(4, 4)

# Step 3: Compute Q, K, V
Q = X @ Wq
K = X @ Wk
V = X @ Wv

# Step 4: Attention scores
d_k = K.shape[1]
scores = Q @ K.T / np.sqrt(d_k)

# Step 5: Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores)

# Step 6: Weighted sum
output = attention_weights @ V

print("Attention Weights:\n", attention_weights)
print("\nOutput:\n", output)
