import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

# Number of data points
N = 2000

# Create noise and time arrays
noise = rng.uniform(low=-1, high=1, size=N)
t = np.linspace(0, 800, N)

# Create data array
d = np.cos(0.1 * t + 1) + 2 * np.cos(0.15 * t + 2) + 5 * np.cos(0.30 * t + 3) + 2 * np.cos(0.31 * t + 4) + 3 * np.cos(t + 5) + noise

# Guess model frequencies
model_frequencies = [0.10, 0.15, 0.30, 0.31, 1.00]

# Initalize model functions and Gram matrix.
r = len(model_frequencies)
m = 2 * r
G = np.zeros(m, dtype=object)
g = np.zeros((2 * r, 2 * r))

# Populate model function arrays (Bretthorst page 32).
for j in range(r):
    G[j] = np.cos(model_frequencies[j] * t)
    G[j + r] = np.sin(model_frequencies[j] * t)

# Populate Gram matrix (Bretthorst page 32).
for j in range(r):
    for k in range(r):
        g[j, k] = np.sum(G[j] * G[k])
        g[j + r, k] = np.sum(G[j + r] * G[k])
        g[j, k + r] = np.sum(G[j] * G[k + r])
        g[j + r, k + r] = np.sum(G[j + r] * G[k + r])

# Find eigenvalues and eigenvectors (Bretthorst 33).
eigenvalues, eigenvectors = np.linalg.eigh(g)

print("Eigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)
print("\ng matrix:\n", g)

# Find orthonormal basis functions (Bretthorst Eq. 3.5).
H = np.zeros(m, dtype=object)
for j, _ in enumerate(H):
    for k in range(m):
        H[j] += (1 / np.sqrt(eigenvalues[j])) * eigenvectors[k][j] * G[k]

# Find projections of data onto orthonormal basis functions, orthonormal amplitudes (Bretthorst Eq. 3.13).
h = np.zeros(m)
for j, _ in enumerate(h):
    h[j] = np.sum(d * H[j])

# Create model function from guess parameters.
model = np.zeros(N)
for j, _ in enumerate(h):
    model += h[j] * H[j]

# Calculate dbar (Bretthorst page 17).
dbar = (1 / N) * np.sum(d ** 2)
# Calculate hbar (Bretthorst Eq. 3.15).
hbar = (1 / m) * np.sum(h ** 2)

# Find probability (Bretthorst Eq. 3.17).
ratio = (m * hbar) / (N * dbar)
log10_probability = 0.5 * (m - N) * np.log10(1 - ratio)

print("Log10 probability: ", log10_probability)

# Find estimated variance (Bretthorst Eq. 4.7).
variance = (1 / (N - m - 2)) * (np.sum(d ** 2) - np.sum(h ** 2))
print("Estimated variance: ", variance)

# Find SNR (Bretthorst Eq. 4.8).
SNR = ((m / N) * (1 + hbar / variance)) ** (0.5)
print("SNR: ", SNR)

plt.plot(t, d, color="sandybrown", alpha=0.8)
plt.plot(t, model, color="cornflowerblue", alpha=0.8)

plt.xlim(min(t), max(t))
plt.xlabel("Time (s)")
plt.ylabel("Intensity")
plt.title("Replication of Bretthorst Figure 6.11")
plt.show()
