import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt


def compute_log_probability(q):
    x = q[0]
    y = q[1]

    sigmax = 1
    sigmay = 1

    log_probability = -(((x ** 2) / (2 * sigmax ** 2)) + ((y ** 2) / (2 * sigmay ** 2)))

    return log_probability


def compute_gradient(q, delta=1e-5):
    gradient = np.zeros_like(q)

    for i, _ in enumerate(q):
        q_plus = np.array(q)
        q_minus = np.array(q)

        q_plus[i] += delta
        q_minus[i] -= delta

        log_probability_plus = compute_log_probability(q_plus)
        log_probability_minus = compute_log_probability(q_minus)

        gradient[i] = (log_probability_plus - log_probability_minus) / (2 * delta)

    return gradient


current_x = 2
current_y = 2

M = 50
epsilon = 0.1
L = 20

ps = np.zeros((M, 2))
qs = np.zeros((M, 2))

progress = tqdm.tqdm(total=M)
log_probabilities = np.zeros(M)

q = [current_x, current_y]
for iteration in range(M):
    p = np.array([random.gauss(0, 1) for _ in range(2)]) * 0.05
    
    for _ in range(L):
        gradient = compute_gradient(q)
        for i, _ in enumerate(q):
            p[i] = p[i] + (epsilon / 2.0) * gradient[i]

        for i, _ in enumerate(q):
            q[i] = q[i] + epsilon * p[i]

        updated_gradient = compute_gradient(q)

        for i, _ in enumerate(q):
            p[i] = p[i] + (epsilon / 2.0) * updated_gradient[i]

    ps[iteration] = p
    qs[iteration] = q
    log_probabilities[iteration] = compute_log_probability(q)

    progress.update(1)


# Extracting the 'x' dimension components
q_x = qs[:, 0]
p_x = ps[:, 0]

# Extracting the 'y' dimension components
q_y = qs[:, 1]
p_y = ps[:, 1]

log_probabilities = list(log_probabilities)
best = log_probabilities.index(max(log_probabilities))
print(q_x[best], q_y[best])

plt.plot(q_x, q_y)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
