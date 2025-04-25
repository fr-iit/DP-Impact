import numpy as np
import DataLoader as DL
import Evaluation as EV
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error

def laplace_noise(scale):
    return np.random.laplace(loc=0, scale=scale)

def blp_mechanism(r, l, u, epsilon):
    b = (u - l) / epsilon  # Calculate the sensitivity parameter b
    while True:
        
        noise = laplace_noise(b)

        r_star = r + noise

        if l <= r_star <= u:
            return r_star
        else:
            continue

def bound_pertubation(R, epsilon):
    l = 1.0
    u = 5.0
    n_users = R.shape[0]
    n_items = R.shape[1]

    perturbed_ratings = np.zeros_like(R)
    for i in range(n_users):
        for j in range(n_items):
            if R[i, j] > 0:
                perturbed_ratings[i, j] = np.round(blp_mechanism(R[i, j], l, u, epsilon), 2)
                # print(perturbed_ratings[i, j])

    return perturbed_ratings


def save_perturbed_ratings(R, filename):
    with open(filename, 'w') as f:
        for user_id in range(R.shape[0]):
            for item_id in range(R.shape[1]):
                if R[user_id, item_id] > 0:  # Only save rated items
                    f.write(f"{user_id + 1}::{item_id + 1}::{R[user_id, item_id]:.2f}::000000000\n")

    print(f"Perturbed ratings saved to {filename}")

