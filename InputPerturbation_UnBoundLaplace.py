import numpy as np
from numpy.linalg import solve

def compute_item_averages(R, beta_m, epsilon_1, epsilon_2, r_min, r_max):

    n_users, m_items = R.shape

    # Step 1: Compute the global average (GAvg) with Laplace noise
    delta_r = r_max - r_min
    global_avg = (np.sum(R) / (n_users * m_items)) + np.random.laplace(0, delta_r / epsilon_1)

    # Step 2: Compute item averages (IAvg) with Laplace noise
    IAvg = np.zeros(m_items)
    for j in range(m_items):
        # Extract ratings for item j
        R_j = R[:, j]
        num_ratings_j = np.count_nonzero(R_j)  # Number of non-zero ratings for item j

        # Sum of ratings for item j
        sum_ratings_j = np.sum(R_j)

        # Compute the differentially private item average
        IAvg[j] = (sum_ratings_j + beta_m * global_avg + np.random.laplace(0, delta_r / epsilon_2)) / (
                    num_ratings_j + beta_m)

        # print(IAvg[j])

        # Clamp the result to [r_min, r_max]
        IAvg[j] = np.clip(IAvg[j], r_min, r_max)

    return IAvg


def compute_user_averages(R, IAvg, beta_u, epsilon_1, epsilon_2):

    n_users, m_items = R.shape

    # Step 1: Compute the adjusted ratings matrix R'
    R_prime = R - IAvg

    # Step 2: Compute the global adjusted average (GAvg') with Laplace noise
    delta_r = 2  # Sensitivity (maximum rating difference for clamping to [-2, 2])
    global_avg_prime = (np.sum(R_prime) / (n_users * m_items)) + np.random.laplace(0, delta_r / epsilon_1)

    # Step 3: Compute user averages (UAvg)
    UAvg = np.zeros(n_users)
    for u in range(n_users):
        # Extract adjusted ratings for user u
        R_u = R_prime[u, :]
        num_ratings_u = np.count_nonzero(R_u)  # Number of non-zero ratings for user u

        # Sum of adjusted ratings for user u
        sum_ratings_u = np.sum(R_u)

        # Compute the differentially private user average
        UAvg[u] = (sum_ratings_u + beta_u * global_avg_prime + np.random.laplace(0, delta_r / epsilon_2)) / (
                    num_ratings_u + beta_u)

        # Clamp the user average to the range [-2, 2]
        UAvg[u] = np.clip(UAvg[u], -2, 2)

    return UAvg


def calculate_discounted_and_clamped_matrix(R, IAvg, UAvg):

    n_users, n_items = R.shape

    # Initialize the adjusted matrix
    R_discounted = np.zeros_like(R)

    # Loop through users and items
    for u in range(n_users):
        for i in range(n_items):
            # Discount the item and user averages
            R_discounted[u, i] = R[u, i] - (IAvg[i] + UAvg[u])
            # Clamp the result to the range [-B, B]
            # R_discounted[u, i] = np.clip(R_discounted[u, i], -B, B)

    return R_discounted

def input_perturbation(R, delta_r, epsilon, B):
    noise = np.random.laplace(0, delta_r / epsilon)
    noise_R = R + noise
    return np.clip(noise_R, -B, B)

def als_matrix_factorization(R, d, lambda_reg, max_iters=150, tol=1e-4):
    """
    Perform ALS-based matrix factorization.
    """
    n_users, n_items = R.shape
    P = np.random.rand(n_users, d)
    Q = np.random.rand(n_items, d)

    prev_rmse = float('inf')
    for iteration in range(max_iters):
        # Solve for P
        for u in range(n_users):
            rated_items = np.where(R[u, :] > 0)[0]
            if len(rated_items) == 0:
                continue
            Q_rated = Q[rated_items, :]
            R_u = R[u, rated_items]
            P[u, :] = solve(Q_rated.T @ Q_rated + lambda_reg * np.eye(d), Q_rated.T @ R_u)

        # Solve for Q
        for i in range(n_items):
            rated_users = np.where(R[:, i] > 0)[0]
            if len(rated_users) == 0:
                continue
            P_rated = P[rated_users, :]
            R_i = R[rated_users, i]
            Q[i, :] = solve(P_rated.T @ P_rated + lambda_reg * np.eye(d), P_rated.T @ R_i)

        # Compute RMSE
        R_pred = np.dot(P, Q.T)
        mask = R > 0
        rmse = np.sqrt(np.mean((R[mask] - R_pred[mask]) ** 2))
        # print(f"Iteration {iteration + 1}: RMSE = {rmse:.4f}")

        # Check for convergence
        if abs(prev_rmse - rmse) < tol:
            print(f"Iteration {iteration + 1} Converged!")
            break
        prev_rmse = rmse

    return P, Q

def predict_ratings(P, Q, avgI, avgU):
    return np.dot(P, Q.T) + avgI.reshape(1, -1) + avgU.reshape(-1, 1)

def save_perturbed_ratings(R, filename):
    with open(filename, 'w') as f:
        for user_id in range(R.shape[0]):
            for item_id in range(R.shape[1]):
                if R[user_id, item_id] > 0:  # Only save rated items
                    f.write(f"{user_id + 1}::{item_id + 1}::{R[user_id, item_id]:.2f}::000000000\n")

    print(f"Perturbed ratings saved to {filename}")
