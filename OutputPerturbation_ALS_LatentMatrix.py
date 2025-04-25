import numpy as np
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error

def compute_rmse(R, P, Q):
    
    R_pred = np.dot(P, Q.T)
    mask = R > 0  # Only consider observed ratings
    mse = mean_squared_error(R[mask], R_pred[mask])
    return np.sqrt(mse)


def matrix_factorization_with_output_perturbation_als(R, K, lambda_reg, epsilon, max_iter=100, tol=1e-4):

    n_users, n_items = R.shape

    P = np.random.rand(n_users, K)
    Q = np.random.rand(n_items, K)

    prev_rmse = float('inf')

    for iteration in range(max_iter):
        for u in range(n_users):
            rated_items = np.where(R[u, :] > 0)[0]  
            if len(rated_items) == 0:
                continue
            Q_rated = Q[rated_items, :]
            R_u = R[u, rated_items]
            P[u, :] = solve(Q_rated.T @ Q_rated + lambda_reg * np.eye(K), Q_rated.T @ R_u)

        # Solve for Q (items)
        for i in range(n_items):
            rated_users = np.where(R[:, i] > 0)[0]  # Get indices of users who rated item i
            if len(rated_users) == 0:
                continue
            P_rated = P[rated_users, :]
            R_i = R[rated_users, i]
            Q[i, :] = solve(P_rated.T @ P_rated + lambda_reg * np.eye(K), P_rated.T @ R_i)
        
        rmse = compute_rmse(R, P, Q)

        # Check for convergence
        if abs(prev_rmse - rmse) < tol:
            print("Converged!")
            break
        prev_rmse = rmse

    R_predicted = np.dot(P, Q.T)

    R_perturbed = R + np.random.normal(0, 0.1, R.shape)  # Small Gaussian noise
    P_perturbed = np.random.rand(n_users, K)
    Q_perturbed = np.random.rand(n_items, K)

    for iteration in range(max_iter):
        for u in range(n_users):
            rated_items = np.where(R_perturbed[u, :] > 0)[0]
            if len(rated_items) == 0:
                continue
            Q_rated = Q_perturbed[rated_items, :]
            R_u = R_perturbed[u, rated_items]
            P_perturbed[u, :] = solve(Q_rated.T @ Q_rated + lambda_reg * np.eye(K), Q_rated.T @ R_u)

        # Solve for Q_perturbed (items)
        for i in range(n_items):
            rated_users = np.where(R_perturbed[:, i] > 0)[0]
            if len(rated_users) == 0:
                continue
            P_rated = P_perturbed[rated_users, :]
            R_i = R_perturbed[rated_users, i]
            Q_perturbed[i, :] = solve(P_rated.T @ P_rated + lambda_reg * np.eye(K), P_rated.T @ R_i)

    R_predicted_perturbed = np.dot(P_perturbed, Q_perturbed.T)

    # Calculate the sensitivity as the maximum absolute difference
    sensitivity = np.max(np.abs(R_predicted - R_predicted_perturbed))

    noise_scale = sensitivity / epsilon

    P_noisy = P + np.random.laplace(0, noise_scale, P.shape)
    Q_noisy = Q + np.random.laplace(0, noise_scale, Q.shape)

    R_predicted_noisy = np.dot(P_noisy, Q_noisy.T)
    R_predicted_noisy = np.clip(R_predicted_noisy, 0, 5)  # Clip ratings to [0, 5]

    return P_noisy, Q_noisy, R_predicted_noisy, sensitivity
