import numpy as np
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error

def compute_rmseALS(R, P, Q):
    """Compute RMSE between actual and predicted ratings."""
    R_pred = np.dot(P, Q.T)
    mask = R > 0  # Only consider observed ratings
    mse = mean_squared_error(R[mask], R_pred[mask])
    return np.sqrt(mse)

def matrix_factorization_with_output_perturbation_als(R, K, lambda_reg, epsilon, max_iter=150, tol=1e-4):

    n_users, n_items = R.shape
    # P = np.random.rand(n_users, K)
    # Q = np.random.rand(n_items, K)
    P = np.abs(np.random.rand(n_users, K))
    Q = np.abs(np.random.rand(n_items, K))

    prev_rmse = float('inf')

    for iteration in range(max_iter):
        # Solve for P (users)
        for u in range(n_users):
            rated_items = np.where(R[u, :] > 0)[0]  # Get indices of rated items
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

        # Compute RMSE
        rmse = compute_rmseALS(R, P, Q)
        # print(f"Iteration {iteration + 1}: RMSE = {rmse:.4f}")

        # Check for convergence
        if abs(prev_rmse - rmse) < tol:
            print(f"Converged! {iteration + 1}")
            break
        prev_rmse = rmse

    # Compute the predicted ratings matrix
    R_predicted = np.dot(P, Q.T)

    # print("Predicted Ratings Matrix (before noise):")
    # print(R_predicted)

    sensitivity = 5 #np.max(R_predicted) - np.min(R_predicted)

    # Add Laplace noise only to observed ratings for output perturbation
    noise_scale = sensitivity / epsilon
    noise = np.random.laplace(0, noise_scale) #, R.shape
    # R_predicted_noisy = R_predicted + (noise * (R > 0))  # Apply noise only to non-zero entries
    R_predicted_noisy = R_predicted + noise  # Apply noise only to non-zero entries

    # Clip noisy predicted ratings to the 5-star scale
    R_predicted_noisy = np.clip(R_predicted_noisy, 0, 5)

    return R_predicted_noisy, sensitivity

