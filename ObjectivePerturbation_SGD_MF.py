import numpy as np
from sklearn.metrics import mean_squared_error

def dp_matrix_factorization(R, K, gamma_init, emax, epsilon, lamb, itr=150, tol=1e-4, decay=0.01):
    n_users, n_items = R.shape
    P = np.random.rand(n_users, K)
    Q = np.random.rand(n_items, K)

    delta_r = 2  # Sensitivity of ratings
    prev_rmse = float('inf')  # Store previous RMSE
    gamma = gamma_init  # Initialize learning rate

    for step in range(itr):
        total_squared_error = 0  # Track sum of squared errors
        count = 0  # Count the number of rated entries

        for u in range(n_users):
            non_zero_indices = np.where(R[u, :] > 0)[0]  # Get rated items by user `u`
            for i in non_zero_indices:
                predicted_rating = np.dot(P[u, :], Q[i, :])
                noisy_error = R[u, i] - predicted_rating + np.random.laplace(0, itr * delta_r / epsilon)

                # Clamp noisy error
                noisy_error = np.clip(noisy_error, -emax, emax)

                # Compute gradients
                P_update = gamma * (noisy_error * Q[i, :] - lamb * P[u, :])
                Q_update = gamma * (noisy_error * P[u, :] - lamb * Q[i, :])

                # Apply updates
                P[u, :] += P_update
                Q[i, :] += Q_update

                # Compute squared error for RMSE calculation
                total_squared_error += noisy_error ** 2
                count += 1  # Count the number of rated elements

        # Compute RMSE for this iteration
        current_rmse = np.sqrt(total_squared_error / count)

        # Print progress
        # print(f"Iteration {step}: RMSE = {current_rmse:.6f}, Learning Rate = {gamma:.6f}")

        # **Dynamic Learning Rate Adjustment**
        if abs(prev_rmse - current_rmse) < tol:
            gamma *= 0.9  # Reduce learning rate by 10% if RMSE change is too small
        elif abs(prev_rmse - current_rmse) > 5 * tol:
            gamma *= 1.05  # Increase learning rate by 5% if RMSE is improving fast

        # Apply Learning Rate Decay Over Time
        gamma = gamma_init / (1 + decay * step)

        # Convergence check
        if abs(prev_rmse - current_rmse) < tol:
            print(f"Converged at iteration {step}: RMSE = {current_rmse:.4f}, Learning Rate = {gamma:.4f}")
            break

        prev_rmse = current_rmse  # Update previous RMSE

    return P, Q

def compute_rmse(R_test, R_predicted):
    """Compute RMSE for the test set."""
    non_zero_indices = R_test > 0
    mse = mean_squared_error(R_test[non_zero_indices], R_predicted[non_zero_indices])
    return np.sqrt(mse)
