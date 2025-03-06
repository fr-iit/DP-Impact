import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve

def laplace_noise(scale, shape):
    """Generate Laplace noise with given scale and shape."""
    return np.random.laplace(0, scale, shape)

def vp_dpmf(P, R, max_q, min_q, lambda_Q, epsilon, delta_r):
    n_users, n_items = R.shape
    K = P.shape[1]  # Number of latent features
    Q = np.random.rand(n_items, K)  # Initialize Q randomly
    Q_perturbed = np.zeros_like(Q)
    max_norm = 5.0

    for i in range(n_items):
        # Find users who have rated item i
        users_who_rated = np.where(R[:, i] > 0)[0]
        if len(users_who_rated) == 0:
            continue

        P_ui = P[users_who_rated, :]  # Subset of P for users who rated item i
        r_i = R[users_who_rated, i]  # Ratings for item i

        # Compute Delta
        max_user_factor_norm = max(np.linalg.norm(P[u, :], ord=1) for u in users_who_rated)
        Delta = max_user_factor_norm * delta_r

        # Generate Laplace noise
        noise = laplace_noise(Delta / epsilon, (K, 1))

        # Compute \bar{q_i} using vector perturbation
        # Q_perturbed[i, :] = inv(P_ui.T @ P_ui + lambda_Q * np.eye(K)) @ (P_ui.T @ r_i - noise.flatten())
        # Q_perturbed[i, :] = np.clip(Q_perturbed[i, :], min_q, max_q)

        # Compute \bar{q_i} using vector perturbation
        q_i_perturbed = inv(P_ui.T @ P_ui + lambda_Q * np.eye(K)) @ (P_ui.T @ r_i - noise.flatten())

        # Normalize q_i if its norm exceeds max_norm
        if np.linalg.norm(q_i_perturbed) > max_norm:
            q_i_perturbed = (q_i_perturbed / np.linalg.norm(q_i_perturbed)) * max_norm

        # Clip each element of q_i within [min_q, max_q]
        q_i_perturbed = np.clip(q_i_perturbed, min_q, max_q)

        # Assign the processed q_i to the perturbed Q matrix
        Q_perturbed[i, :] = q_i_perturbed


    print(f"original, max: {max_q}, min: {min_q}")
    print(f'max Q: {np.max(Q_perturbed)} and min Q for perturb: {np.min(Q_perturbed)}')
    return Q_perturbed

def als_matrix_factorization(R, K, lambda_reg, max_iters=150, tol=1e-4):

    n_users, n_items = R.shape
    P = np.random.rand(n_users, K)
    Q = np.random.rand(n_items, K)

    prev_rmse = float('inf')

    for iteration in range(max_iters):
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
        R_pred = np.dot(P, Q.T)
        mask = R > 0  # Only consider observed ratings
        rmse = np.sqrt(np.mean((R[mask] - R_pred[mask]) ** 2))

        print(f"Iteration {iteration + 1}: RMSE = {rmse:.4f}")

        # Check for convergence
        if abs(prev_rmse - rmse) < tol:
            print("Converged!")
            break
        prev_rmse = rmse

    return P, Q

def predict_ratings(P, Q):
    return np.dot(P, Q.T)

