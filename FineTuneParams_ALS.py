from sklearn.model_selection import KFold
import ALS as als
import Evaluation as EV
import DataLoader as DL
from itertools import product
import numpy as np

def mask_split(R, train_idx, test_idx):
    
    R_train, R_test = np.zeros_like(R), np.zeros_like(R)
    R_train[train_idx, :] = R[train_idx, :]
    R_test[test_idx, :] = R[test_idx, :]
    return R_train, R_test

def masked_rmse(R_actual, R_predicted):
    mask = R_actual > 0  # Only consider nonzero ratings
    return np.sqrt(np.mean((R_actual[mask] - R_predicted[mask])**2))

def cross_validate_als(R, k_values, lambda_values, max_iters_values, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    best_rmse = float('inf')
    best_params = None
    best_K = 0
    best_reg = 0
    best_iter = 0


    for K, lambda_reg, max_iter in product(k_values, lambda_values, max_iters_values):
        total_rmse = 0
        print(f"K={K}, λ={lambda_reg}, max_iter={max_iter}")
        for train_idx, test_idx in kf.split(R):
            R_train, R_test = mask_split(R, train_idx, test_idx)
            P, Q = als.als_explicit(R_train, K=K, lambda_reg=lambda_reg, max_iter=max_iter)
            R_predicted = P @ Q.T  # Predict ratings

            total_rmse += masked_rmse(R_test, R_predicted)  # Compute RMSE only for known ratings

        avg_rmse = total_rmse / num_folds
        print(f"K={K}, λ={lambda_reg}, max_iter={max_iter} → Avg RMSE={avg_rmse:.4f}")

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            # best_params = (K, lambda_reg, max_iter)
            best_K = K
            best_reg = lambda_reg
            best_iter = max_iter

    # print(f"\nBest Parameters: K={best_params[0]}, λ={best_params[1]}, max_iter={best_params[2]}, RMSE={best_rmse:.4f}")
    print(f"\nBest Parameters: K={best_K}, λ={best_reg}, max_iter={best_iter}, RMSE={best_rmse:.4f}")
    return best_K, best_reg, best_iter

# Load dataset
R = DL.load_user_item_matrix_10m()
R = R.astype(np.float32)
# best_params
best_K, best_reg, best_iter = cross_validate_als(R, [10, 15, 20, 25, 50], [0.05, 0.09, 0.1, 0.5, 0.9, 1, 1.2,1.5, 1.7, 1.8, 1.9, 2.0, 2.1], [10, 20, 50, 100, 150, 200])
