import numpy as np
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error
import DataLoader as DL
import Evaluation as EV
import FineTuneParams_ALS as fnParam
from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import combinations
from scipy.spatial.distance import cosine


# --- perturbation methods ---
import InputPerturbation_UnBoundLaplace as als_udp
import InputPerturbation_BoundLaplace as als_bdp
import ObjectivePerturbation_VPMF as vpmf
import OutputPerturbation_ALS_LatentMatrix as alslm
import OutputPerturbation_ALS_PredictedRating as alspr
import ObjectivePerturbation_SGD_MF as sgdmf


def mean_squared_error_10m(y_true, y_pred, chunk_size=500_000):
    """
    Computes MSE in chunks to avoid large intermediate arrays.
    y_true and y_pred must be the same shape (1D).
    """
    n = len(y_true)
    total_sq_error = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        diff = y_true[start:end] - y_pred[start:end]
        total_sq_error += np.sum(diff * diff)
    return total_sq_error / n

def compute_rmseALS_10m(R, P, Q, chunk_size=500_000):
    # Convert to float32
    P = P.astype(np.float32, copy=False)
    Q = Q.astype(np.float32, copy=False)

    R_pred = np.dot(P, Q.T)
    mask = R > 0  # only observed

    # Extract 1D arrays
    y_true = R[mask].astype(np.float32, copy=False)
    y_pred = R_pred[mask]  # already float32
    # Use chunk-based MSE
    mse = mean_squared_error_10m(y_true, y_pred, chunk_size)
    return np.sqrt(mse)

def compute_rmseALS(R, P, Q):
    """Compute RMSE between actual and predicted ratings."""
    R_pred = np.dot(P, Q.T)
    mask = R > 0  # Only consider observed ratings
    mse = mean_squared_error(R[mask], R_pred[mask])
    return np.sqrt(mse)


def als_explicit(R, K, lambda_reg, max_iter=150, tol=1e-4): #, tol=1e-4

    n_users, n_items = R.shape
    P = np.random.rand(n_users, K)
    Q = np.random.rand(n_items, K)

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
            print("Converged!")
            print(f"Iteration {iteration + 1}: RMSE = {rmse:.4f}")
            break
        prev_rmse = rmse

    return P, Q

def predict_ratings(P, Q):

    # for ml10m
    P = P.astype(np.float32)
    Q = Q.astype(np.float32)
    return np.dot(P, Q.T)

def split_data(R, test_size=0.2):

    train = R.copy()
    test = np.zeros_like(R)

    for user_idx in range(R.shape[0]):
        non_zero_indices = np.where(R[user_idx] > 0)[0]
        if len(non_zero_indices) == 0:
            continue
        test_indices = np.random.choice(non_zero_indices, size=int(test_size * len(non_zero_indices)), replace=False)
        train[user_idx, test_indices] = 0  # Mask test ratings in train set
        test[user_idx, test_indices] = R[user_idx, test_indices]  # Save original test ratings

    return train, test

def compute_arp(recommendations, item_popularity):
    arps = []
    for rec_list in recommendations.values():
        valid_items = [int(i) for i in rec_list if 0 <= int(i) < len(item_popularity)]
        arps.append(np.mean([item_popularity[i] for i in valid_items]) if valid_items else 0)
    return np.array(arps)

def compute_item_popularity(interaction_matrix):
    num_users = interaction_matrix.shape[0]
    return np.sum(interaction_matrix > 0, axis=0) / num_users  # Fraction of users who interacted

def compute_gamma(user_profiles, item_popularity):
    gammas = []
    for profile in user_profiles:
        gammas.append(np.mean([item_popularity[i] for i in profile]) if len(profile) > 0 else 0)
    return np.array(gammas)


def compute_rmse_for_user_group(R_test, R_predicted, user_group):
    """Compute RMSE for a specific group of users (cold-start or normal)."""
    mask = R_test[user_group, :] > 0  # Only consider observed ratings
    mse = mean_squared_error(R_test[user_group, :][mask], R_predicted[user_group, :][mask])
    return np.sqrt(mse)

if __name__ == "__main__":

    num_runs = 5  # Number of times to run the experiment
    dataset_name  = 'ml1m' #  1. 'ml100k' ; 2. 'ml1m' ; 3. 'yahoo' ; 4. 'ml10m'
    perturbation_name = 'DP-SGD' # 1. 'NoDP' ; 2. 'UDP-MF' ; 3. 'BDP-MF' ;
                                # 4. 'DP-SGD' ; 5. 'VP-DPMF' ; 6.'DP-LMMF' ; 7. 'DP-PRMF'
    # Store results for each metric
    # accuracy
    rmse_list = []
    recall_list = []
    mrr_list = []

    # popularity bias
    arp_list = []

    # cold-start vs normal
    cold_rmse_list = []
    normal_rmse_list = []
    cold_MRR_list = []
    normal_MRR_list = []

    if dataset_name == 'ml100k':
        R = DL.load_user_item_matrix_100k() #DL.load_rating_matrix_100k_BoundPerturb() #
    elif dataset_name == 'ml1m':
        R = DL.load_user_item_matrix_1m() # DL.load_rating_matrix_1m_BoundPerturb()  #
    elif dataset_name == 'yahoo':
        R = DL.load_user_item_matrix_yahoo() # DL.load_rating_matrix_Yahoo_BoundPerturb()  #
    elif dataset_name == 'ml10m':
        R = DL.load_user_item_matrix_10m()
        R = R.astype(np.float32)

    K, lambda_reg, max_iter = fnParam.cross_validate_als(R, [10, 15, 20, 25, 50],
                                                     [0.05, 0.09, 0.1, 0.5, 0.9, 1, 1.2, 1.5, 1.7, 1.8, 1.9, 2.0, 2.1],
                                                     [10, 20, 50, 100, 150, 200])

    # global variables
    topN = 10
    lambda_Q = 0.1
    delta_r = 4.0
    epsilon = 0.01
    # for sgd mf
    gamma = 0.005
    emax = 2

    user_profiles = DL.user_profile(R)
    item_popularity = compute_item_popularity(R)
    long_tail_items = set(np.where(item_popularity < 0.21)[0]) # Define long-tail items (e.g., items with <20% user interactions)
    U_cold, U_normal = DL.identify_cold_start_users(R)

    # Variables to track the best run
    best_rmse = float('inf')  # Initialize with a high value
    best_recall = 0  # Initialize with the lowest value
    best_run = -1
    best_R_predicted = None  # Store the best predicted rating matrix

    for run in range(num_runs):
        print(f"===== Run {run + 1}/{num_runs} =====")


        # 2) ml-10m: Split data, optionally also downcast
        # R_train = R_train.astype(np.float32, copy=False)
        # R_test = R_test.astype(np.float32, copy=False)

        # perturbation methods
        if perturbation_name == 'NoDP':
            R_train, R_test = split_data(R, test_size=0.2)
            P, Q = als_explicit(R_train, K=K, lambda_reg=lambda_reg, max_iter= max_iter)
            R_predicted = predict_ratings(P, Q)
            best_filename = "NoDP.dat"

        elif perturbation_name == 'UDP-MF':
            # variables and privacy budgets define
            beta_m = 1.0
            beta_u = 1.0
            B = 1
            G_ep = 0.3 * epsilon  # Global average privacy parameter
            I_ep = 0.25 * epsilon  # Item average privacy parameter
            U_ep = 0.25 * epsilon  # Global average privacy parameter for user
            Input_ep = 0.2 * epsilon
            print(f'G_ep: {G_ep}, I_ep: {I_ep}, U_ep: {U_ep}, Input_ep: {Input_ep}')
            r_min, r_max = 1, 5  # Rating limits

            # function calls
            IAvg = als_udp.compute_item_averages(R, beta_m, G_ep, I_ep, r_min, r_max)
            UAvg = als_udp.compute_user_averages(R, IAvg, beta_u, G_ep, U_ep)
            R_discounted_clamped = als_udp.calculate_discounted_and_clamped_matrix(R, IAvg, UAvg)
            input_R = als_udp.input_perturbation(R_discounted_clamped, delta_r=2, epsilon=Input_ep, B=B)
            UPR_train, UPR_test = DL.split_data(input_R, test_size=0.2)
            R_test = UPR_test
            P, Q = als_explicit(UPR_train, K=K, lambda_reg=lambda_reg, max_iter=max_iter)
            R_predicted = als_udp.predict_ratings(P, Q, IAvg, UAvg)
            best_filename = f"InpDP_UDPMF_ep" + str(epsilon) + ".dat"

        elif perturbation_name == 'BDP-MF':
            pertrub_rating = als_bdp.bound_pertubation(R, epsilon)
            BPR_train, BPR_test = split_data(pertrub_rating, test_size=0.2)
            R_test = BPR_test
            P, Q = als_explicit(BPR_train, K=K, lambda_reg=lambda_reg, max_iter=max_iter)
            R_predicted = predict_ratings(P, Q)
            best_filename = f"InpDP_BDPMF_ep" + str(epsilon) + ".dat"

        elif perturbation_name == 'DP-SGD':
            R_train, R_test = split_data(R, test_size=0.2)
            P, Q = sgdmf.dp_matrix_factorization(R_train, K, gamma, emax, epsilon, lamb=lambda_reg, itr=max_iter)
            R_predicted = predict_ratings(P, Q)
            best_filename = f"ObjDP_DP-SGD_ep" + str(epsilon) + ".dat"

        elif perturbation_name == 'VP-DPMF':
            R_train, R_test = split_data(R, test_size=0.2)
            P, Q = vpmf.als_matrix_factorization(R_train, K=K, lambda_reg=lambda_reg, max_iter=max_iter)
            max_q = np.max(Q)
            min_q = np.min(Q)
            Q_perturbed = vpmf.vp_dpmf(P, R_train, max_q, min_q, lambda_Q, epsilon, delta_r)
            R_predicted = predict_ratings(P, Q_perturbed)
            best_filename = f"ObjDP_VP-DPMF_ep" + str(epsilon) + ".dat"

        elif perturbation_name == 'DP-LMMF':
            R_train, R_test = split_data(R, test_size=0.2)
            P_noisy, Q_noisy, R_predicted, sensitivity = alslm.matrix_factorization_with_output_perturbation_als(
                R_train, K=K, lambda_reg=lambda_reg, epsilon=epsilon, max_iter=max_iter)
            best_filename = f"OutDP_DP-LMMF_ep" + str(epsilon) + ".dat"

        elif perturbation_name == 'DP-PRMF':
            R_train, R_test = split_data(R, test_size=0.2)
            R_predicted, sensitivity = alspr.matrix_factorization_with_output_perturbation_als(R_train, K, lambda_reg,
                                                                                               epsilon, max_iter=max_iter)
            best_filename = f"OutDP_DP-PRMF_ep" + str(epsilon) + ".dat"

        # Compute evaluation metrics

        rmse = EV.compute_rmse(R_test, R_predicted)
        recall_at_10 = EV.compute_recall_at_k(R_test, R_predicted, k=topN)
        allMRR = EV.cal_MRR(trueRank, predictedRank, topN)

        # Check best run so far for saving top10 recommendation (Lowest RMSE & Highest Recall@K)
        if (rmse < best_rmse) or (recall_at_10 > best_recall):
            print(f"New Best Run Found! Run {run + 1} with RMSE = {rmse:.4f} and Recall@{topN} = {recall_at_10:.4f}")

            best_rmse = rmse
            best_recall = recall_at_10
            best_run = run + 1
            best_R_predicted = R_predicted.copy()  # Save the best predicted rating matrix


        # Note: MRR calculation for cold-start and normal
        trueRank = DL.GroundTruth_ranking(R_test)
        predictedRank = EV.generate_ranked_recommendations(R_predicted, topN)

        mrr_cold = EV.cal_MRR_group(trueRank, predictedRank, user_group=U_cold, top_k=topN)
        mrr_normal = EV.cal_MRR_group(trueRank, predictedRank, user_group=U_normal, top_k=topN)

        # Note: RMSE calculation for cold-start and normal
        rmse_cold_start = compute_rmse_for_user_group(R_test, R_predicted, U_cold)
        rmse_normal = compute_rmse_for_user_group(R_test, R_predicted, U_normal)

        arps = compute_arp(predictedRank, item_popularity)

        # Store results for averaging
        rmse_list.append(rmse)
        recall_list.append(recall_at_10)
        mrr_list.append(allMRR)

        cold_MRR_list.append(mrr_cold)
        normal_MRR_list.append(mrr_normal)
        cold_rmse_list.append(rmse_cold_start)
        normal_rmse_list.append(rmse_normal)
        arp_list.append(np.mean(arps))

    # Compute average and standard deviation
    def compute_avg_std(metric_list):
        return np.mean(metric_list), np.std(metric_list)

    rmse_avg, rmse_std = compute_avg_std(rmse_list)
    recall_avg, recall_std = compute_avg_std(recall_list)
    mrr_avg, mrr_std = compute_avg_std(mrr_list)
    arp_avg, arp_std = compute_avg_std(arp_list)

    # Print final averaged results
    print("\n===== Final Averaged Results Across 5 Runs =====")
    print(f"Privacy Budget :{epsilon}")
    print(f"RMSE (Avg ± Std): {rmse_avg:.4f} ± {rmse_std:.4f}")
    print(f"Recall@{topN} (Avg ± Std): {recall_avg:.4f} ± {recall_std:.4f}")
    print(f"MRR@{topN} (Avg ± Std): {mrr_avg:.4f} ± {mrr_std:.4f}")
    print(f"ARP (Avg ± Std): {arp_avg:.4f} ± {arp_std:.4f}")
    print(f"RMSE for Cold-Start Users: {np.mean(cold_rmse_list):.4f}")
    print(f"RMSE for Normal Users: {np.mean(normal_rmse_list):.4f}")
    print(f"MRR for Cold-Start Users: {np.mean(cold_MRR_list):.4f}")
    print(f"MRR for Normal Users: {np.mean(normal_MRR_list):.4f}")

    if best_R_predicted is not None:

        DL.save_topN_recommendations(best_R_predicted, best_filename, topN=10)

    print(f"Best run: {best_run} with RMSE = {best_rmse:.4f} and Recall@{topN} = {best_recall:.4f}")
    print(f"Saved best Top-10 recommendations to {best_filename}")


