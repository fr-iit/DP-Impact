import csv
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error

def compute_rmse(R_test, R_predicted):
    # Only consider non-zero ratings in the test set
    non_zero_indices = R_test > 0
    mse = mean_squared_error(R_test[non_zero_indices], R_predicted[non_zero_indices])
    return np.sqrt(mse)

def compute_recall_at_k(R_test, R_predicted, k):
    recall_values = []
    num_users_with_recall = 0
    for user_idx in range(R_test.shape[0]):
        actual_items = np.where(R_test[user_idx] > 0)[0]
        if len(actual_items) == 0:
            continue  # Skip users with no test data

        predicted_items = np.argsort(R_predicted[user_idx])[-k:][::-1]
        hits = len(set(predicted_items).intersection(actual_items))
        recall = hits / len(actual_items)

        if recall > 0:
            num_users_with_recall += 1

        recall_values.append(recall)

    print(f"Users with non-zero recall: {num_users_with_recall}/{R_test.shape[0]}")
    print(f"test user shape: {R_test.shape[0]}")
    return np.mean(recall_values)

def read_recommendation_file(filepath):
    recommendations = defaultdict(list)
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            user_id = row[0]
            items = row[1:]
            recommendations[user_id] = items
    return recommendations

def write_recommendation_file(filepath, data):

    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        for user_id, items in data.items():
            writer.writerow([user_id] + items)

# Compute Jaccard distance to measure changes in recommendation lists
def jaccard_distance(x, y):
    set_x = set(x)
    set_y = set(y)
    return 1 - len(set_x.intersection(set_y)) / len(set_x.union(set_y))

def generate_ranked_recommendations(R_pred, top_k=10):
    predicted_rankings = {}
    for user_id in range(R_pred.shape[0]):
        sorted_items = np.argsort(R_pred[user_id])[::-1]  # Sort items by score (descending)
        top_items = [str(item) for item in sorted_items[:top_k]]  # Convert to string
        predicted_rankings[user_id] = top_items

    return predicted_rankings

def cal_MRR(test_data, predicted_rankings, top_k=10):
    reciprocal_ranks = []
    user_MRR_scores = {}

    for user, true_items in test_data.items():
        if user not in predicted_rankings or not predicted_rankings[user]:  # Check if user has predictions
            user_MRR_scores[user] = 0.0
            continue

        ranked_items = predicted_rankings[user][:top_k]

        found = False  # Flag to track if a correct prediction is found
        for rank, item in enumerate(ranked_items, start=1):
            if item in true_items:
                user_MRR_scores[user] = 1 / rank
                reciprocal_ranks.append(1 / rank)
                found = True
                break  

        if not found:  
            user_MRR_scores[user] = 0.0

    mean_MRR = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return mean_MRR

def cal_MRR_group(test_data, predicted_rankings, user_group, top_k=10):

    reciprocal_ranks = []

    if user_group is None:
        user_group = test_data.keys()

    user_MRR_scores = {}

    # Only iterate over users in user_group
    for user in user_group:
        # If user not in test_data, skip
        if user not in test_data:
            user_MRR_scores[user] = 0.0
            continue

        true_items = test_data[user]
        
        if user not in predicted_rankings or not predicted_rankings[user]:
            user_MRR_scores[user] = 0.0
            continue

        ranked_items = predicted_rankings[user][:top_k]

        found = False
        for rank, item in enumerate(ranked_items, start=1):
            if item in true_items:
                user_MRR_scores[user] = 1.0 / rank
                reciprocal_ranks.append(1.0 / rank)
                found = True
                break

        if not found:
            user_MRR_scores[user] = 0.0

    mean_MRR = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    return mean_MRR  
