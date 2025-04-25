
import numpy as np
import pandas as pd
import csv
import DataLoader as DL
import Evaluation as EV


def load_recommendations(filename):
    
    recommendations = {}

    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(",")  # Split by comma
            if len(parts) < 2:
                continue  # Skip empty lines

            user_id = int(parts[0])  # First column is user_id
            items = list(map(int, parts[1:]))  
            recommendations[user_id] = items

    return recommendations


def get_ground_truth_from_matrix(R, rating_threshold=4.0):
    ground_truth = {}
    num_users, num_items = R.shape

    for user in range(num_users):
        relevant_items = set(np.where(R[user] >= rating_threshold)[0])  # Find items where rating ≥ threshold
        if relevant_items:
            ground_truth[user] = relevant_items

    return ground_truth


dataset = 'ml-1m'

if dataset == 'ml-1m':
    R = DL.load_user_item_matrix_1m()
elif dataset == 'ml-100k':
    R = DL.load_user_item_matrix_100k()
elif dataset == 'ml-yahoo':
    R = DL.load_user_item_matrix_yahoo()

U_cold, U_normal = DL.identify_cold_start_users(R)
truth_file = dataset+'/test_data.dat'
ground_truth = load_recommendations(truth_file) #get_ground_truth_from_matrix(R)
filename = dataset+"/TopN-RecommendedList/OutDP_LMMF_ep4.0.dat"
recommendations = load_recommendations(filename)
mrr_cold = EV.cal_MRR_group(ground_truth, recommendations,top_k=10, user_group=U_cold)
mrr_normal = EV.cal_MRR_group(ground_truth, recommendations,top_k=10, user_group=U_normal)
print(f"MRR (Cold-Start): {mrr_cold:.4f}")
print(f"MRR (Normal):     {mrr_normal:.4f}")
print(f"Gap cold & normal (recall): {abs(mrr_cold - mrr_normal):.4f} ")

