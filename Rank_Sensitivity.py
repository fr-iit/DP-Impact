import csv
from itertools import islice

# Function to calculate Rank-Biased Overlap (RBO)
def calculate_rbo(list1, list2, p=0.9):
    overlap = 0
    rbo_score = 0
    for d in range(1, len(list1) + 1):
        overlap += len(set(list1[:d]) & set(list2[:d]))
        rbo_score += (overlap / d) * (p ** (d - 1))
    return (1 - p) * rbo_score

# Function to calculate Jaccard Similarity
def calculate_jaccard(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

# Function to calculate RLS
def calculate_rls(file1, file2, top_k=10):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        rbo_scores = []
        jaccard_scores = []

        for row1, row2 in zip(reader1, reader2):
            user1, recs1 = row1[0], row1[1:]
            user2, recs2 = row2[0], row2[1:]

            # Ensure we're comparing the same user
            if user1 != user2:
                raise ValueError("User IDs in both files must match.")

            # Limit to top-K recommendations
            recs1_top_k = list(islice(recs1, top_k))
            recs2_top_k = list(islice(recs2, top_k))

            # Calculate RBO and Jaccard scores
            rbo_score = calculate_rbo(recs1_top_k, recs2_top_k)
            # print(rbo_score)
            jaccard_score = calculate_jaccard(recs1_top_k, recs2_top_k)
            # print(jaccard_score)

            rbo_scores.append(rbo_score)
            jaccard_scores.append(jaccard_score)

        avg_rob = sum(rbo_scores)/len(rbo_scores) if rbo_scores else 0
        print(f'avg rbo: {avg_rob}')
        avg_jac = sum(jaccard_scores)/len(jaccard_scores) if jaccard_scores else 0
        print(f'avg jac: {avg_jac}')
        # Average the RBO and Jaccard scores for RLS
        rls_scores = [(rbo + jaccard) / 2 for rbo, jaccard in zip(rbo_scores, jaccard_scores)]
        average_rls = sum(rls_scores) / len(rls_scores) if rls_scores else 0

        return average_rls


if __name__ == "__main__":

    file_without_dp = "ml-yahoo/TopN-RecommendedList/NoDP.dat"  # Format: user_id, item1, item2, ..., itemN
    file_with_dp = "ml-yahoo/TopN-RecommendedList/OutDP_PRMF_ep4.0.dat"        # Format: user_id, item1, item2, ..., itemN

    # Calculate RLS
    rls_score = calculate_rls(file_without_dp, file_with_dp, top_k=10)

    print(f"Rank List Sensitivity (RLS): {rls_score:.4f}")
