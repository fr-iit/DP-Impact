import numpy as np
import pandas as pd
import csv

def split_data(R, test_size=0.2):
    """Split rating matrix into train and test sets."""
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

# Function to split the data into train and test sets
def split_data_save(R, ep, name, test_size=0.2):
    train = R.copy()
    test = np.zeros_like(R)

    countR = np.count_nonzero(R)

    for user_idx in range(R.shape[0]):
        non_zero_indices = np.where(R[user_idx] > 0)[0]
        test_indices = np.random.choice(non_zero_indices,
                                         size=int(test_size * len(non_zero_indices)),
                                         replace=False)
        train[user_idx, test_indices] = 0
        test[user_idx, test_indices] = R[user_idx, test_indices]

    if name == 'Bound':
        print('saving: ', name)
        with open('ml-1m/SplitData/'+str(name)+'TrainData'+str(ep)+'.dat', 'w', newline='') as train_file:
            writer = csv.writer(train_file)
            writer.writerows(train)

        with open('ml-1m/SplitData/'+str(name)+'TestData'+str(ep)+'.dat', 'w', newline='') as test_file:
            writer = csv.writer(test_file)
            writer.writerows(test)
    elif name == 'Unbound':
        print('saving: ', name)
        with open('ml-1m/SplitData/'+str(name)+'TrainData'+str(ep)+'.dat', 'w', newline='') as train_file:
            writer = csv.writer(train_file)
            writer.writerows(train)

        with open('ml-1m/SplitData/'+str(name)+'TestData'+str(ep)+'.dat', 'w', newline='') as test_file:
            writer = csv.writer(test_file)
            writer.writerows(test)
    else:
        print('saving: ', name)
        with open('ml-1m/SplitData/TrainData.dat', 'w', newline='') as train_file:
            writer = csv.writer(train_file)
            writer.writerows(train)

        with open('ml-1m/SplitData/TestData.dat', 'w', newline='') as test_file:
            writer = csv.writer(test_file)
            writer.writerows(test)

    # Count non-zero entries in train and test datasets
    train_non_zero = np.count_nonzero(train)
    test_non_zero = np.count_nonzero(test)

    print(f"Data has been split: {len(train)} rows for training and {len(test)} rows for testing.")
    print("Non-zero entries in training dataset:", train_non_zero)
    print("Non-zero entries in testing dataset:", test_non_zero)
    print("Non-zero entries in dataset:", countR)
    # return train, test


def load_TrainTest_matrix(name, ep, max_user):

    if name == 'Bound':
        print('reading: ', name)
        train = np.loadtxt('ml-1m/SplitData/'+str(name)+'TrainData'+str(ep)+'.dat', delimiter=',')
        test = np.loadtxt('ml-1m/SplitData/'+str(name)+'TestData'+str(ep)+'.dat', delimiter=',')
    elif name == 'Unbound':
        print('reading: ', name)
        train = np.loadtxt('ml-1m/SplitData/'+str(name)+'TrainData'+str(ep)+'.dat', delimiter=',')
        test = np.loadtxt('ml-1m/SplitData/'+str(name)+'TestData'+str(ep)+'.dat', delimiter=',')
    else:
        print('reading: ', name)
        train = np.loadtxt('ml-1m/SplitData/TrainData.dat', delimiter=',')
        test = np.loadtxt('ml-1m/SplitData/TestData.dat', delimiter=',')


    # If the file has more rows than expected, slice the matrix.
    if train.shape[0] > max_user:
        train = train[:max_user, :]

    if test.shape[0] > max_user:
        test = test[:max_user, :]

    return train, test

def GroundTruth_ranking(test_matrix):
    test_dict = {}
    for user_id in range(test_matrix.shape[0]):  # Iterate through users
        interacted_items = np.where(test_matrix[user_id] > 0)[0]  # Non-zero ratings
        if len(interacted_items) > 0:
            # test_dict[user_id] = set(interacted_items)
            test_dict[user_id] = {str(item) for item in interacted_items}
    return test_dict


def split_cold_start_users(R_train, min_interactions=5):
    """Segment cold-start and existing users based on the number of interactions in training data."""
    user_interactions = np.count_nonzero(R_train, axis=1)
    cold_start_users = np.where(user_interactions < min_interactions)[0]
    existing_users = np.where(user_interactions >= min_interactions)[0]

    print(f"Cold-Start Users: {len(cold_start_users)}, Existing Users: {len(existing_users)}")
    return cold_start_users, existing_users

def save_topN_recommendations(R_predicted, filename, topN):
    filewrite = 'ml-10m/TopN-RecommendedList/'+filename
    with open(filewrite, mode='w', newline='') as file:
        writer = csv.writer(file)

        for user_idx in range(R_predicted.shape[0]):
            # Get top-10 recommended item indices sorted by highest predicted rating
            topN_items = np.argsort(R_predicted[user_idx])[-topN:][::-1]

            # Write to CSV (user_id, top10_item1, top10_item2, ...)
            writer.writerow([user_idx] + topN_items.tolist())

    print(f"Top-N recommendations saved to {filewrite}")

# dataset: yahoo
def load_user_item_matrix_yahoo():

    movies = set()  # Using set to automatically deduplicate
    users = set()  # Using set to automatically deduplicate
    ratings = []

    with open('ml-yahoo/yahoo_mergerating.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header

        for row in reader:
            userid, movieid, Rating, Genres, gender = row

            user_id = int(userid)
            movie_id = int(movieid)
            rating = float(Rating)

            movies.add(movie_id)
            users.add(user_id)
            ratings.append((user_id, movie_id, rating))

    # Map unique user and movie IDs to their indices
    num_unique_users = len(users)
    num_unique_movies = len(movies)

    print(f'Number of unique users: {num_unique_users}, Number of unique movies: {num_unique_movies}')

    # Create a user-item matrix with dimensions (num_users, num_movies)
    df = np.zeros(shape=(num_unique_users, num_unique_movies))

    # Fill the matrix with ratings
    for user_id, movie_id, rating in ratings:
        df[user_id - 1, movie_id - 1] = rating  # Subtracting 1 to align with 0-indexing

    # Calculate density
    count_non_zero = np.count_nonzero(df)
    density = (count_non_zero / df.size) * 100

    print(f'Yahoo Data Density: {density:.2f}%')

    return df

def load_rating_matrix_Yahoo_BoundPerturb(max_user=2837, max_item=8584):

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-yahoo/InputPerturbation/BoundDP_ep0.01.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df


def load_gender_vector_yahoo():
    gender = []
    m_count = 0
    f_count = 0
    with open('ml-yahoo/update_users.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)

        for row in reader:
            userid,gender_val, uid = row
            if gender_val == 'm':
                gender.append(0)
                m_count +=1
            else:
                gender.append(1)
                f_count +=1
    print('male count: ', m_count, " & female count: ", f_count)

    return np.asarray(gender)

def gender_user_dictionary_yahoo():
    gender_dict = {}
    with open("ml-yahoo/update_users.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            uid, gender, userid = row
            if userid not in gender_dict:
                gender_dict[int(userid)-1] = gender
    #print(gender_dict)
    return gender_dict

# --- end yahoo

# dataset: ml100k
def load_user_item_matrix_100k(max_user=943, max_item=1682):

    df = np.zeros(shape=(max_user, max_item))
    print('original data')
    with open("ml-100k/u.data", 'r') as f: #u.data u1.base
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split()
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            df[user_id-1, movie_id-1] = rating

    return df

def load_rating_matrix_100k_BoundPerturb(max_user=943, max_item=1682):

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-100k/InputPerturbation/BoundDP_ep5.0.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df


def load_gender_vector_100k(max_user=943):

    gender_vec = []
    m = 0
    ff = 0
    # with open("data/ml-100k/userObs.csv", 'r') as f:
    with open("ml-100k/u.user", 'r') as f:
        for line in f.readlines():
            userid, age, gender, occupation, zipcode = line.split("|")
            if gender == "M":
                m += 1
                gender_vec.append(0)
            else:
                ff += 1
                gender_vec.append(1)
    print(f'male: {m}, female: {ff}')
    return np.asarray(gender_vec)

def gender_user_dictionary_100k():
    gender_dict = {}
    with open("ml-100k/u.user", 'r') as f:
        for line in f.readlines():
            userid, age, gender, occupation, zipcode = line.split("|")
            if userid not in gender_dict:
                gender_dict[int(userid)-1] = gender
    return gender_dict

# --- end ml100k
# dataset: ml1m
def load_user_item_matrix_1m(max_user=6040, max_item=3952):

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-1m/ratings.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_rating_matrix_1m_unboundPerturb(max_user=6040, max_item=3952):

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-1m/InputPerturbation/UnboundLaplace.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_rating_matrix_1m_BoundPerturb(max_user=6040, max_item=3952):

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-1m/InputPerturbation/BoundDP_ep0.01.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_gender_vector_1m(max_user=6040):

    gender_vec = []
    m = 0
    ff = 0
    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines()[:max_user]:
            user_id, gender, age, occ, postcode = line.split("::")
            if gender == "M":
                m +=1
                gender_vec.append(0)
            else:
                ff +=1
                gender_vec.append(1)
    print(f'ml-1m; male: {m}, female: {ff}')
    return np.asarray(gender_vec)

def gender_user_dictionary_1m():
    gender_dict = {}
    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines():
            userid, gender, age, occupation, zipcode = line.split("::")
            if userid not in gender_dict:
                gender_dict[int(userid)-1] = gender
    return gender_dict
# --- end ml1m

# --- start ml-10m
def load_user_item_matrix_10m(max_user=72000, max_item=10000):

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-10m/ratings.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating
    print(df.shape)
    return df
# --- end ml-10m

def DensityCount(data = '10m'):

    if data == '1m':
        X = load_rating_matrix_1m_BoundPerturb()
    elif data == '100k':
        X = load_user_item_matrix_100k()
    elif data == 'yahoo':
        X = load_user_item_matrix_yahoo()
        print(f'user: {X.shape[0]}, items: {X.shape[1]}')
    elif data == '10m':
        X = load_user_item_matrix_10m()

    total_entries = X.shape[0] * X.shape[1]
    no_of_ratings = np.count_nonzero(X)
    density = (no_of_ratings/total_entries) * 100

    # obs_total_entries = X_obs.shape[0] * X_obs.shape[1]
    # obs_ratings_no = np.count_nonzero(X_obs)
    # obs_density = (obs_ratings_no/obs_total_entries) * 100

    # print(f"data: {data} , density: {density}, Obs_density: {obs_density}")
    print(f"data: {data} , density: {density}")
    print(f'no_of_ratings: {no_of_ratings}')

def find_max_min_rating_users(rating_matrix):

    # Count the number of non-zero ratings for each user
    user_ratings_count = np.count_nonzero(rating_matrix, axis=1)

    # Find the user with the maximum and minimum ratings
    max_user = np.argmax(user_ratings_count)  # User index with the most ratings
    max_ratings_count = user_ratings_count[max_user]

    min_user = np.argmin(user_ratings_count)  # User index with the least ratings
    min_ratings_count = user_ratings_count[min_user]



    # Find the minimum number of ratings
    min_rating_count = np.min(user_ratings_count)
    max_rating_count = np.max(user_ratings_count)

    # Count how many users have the minimum number of ratings
    # user_count_min = np.sum(user_ratings_count == min_rating_count)
    user_count_min = np.sum(user_ratings_count <= 100)
    user_count_max = np.sum(user_ratings_count > 100)

    print(f"User with maximum ratings: User {max_user}, Number of ratings: {max_ratings_count}")
    print(f"User with minimum ratings: User {min_user}, Number of ratings: {min_ratings_count}")
    print(f'number of user who have min rating: {user_count_min}')
    print(f'number of user who have max rating: {user_count_max}')

    # return max_user, max_ratings_count, min_user, min_ratings_count

def identify_cold_start_users(rating_matrix, cold_start_threshold=0.05):

    # Count the number of non-zero ratings for each user
    user_ratings_count = np.count_nonzero(rating_matrix, axis=1)

    # Find the maximum number of ratings by any user
    max_ratings = np.max(user_ratings_count)

    print(f'max_ratings: {max_ratings}')

    # Define the threshold for cold-start users
    cold_start_limit = cold_start_threshold * max_ratings

    print(f'cold_start_limit: {cold_start_limit}')

    # Identify cold-start users
    cold_start_users = [user for user, count in enumerate(user_ratings_count) if count <= cold_start_limit]

    # Identify normal users
    normal_users = [user for user, count in enumerate(user_ratings_count) if count > cold_start_limit]

    # Count the number of cold-start users
    cold_start_count = len(cold_start_users)

    print(f"Cold-Start Users: {len(cold_start_users)}")
    print(f"Normal Users: {len(normal_users)}")
    # print(f"Count of Cold-Start Users: {cold_start_count}")

    return cold_start_users, normal_users #, cold_start_count

def matrix_to_interaction_data(user_item_matrix):

    user_ids, item_ids = np.where(user_item_matrix > 0)  # Find non-zero interactions
    data = {
        "user_id": user_ids + 1,  # Convert to 1-based indexing
        "item_id": item_ids + 1  # Convert to 1-based indexing
    }
    return pd.DataFrame(data)


def calculate_and_save_item_popularity(interaction_data, output_file):
    interaction_data["item_id"] = interaction_data["item_id"].astype(int)
    # Calculate item popularity (fraction of users interacting with each item)
    total_users = interaction_data["user_id"].nunique()
    item_popularity = (
        interaction_data.groupby("item_id").size() / total_users
    ).reset_index(name="popularity_score")

    # Save to .dat file in the format: itemid|popularity_score
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file, delimiter="|")
        for _, row in item_popularity.iterrows():
            writer.writerow([int(row["item_id"]), f"{row['popularity_score']:.4f}"])

    print(f"Item popularity saved to: {output_file}")

def get_popular_items(R, topN=10):

    item_popularity = np.sum(R > 0, axis=0)  # Count non-zero ratings per item
    top_items = np.argsort(item_popularity)[-topN:][::-1]  # Get top-N items in descending order

    return list(top_items)

def user_profile(R):

    num_users = R.shape[0]
    user_profiles = [np.where(R[u] > 0)[0] for u in range(num_users)]
    return user_profiles


########## call functions

# load_gender_vector_100k()
# load_gender_vector_1m()
# DensityCount()

# find_max_min_rating_users(R)
# identify_cold_start_users(R)

# output_file = "ml-1m/item_popularity.dat"
# R = load_user_item_matrix_1m()
# interaction = matrix_to_interaction_data(R)
# calculate_and_save_item_popularity(interaction, output_file)