import numpy as np
import scipy
import networkx as nx

def replace_missing_with_movie_mean(ratings):
    """
    Replace missing values in the ratings matrix with the mean rating of each movie.

    Parameters:
    - ratings (numpy.ndarray): The ratings matrix with missing values (zeros).

    Returns:
    - numpy.ndarray: The modified ratings matrix with missing values replaced by the mean of each movie.
    - numpy.ndarray: Array containing the mean rating for each movie.
    """
    # Make a copy of the ratings matrix
    ratings_movie_mean = ratings.copy()

    # Initialize an array to hold the mean of each movie
    mean_movie_ratings = np.zeros(ratings_movie_mean.shape[1])

    # Find the mean rating for each movie
    for i in range(ratings_movie_mean.shape[1]):
        mean_movie_ratings[i] = np.mean(ratings_movie_mean[np.where(ratings_movie_mean[:, i] != 0)[0], i])

    # Replace missing values with the mean of the movie
    for i in range(ratings_movie_mean.shape[0]):
        # Find the indices of the missing values
        missing_values = np.where(ratings_movie_mean[i] == 0)[0]

        # Replace the missing values with the average rating of the movie
        ratings_movie_mean[i, missing_values] = mean_movie_ratings[missing_values]

    return ratings_movie_mean, mean_movie_ratings



def replace_missing_with_user_mean(ratings):
    """
    Replace missing values in the ratings matrix with the mean rating of each user.

    Parameters:
    - ratings (numpy.ndarray): The ratings matrix with missing values (zeros).

    Returns:
    - numpy.ndarray: The modified ratings matrix with missing values replaced by the mean of each user.
    - numpy.ndarray: Array containing the mean rating for each user.
    """

    # Make a copy of the ratings matrix
    ratings_user_mean = ratings.copy()

    # Initialize an array to hold the mean of each user
    mean_user_ratings = np.zeros(ratings_user_mean.shape[0])

    # Find the mean rating for each user
    for i in range(ratings_user_mean.shape[0]):
        mean_user_ratings[i] = np.mean(ratings_user_mean[i][np.where(ratings_user_mean[i] != 0)[0]])

    # Replace missing values with the mean of the user
    for i in range(ratings_user_mean.shape[0]):
        # Find the indices of the missing values
        missing_values = np.where(ratings_user_mean[i] == 0)[0]

        # Replace the missing values with the average rating of the user
        ratings_user_mean[i, missing_values] = mean_user_ratings[i]

    return ratings_user_mean, mean_user_ratings



def replace_missing_with_combined_mean(ratings, a, mean_user_ratings, mean_movie_ratings):
    """
    Replace missing values in the ratings matrix with a weighted combination of mean user and mean movie ratings.

    Parameters:
    - ratings (numpy.ndarray): The ratings matrix with missing values (zeros).
    - a (float): The weight parameter for the user mean rating.
    - mean_user_ratings (numpy.ndarray): Array containing the mean rating for each user.
    - mean_movie_ratings (numpy.ndarray): Array containing the mean rating for each movie.

    Returns:
    - numpy.ndarray: The modified ratings matrix with missing values replaced by the weighted combination of mean user and mean movie ratings.
    """

    # Make a copy of the ratings matrix
    ratings_combined_mean = ratings.copy()

    # Replace missing values with the weighted combination of mean user and mean movie ratings
    for i in range(ratings_combined_mean.shape[0]):
        # Find the indices of the missing values
        missing_values = np.where(ratings_combined_mean[i] == 0)[0]

        # Check if missing_values is not empty before proceeding
        if len(missing_values) > 0:
            # Replace the missing values with the weighted combination of mean user and mean movie ratings
            ratings_combined_mean[i, missing_values] = ((a * mean_user_ratings[i]) +
                                                         ((1 - a) * mean_movie_ratings[missing_values]))

    return ratings_combined_mean

# Example usage:
# Assuming you have a 'ratings' matrix and mean_user_ratings, mean_movie_ratings from previous calculations
# You can call the function like this:
# modified_ratings = replace_missing_with_combined_mean(ratings, 0.5, mean_user_ratings, mean_movie_ratings)


# function takes as input 2 users
# calculates the intersection of the movies that both users have rated
# keeps only non-zero indexes
# calculates the spearman correlation coefficient between the 2 users
# returns the spearman correlation coefficient
def similarity_spearman(user1, user2, min_common_elements=3):
    """
    Calculates the Spearman correlation coefficient between two users' movies intersection.


    Parameters:
    - user1 (numpy.ndarray): The ratings of user1.
    - user2 (numpy.ndarray): The ratings of user2.
    - min_common_elements (int): The minimum number of common elements between the two users.

    Returns:
    - None, if the intersection is less than 3 or the number of common elements is less than min_common_elements.
    - The result of the spearman correlation coefficient from scipy, otherwise.

    """



    # find the intersection of the movies that both users have rated
    intersection = np.intersect1d(np.nonzero(user1), np.nonzero(user2))

    # if intersection is less than 3, then we cannot calculate the coefficient
    # require
    if intersection.shape[0] < 3 or intersection.shape[0] <= min_common_elements:
        return None

    # keep only the ratings of the intersection
    user1 = user1[intersection]
    user2 = user2[intersection]


    # calculate the spearman correlation coefficient
    spearman_coeff = scipy.stats.spearmanr(user1, user2)

    # return the coefficient
    return spearman_coeff


def calculate_spearman_metrics(val_users, train_users, nearest_neighbors_indices, min_common_elements=3):
    """
    Calculate Spearman similarity for each validation user compared to its nearest neighbors in the training set.

    Parameters:
    - val_users (list): List of validation users.
    - train_users (list): List of training users.
    - nearest_neighbors_indices (list of lists): Indices of nearest neighbors for each validation user.
    - min_common_elements (int, optional): Minimum number of common elements required to compute similarity. Default is 3.

    Returns:
    - list of lists: List of Spearman similarity results for each validation user.

    Example:
    spearman_results = calculate_spearman_metrics(val_users, train_users, nearest_neighbors_indices, min_common_elements=3)
    """

    spearman_results = [
        [similarity_spearman(val_user, train_users[train_index], min_common_elements=min_common_elements) for train_index in nearest_neighbors_indices[val_user_index]]
        for val_user_index, val_user in enumerate(val_users)
    ]

    return spearman_results


def select_best_neighbors(val_users, train_users, nearest_neighbors_indices, corr_threshold=0.4, p_value_threshold=0.05, min_common_elements=3, full_results=True):
    """
    Select the best neighbors for each validation user based on Spearman similarity.

    Parameters:
    - val_users (list): List of validation users.
    - train_users (list): List of training users.
    - nearest_neighbors_indices (list of lists): Indices of nearest neighbors for each validation user.
    - corr_threshold (float, optional): Threshold for Spearman correlation to consider a neighbor. Default is 0.4.
    - p_value_threshold (float, optional): Threshold for p-value to consider a neighbor. Default is 0.05.
    - min_common_elements (int, optional): Minimum number of common elements required to compute similarity. Default is 3.
    - full_results (bool, optional): If True, return detailed results including correlation statistic and p-value. Default is True.

    Returns:
    - list of dictionaries or list of lists: List of selected neighbors for each validation user. Each dictionary (or list) contains indices of selected neighbors along with correlation statistic and p-value (if full_results is True).

    Example:
    all_best_neighbors = select_best_neighbors(val_users, train_users, nearest_neighbors_indices, corr_threshold=0.4, p_value_threshold=0.05, min_common_elements=3, full_results=True)
    """

    if full_results:

        all_best_neighbors = [
            {
                train_index: (result.statistic, result.pvalue)
                for train_index, result in zip(nearest_neighbors_indices[val_user_index], val_user_similarities)
                if result is not None and result.statistic > corr_threshold and result.pvalue < p_value_threshold
            }
            for val_user_index, val_user_similarities in enumerate(calculate_spearman_metrics(val_users, train_users, nearest_neighbors_indices, min_common_elements=min_common_elements))
            if val_user_similarities
        ]
    else:
         all_best_neighbors = [
            [
                train_index
                for train_index, result in zip(nearest_neighbors_indices[val_user_index], val_user_similarities)
                if result is not None and result.statistic > corr_threshold and result.pvalue < p_value_threshold
            ]
            for val_user_index, val_user_similarities in enumerate(calculate_spearman_metrics(val_users, train_users, nearest_neighbors_indices, min_common_elements=min_common_elements))
            if val_user_similarities
        ]

    return all_best_neighbors


def create_graph(
    _val_users,
    _train_users,
    _val_index,
    _train_index,
    _results,
):
    """
    Create a graph based on the input relationships between validation and similar training users.

    Parameters:
    - _val_users (list): List of validation users.
    - _train_users (list): List of training users.
    - _val_index (list): List of indices corresponding to validation users.
    - _train_index (list): List of indices corresponding to training users.
    - _results (list): List of selected neighbors for each validation user.

    Returns:
    - networkx.Graph: A graph where nodes represent users, and edges represent relationships between validation and similar training users.

    Example:
    G = create_graph(_val_users, _train_users, _val_index, _train_index, _results)
    """

    # create a graph
    G = nx.Graph()

    # for each validation user
    for val_user_index, result in enumerate(_results):
        # for each similar user
        for train_user_index in result:

            # find the true index of the validation user and the similar user
            true_val_index = _val_index[val_user_index]
            true_train_index = _train_index[train_user_index]

            # add an edge between the validation user and the similar user
            G.add_edge(true_val_index, true_train_index)

    return G


# find the count of nodes and number of components in each graph
def find_graphs_info(_graphs, val_index, print_results=False):

    results_graph_info = []

    for i, G in enumerate(_graphs):
        # find the number of nodes
        num_nodes = G.number_of_nodes()

        # find all validation nodes
        val_nodes = [node for node in G.nodes if node in val_index]

        # find count of them
        val_nodes = len(val_nodes)

        # find the number of components
        num_components = nx.number_connected_components(G)

        # find the number of articulation points
        num_articulation_points = len(list(nx.articulation_points(G)))

        val_articulation_points = len([node for node in val_index if node in list(nx.articulation_points(G))])

        results_graph_info.append([num_nodes, val_nodes, num_components, num_articulation_points, val_articulation_points])

        if print_results:
            print(f"Graph {i}: Node Count(Train:{num_nodes - val_nodes}  Val:{val_nodes}), Articul Points(Train:{num_articulation_points - val_articulation_points}  Val:{val_articulation_points}), Components: {num_components},")

    # convert to numpy array
    results = np.array(results_graph_info)

    return results

