import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendation(user_id, top_n, matrix, normalized_matrix, user_similarity):
    """

    :param user_id: user to recommend
    :param top_n: recommendation number
    :param matrix: user-item matrix
    :param user_similarity: user cosine similarity matrix
    :return: recommendation dictionary
    """

    target_user_mean = matrix.loc[int(user_id)].mean()

    user_ratings = matrix.loc[int(user_id)]
    unrated_items = user_ratings[user_ratings.isna()].index

    similarity_scores = user_similarity[int(user_id)]

    predictions = {}

    for item in unrated_items:

        ratings_of_item = normalized_matrix[item].replace(0, np.nan).dropna()
        if ratings_of_item.empty:
            continue

        users = ratings_of_item.index # list of users who rated the movie

        current_similarities = similarity_scores[users]
        positive_mask = current_similarities > 0

        #sum(ratings*similarity) / sum(similarities)
        numerator = np.dot(ratings_of_item[positive_mask], current_similarities[positive_mask])
        denominator = current_similarities[positive_mask].sum()

        if denominator == 0:
            predictions[item] = 0
            continue

        raw_prediction = target_user_mean +  numerator / denominator

        if raw_prediction > 5.0:
            predictions[item] = 5.0
        elif raw_prediction < 0.5:
            predictions[item] = 0.5
        else:
            predictions[item] = raw_prediction

    recommendations = sorted(predictions.items(), key = lambda x: x[1], reverse = True)
    return recommendations[:int(top_n)]


if __name__ == '__main__':

    movies = pd.read_csv('MovieLens_dataset/movies.csv')
    ratings = pd.read_csv('MovieLens_dataset/ratings.csv')

    matrix = ratings.pivot_table(index = 'userId', columns = 'movieId', values = 'rating')
    normalized_matrix = matrix.apply(lambda row: row- row.mean(), axis = 1).fillna(0)

    similarity_array = cosine_similarity(normalized_matrix)
    user_similarity = pd.DataFrame(similarity_array, index = matrix.index, columns = matrix.index)

    while(True):
        choice = input('\n1. Get recommendations\n0. Exit\n')

        if choice == '0':
            break

        elif choice == '1':
            user_id = input('Enter user ID: ')
            top_n = input('How many recommendations do you want? ')

            recommendations = get_recommendation(user_id, top_n, matrix, normalized_matrix, user_similarity)

            print("\n--- TOP {} RECOMMENDATIONS FOR USER {} ---".format(top_n, user_id))

            for movie_id, score in recommendations:
                movie_name = movies[movies.movieId == movie_id].iloc[0, 1]
                print(f"Movie: {movie_name}     Predicted Rating: {score:.2f} ")
