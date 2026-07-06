import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(userId, matrix, item_similarity, top_n):

    """
    :param userId: userId string
    :param matrix: user-item matrix as dataframe
    :param item_similarity: cosine similarity matrix of items as dataframe
    :param top_n: wanted number of recommendations
    :return: recommendation dictionary
    """

    user_ratings = matrix.loc[int(userId)] # get the series of wanted user's ratings

    unrated_items = user_ratings[user_ratings.isna()].index #list of unrated items
    rated_items = user_ratings.dropna().index

    valid_ratings = user_ratings[rated_items]

    predictions = {}

    for item in unrated_items:
        similarity_scores = item_similarity[item] #series of similarity scores of current item
        similarity_scores = similarity_scores[rated_items] #we only need similarities between rated items

        positive_mask = similarity_scores > 0 # positive-masking, ignores zero or negative similarities

        # sum(ratings*similarities) /sum(similarities)
        numerator = np.dot(valid_ratings[positive_mask], similarity_scores[positive_mask])
        denominator = similarity_scores[positive_mask].sum()

        if denominator == 0:
            predictions[item] = 0
            continue

        predictions[item] = numerator / denominator

    recommendations = sorted(predictions.items(), key = lambda x: x[1], reverse = True)
    return recommendations[:int(top_n)]


def calculate_rmse(ratings_df, matrix, item_similarity):
    actuals = []
    predictions = []

    for index, row in ratings_df.iterrows():
        user_id = row['userId']
        item_id = row['movieId']
        actual_rating = row['rating']

        if user_id not in matrix.index or item_id not in item_similarity.columns:
            continue

        user_ratings = matrix.loc[user_id]
        rated_items = user_ratings.dropna().index
        rated_items = rated_items[rated_items != item_id] #removing the target movie fromm the list

        if len(rated_items) == 0:
            continue

        valid_ratings = user_ratings[rated_items]
        similarity_scores = item_similarity[item_id][rated_items]

        positive_mask = similarity_scores > 0

        if positive_mask.sum() == 0:
            continue

        numerator = np.dot(valid_ratings[positive_mask], similarity_scores[positive_mask])
        denominator = similarity_scores[positive_mask].sum()

        if denominator == 0:
            continue

        predicted_rating = numerator / denominator

        actuals.append(actual_rating)
        predictions.append(predicted_rating)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return rmse

if __name__ == '__main__':

    movies = pd.read_csv('MovieLens_dataset/movies.csv')
    ratings = pd.read_csv('MovieLens_dataset/ratings.csv')

    average_allRatings = ratings.rating.mean() #this value will be used for filling NaN values in user-item matrix
    matrix = ratings.pivot_table(index = 'userId', columns = 'movieId', values = 'rating') #user-item matrix

    normalized_matrix = matrix.apply(lambda col: col - col.mean(), axis = 0).fillna(0) #rescaling rating scores

    similarity_array = cosine_similarity(normalized_matrix.T) #transpose of user-item matrix because similarity should be between items

    item_similarity = pd.DataFrame(similarity_array, index = matrix.columns, columns = matrix.columns) #cosine similarity matrix as df

    while(True):

       choice = input("1. Get Recommendations\n2. See past preferences\n0. Exit\n")

       if choice == '0':
           break

       elif choice == '1':

         userId = input("Enter user ID: ")
         top_n = input("How many recommendations do you want? ")

         recommendations = get_recommendations(userId, matrix, item_similarity, top_n)

         print("\n--- TOP {} RECOMMENDATIONS FOR USER {} ---".format(top_n, userId))

         for movie_id, score in recommendations:
            movie_name = movies[movies.movieId == movie_id].iloc[0, 1]
            print(f"Movie: {movie_name}     Predicted Rating: {score:.2f} ")

         rmse_score = calculate_rmse(ratings, matrix, item_similarity)
         print(f"Model RMSE Score: {rmse_score:.4f}\n")

       elif choice == '2':

           user_id = input("Enter user ID: ")
           print(f"\n--- USER {user_id}'S FAVOURITES ---\n")

           fav_ratings = ratings[ratings.userId == int(user_id)]
           fav_ratings = fav_ratings[fav_ratings.rating > 3.5]
           fav_ratings = fav_ratings.sort_values(by = 'rating', ascending = False)

           num = 0
           for movie_id in fav_ratings.movieId:
               movie_name = movies[movies.movieId == movie_id].iloc[0, 1]
               print(f"{num}. {movie_name}")
               num += 1
           print("\n")