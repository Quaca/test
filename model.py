import time

import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import seaborn as sns
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
	

movies = pd.read_csv("C:/Users/User/Downloads/ml-latest-small/ml-latest-small/movies.csv",
    usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})

ratings = pd.read_csv("C:/Users/User/Downloads/ml-latest-small/ml-latest-small/ratings.csv",
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

num_users = len(ratings.userId.unique())
num_items = len(ratings.movieId.unique())
print('There are {} unique users and {} unique movies in this data set'.format(num_users, num_items))

# get count
ratings_cnt_tmp = pd.DataFrame(ratings.groupby('rating').size(), columns=['count'])
ratings_cnt_tmp

movies_count = pd.DataFrame(ratings.groupby('movieId').size(), columns=['count'])
movies_count['count'].quantile(np.arange(1, 0.6, -0.05))

popularity_thres = 50
popular_movies = list(set(movies_count.query('count >= @popularity_thres').index))
ratings_drop_movies = ratings[ratings.movieId.isin(popular_movies)]
print('shape of original ratings data: ', ratings.shape)
print('shape of ratings data after dropping unpopular movies: ', ratings_drop_movies.shape)

users_count = pd.DataFrame(ratings_drop_movies.groupby('userId').size(), columns=['count'])
users_count['count'].quantile(np.arange(1, 0.5, -0.05))

ratings_thres = 50
active_users = list(set(users_count.query('count >= @ratings_thres').index))
ratings_drop_users = ratings_drop_movies[ratings_drop_movies.userId.isin(active_users)]

movie_user_matrix = ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)

movie_to_idx = {
    movie: i for i, movie in 
    enumerate(list(movies.set_index('movieId').loc[movie_user_matrix.index].title))
}
# transform matrix to scipy sparse matrix
movie_user_matrix_sparse = csr_matrix(movie_user_matrix.values)

# define model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# fit
model_knn.fit(movie_user_matrix_sparse)


def fuzzy_matching(mapper, fav_movie, verbose=True):
    match_tuple = []
#     get match
    for title, idx in mapper.items():
        ratio = similar(title.lower(), fav_movie.lower()) * 100
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
#     for title, idx in mapper.items():
#         match_tuple.append((title, idx))
    # sort
    print(match_tuple)
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]
	
	
def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations): 
    # fit
    model_knn.fit(data)
    # get input movie index
    print('You have input movie:', fav_movie)
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)
    # inference
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    # get list of raw idx of recommendations
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(fav_movie))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))

my_favorite = 'Iron Man'

make_recommendation(
    model_knn=model_knn,
    data=movie_user_matrix_sparse,
    fav_movie=my_favorite,
    mapper=movie_to_idx,
    n_recommendations=10)