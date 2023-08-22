import joblib
import pandas as pd
import os
from surprise import Dataset, Reader,SVDpp
from collections import defaultdict
import numpy as np
from surprise.model_selection import train_test_split
from surprise import accuracy
from utils import Dataloader


model = joblib.load('svdpp_model.joblib')

DIR_PATH = "./data/"
users_df = Dataloader.load_users(DIR_PATH)
ratings_df = Dataloader.load_ratings(DIR_PATH)
movies_df = Dataloader.load_movies(DIR_PATH)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

def recommend_movies(user_id, num_movies=10):
    # 사용자가 평가한 아이템을 제외한 추천 아이템 생성
    user_movies = set([movies_df for (movieid_id, _) in trainset.ur[user_id]])
    all_movies = set([i for i in range(trainset.n_movies)])
    recommended_movies = list(all_movies - user_movies)

    # 추천 아이템을 모델로 예측하여 평점이 높은 순으로 정렬
    predictions = [(movieid_id, model.predict(str(user_id), str(movieid_id)).est) for movieid_id in recommended_movies]
    predictions.sort(key=lambda x: x[1], reverse=True)

    # 상위 n개의 추천 아이템 반환
    top_n = predictions[:num_movies]
    return top_n

# 예시: 사용자 1에 대한 상위 5개 아이템 추천
user_id = 1
recommended_movies = recommend_movies(user_id, num_movies=5)
print(f"Top 5 recommended movies for User {user_id}:")
for movieid_id_id, rating in recommended_movies:
    print(f"movieid_id {movieid_id_id}, Predicted Rating: {rating:.2f}")
