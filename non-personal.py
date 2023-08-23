from utils import Dataloader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt # 3.7.2
import random
from mlxtend.frequent_patterns import apriori, association_rules #0.22.0
import matplotlib.colors as mcl
from matplotlib.colors import LinearSegmentedColormap
from mlxtend.preprocessing import TransactionEncoder
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re


path = './data/'
ratings_df = Dataloader.load_ratings(path)


def rating_based(ratings_df, k=5):
    # 평점 기준으로 sorting
    rating_count = ratings_df.groupby('movieId').count()[['rating']].reset_index()
    rating_count.columns = ['movieId', 'rating_count']
    # 평점이 30개 미만인 영화들 삭제
    rating_count = rating_count[rating_count['rating_count'] >= 30]
    # rating_count 데이터프레임에 각 영화별 평점 평균 컬럼 삽입
    rating_count2 = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    rating_count['mean_rating'] = rating_count2['rating']
    # 평점 기반 sorting
    rating_count = rating_count.sort_values(by='mean_rating', ascending=False)
    return rating_count['movieId'][:k]

def rating_random_mixed(ratings_df, k=5):
    # 평점 기준으로 sorting
    rating_count = ratings_df.groupby('movieId').count()[['rating']].reset_index()
    rating_count.columns = ['movieId', 'rating_count']
    # 평점이 30개 미만인 영화들 삭제
    rating_count = rating_count[rating_count['rating_count'] >= 30]
    # rating_count 데이터프레임에 각 영화별 평점 평균 컬럼 삽입
    rating_count2 = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    rating_count['mean_rating'] = rating_count2['rating']
    # 평점 기반 sorting
    rating_count = rating_count.sort_values(by='mean_rating', ascending=False)

    top = round(k * 0.8)
    random_num = k - top

    # 평점
    top_rating_count = rating_count['movieId'][:top]
    top_rating_movies = list(top_rating_count.values)

    # 무작위
    # 중복을 막기 위해, 평점 기반 모델로 나온 출력값 제외하고 랜덤하게 영화 추천
    dropped_movies = rating_count[~rating_count['movieId'].isin(top_rating_movies)]
    random_movies = random.sample(dropped_movies['movieId'].tolist(), random_num)
    return top_rating_movies + random_movies

def random(ratings_df, k=5):
    # 평점 기준으로 sorting
    rating_count = ratings_df.groupby('movieId').count()[['rating']].reset_index()
    rating_count.columns = ['movieId', 'rating_count']
    # 평점이 30개 미만인 영화들 삭제
    rating_count = rating_count[rating_count['rating_count'] >= 30]
    # rating_count 데이터프레임에 각 영화별 평점 평균 컬럼 삽입
    rating_count2 = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    rating_count['mean_rating'] = rating_count2['rating']
    # 평점 기반 sorting
    rating_count = rating_count.sort_values(by='mean_rating', ascending=False)
    return random.sample(rating_count['movieId'].tolist(), k)
