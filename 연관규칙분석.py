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
from joblib import dump, load

#데이터 폴더 경로
DIR_PATH = "./data/"

#데이터 호출
ratings_df = Dataloader.load_ratings(DIR_PATH)

def hybrid_model_loader():

    # 모델 불러오기
    association_rules_df = load('association_rules_df.joblib')
    
    return association_rules_df


def recommend_movies(association_rules_df, user_watched_movies, min_lift=1):
    recommended_movies = []
    top_n = round(len(user_watched_movies)*0.2)

    for _, row in association_rules_df.iterrows():
        antecedent = row['antecedents']
        consequent = row['consequents']
        lift = row['lift']

        if antecedent.issubset(user_watched_movies) and lift >= min_lift:
            recommended_movie = list(consequent.difference(user_watched_movies))
            recommended_movies.extend(recommended_movie)

    recommended_movies = list(set(recommended_movies))
    
    # 상위 N개의 영화 추천
    recommended_movies = sorted(recommended_movies, key=lambda x: recommended_movies.count(x), reverse=True)[:top_n]
    
    return recommended_movies



