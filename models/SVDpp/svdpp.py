import os
import sys

# 현재 스크립트의 디렉토리
script_dir = os.path.dirname(os.path.abspath(__file__))

# ../../utils/Dataloader.py 의 절대 경로 생성
dataloader_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(dataloader_path)

from joblib import dump, load
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import train_test_split
from surprise import accuracy
from utils import Dataloader

path = './data'
movies_df = Dataloader.load_movies(path)
ratings_df = Dataloader.load_ratings(path)
users_df = Dataloader.load_users(path)

def svdpp_model_loader():

    # SVDpp모델 불러오기
    svdpp_predictions_df = load('models/SVDpp/hybrid_models/svdpp_model.joblib')

    return svdpp_predictions_df

# Weighted 추천 생성
def svdpp_recommendation(user_id):
    
    # 모델 loader
    svdpp_predictions_df = svdpp_model_loader()

    svdpp_user_preds = svdpp_predictions_df[svdpp_predictions_df['uid'] == user_id]

    # 사용자가 이미 본 영화 제외
    user_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    svdpp_user_preds = svdpp_user_preds[~svdpp_user_preds.index.isin(user_movies)]

    # 가장 높은  예측 값을 가진 영화 순으로 정렬하여 추천
    svdpp_recommendations = svdpp_user_preds['est'].sort_values(ascending=False).index.tolist()

    return svdpp_recommendations

# 간단한 추천 함수
def recommend_user_to_movie(userId):

    # 특정 사용자에게 추천
    user_id_to_recommend = userId  # 원하는 사용자 ID 입력
    recommendations = svdpp_recommendation(user_id_to_recommend)

    # 영화 ID와 영화 제목을 매핑하는 딕셔너리 생성
    movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

    # 추천된 영화 ID들을 실제 영화 데이터에서 영화 제목으로 변환
    recommended_movie_titles = [movie_id_to_title[ratings_df.loc[movie_idx]['movieId']] for movie_idx in recommendations]

    print("Recommended movies for User {}: ".format(user_id_to_recommend))
    for movie_title in recommended_movie_titles[:5]:
        print(movie_title)

if __name__ == '__main__':
    user_id = int(input('유저 아이디를 입력해주세요: '))
    recommend_user_to_movie(user_id)