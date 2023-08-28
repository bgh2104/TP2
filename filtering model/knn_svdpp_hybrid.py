from joblib import dump, load
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVDpppp, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
from utils import Dataloader

path = 'data'
movies_df = Dataloader.load_movies(path)
ratings_df = Dataloader.load_ratings(path)
users_df = Dataloader.load_users(path)

def hybrid_model_loader():

    # SVDpp모델 불러오기
    svdpp_predictions_df = load('hybrid_models/svdpp_model.joblib')

    # KNN모델 불러오기
    knn_predictions_df = load('hybrid_models/knn_model.joblib')
    
    return svdpp_predictions_df, knn_predictions_df

# Weighted Hybrid 추천 생성
def hybrid_recommendation(user_id):
    
    # 모델 loader
    svdpp_predictions_df, knn_predictions_df = hybrid_model_loader()

    svdpp_user_preds = svdpp_predictions_df[svdpp_predictions_df['uid'] == user_id]
    knn_user_preds = knn_predictions_df[knn_predictions_df['uid'] == user_id]

    # 두 알고리즘의 예측 값을 가중 평균하여 추천 리스트 생성
    weight_svdpp = 0.7  # SVDpp 모델의 가중치
    weight_knn = 0.3  # KNN 모델의 가중치

    hybrid_preds = (weight_svdpp * svdpp_user_preds['est']) + (weight_knn * knn_user_preds['est'])

    # 사용자가 이미 본 영화 제외
    user_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    hybrid_preds = hybrid_preds[~hybrid_preds.index.isin(user_movies)]

    # 가장 높은 가중 평균 예측 값을 가진 영화 순으로 정렬하여 추천
    hybrid_recommendations = hybrid_preds.sort_values(ascending=False).index.tolist()

    return hybrid_recommendations

# 간단한 추천 함수
def recommend_user_to_movie(userId):

    # 특정 사용자에게 추천
    user_id_to_recommend = userId  # 원하는 사용자 ID 입력
    recommendations = hybrid_recommendation(user_id_to_recommend)

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