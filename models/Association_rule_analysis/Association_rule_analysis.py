import os
import sys

# 현재 스크립트의 디렉토리
script_dir = os.path.dirname(os.path.abspath(__file__))

# ../../utils/Dataloader.py 의 절대 경로 생성
dataloader_path = os.path.abspath(os.path.join(script_dir, '..', '..'))

sys.path.append(dataloader_path)

from utils import Dataloader
import pandas as pd
from joblib import load

#데이터 폴더 경로
DIR_PATH = "./data"

#데이터 호출
ratings_df = Dataloader.load_ratings(DIR_PATH)

def model_loader():
    # 모델 불러오기
    association_df = load('model/Association_rule_analysis/association_df.joblib')
    return association_df

association_df = model_loader()

# 단일 사용자 대상 추천영화 리스트 보여주는 함수
def generate_recommendationsdf(user_id, top_n=10):
    
    movie_matrix = ratings_df.groupby('userId')['movieId'].agg(list)
    movie_matrix = pd.DataFrame(movie_matrix)
    user_watched_movies = movie_matrix.loc[user_id][0]

    recommended_movies_df = pd.DataFrame(columns=['antecedent', 'recommended_movie', 'support', 'confidence', 'lift'])
    antecedent = []
    recommended_movie = []
    support = []
    confidence = []
    lift = []
    i_list = []

    for i in association_df['antecedents'].values:
        if not i in i_list:
            i_list.append(i)
            if i in user_watched_movies:
                filtered_records = association_df[association_df['antecedents'] == i]
                filtered_records = filtered_records.nlargest(5, 'lift')
                for j in range(len(filtered_records)):
                    antecedent.append(filtered_records['antecedents'].iloc[j])
                    recommended_movie.append(filtered_records['consequents'].iloc[j])
                    support.append(filtered_records['support'].iloc[j])
                    confidence.append(filtered_records['confidence'].iloc[j])
                    lift.append(filtered_records['lift'].iloc[j])
        else:
            pass


    recommended_movies_df['antecedent'] = antecedent
    recommended_movies_df['recommended_movie'] = recommended_movie
    recommended_movies_df['support'] = support
    recommended_movies_df['confidence'] = confidence
    recommended_movies_df['lift'] = lift
    recommended_movies_df['lift'] = recommended_movies_df['lift'].astype(float)            
    recommended_movies_df = recommended_movies_df.nlargest(top_n, 'lift')
    recommended_movies_df = recommended_movies_df.reset_index(drop=True)
    recommend_movies = recommended_movies_df['recommended_movie']

    return list(recommend_movies), recommended_movies_df

# 모델 성능 평가
def evaluate(association_df, recommended_movies_df):
    movie_matrix = ratings_df.groupby('userId')['movieId'].agg(list)
    user_watched_movies = pd.DataFrame(movie_matrix)

    # 모든 사용자들의 추천 영화에 대한 lift 값을 저장할 리스트
    all_lift_values = []
    confidence = []
    support = []

    # 모든 사용자들의 추천 영화에 대한 lift 값을 구해서 리스트에 저장
    for i in user_watched_movies.index:
        user_watched = user_watched_movies.loc[i].values.flatten().tolist()[0]
        recommend_movies_df_i = recommended_movies_df(association_df, user_watched, top_n=10)
        all_lift_values.extend(recommend_movies_df_i['lift'])
        confidence.extend(recommend_movies_df_i['confidence'])
        support.extend(recommend_movies_df_i['support'])

    # 모든 사용자들의 추천 영화에 대한 lift 값의 평균 계산
    average_lift = sum(all_lift_values) / len(all_lift_values)
    average_confidence = sum(confidence) / len(confidence)
    average_support = sum(support) / len(support)

    # 평균 lift 값 출력
    print("Average Lift:", round(average_lift,3))
    print("Average confidence:", round(average_confidence,3))
    print("Average support:", round(average_support,3))