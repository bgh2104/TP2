from utils import Dataloader
import pandas as pd
from joblib import dump, load

#데이터 폴더 경로
DIR_PATH = "./data/"

#데이터 호출
ratings_df = Dataloader.load_ratings(DIR_PATH)

def model_loader():
    # 모델 불러오기
    association_df = load('association_df.joblib')
    return association_df

association_df = model_loader()

# 단일 사용자 대상 추천영화 리스트 보여주는 함수
def generate_recommendations(user_id, top_n=10):
    
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

    return list(recommend_movies)
