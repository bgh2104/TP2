import spacy
import re
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import pandas as pd

nlp = spacy.load('en_core_web_sm')
wv = Word2Vec.load("./models/word2vec.model").wv


# nlp(sentence)를 호출하여 SpaCy 언어 처리 모델을 사용하여 문장을 토큰화하고 파싱합니다.
# doc[:-3]를 통해 마지막 세 개의 토큰을 제외한 나머지 토큰을 선택합니다. 이렇게 하는 이유는 종종 문장의 마지막에 있는 토큰들은 문장 부호 등과 관련이 있어서 토큰화 결과에 영향을 줄 수 있기 때문입니다.
# token.lemma_를 사용하여 각 토큰의 원형(lemma)을 추출합니다. 원형은 단어의 기본 형태를 나타냅니다. 예를 들어, "running"의 원형은 "run", "better"의 원형은 "good" 등입니다.
# 최종적으로 토큰의 원형으로 이루어진 리스트인 word_list를 반환합니다.


def tokenizer(sentence):
    doc = nlp(sentence)
    word_list = [token.lemma_ for token in doc[:-3]]
    
    return word_list



# vector = 0: 벡터 초기화. 이 벡터는 문장 벡터를 구성하기 위해 토큰들의 벡터를 누적할 것입니다.
# for token in token_list:: 주어진 토큰 리스트의 각 토큰에 대해서 다음 과정을 반복합니다.
# vector += wv[token]: 각 토큰을 해당 토큰의 워드 임베딩 벡터로 변환하고, 기존의 vector에 더해줍니다. 이렇게 하면 문장 내의 모든 토큰 벡터가 더해집니다.
# return vector / len(token_list): 토큰 벡터의 누적합을 토큰의 총 개수로 나눠서 평균 벡터를 계산합니다. 이 평균 벡터가 문장 벡터가 됩니다.


def vectorizer(token_list):
    vector=0
    for token in token_list:
        vector += wv[token]
    return vector / len(token_list)



# movie_df 전처리
def preprocessing_movie(movies_df):
    # 장르 분할
    movies_df['genres'] = movies_df['genres'].str.split('|')
    edited_movies_df = movies_df.explode('genres')
    # 장르를 label encoding
    genres = edited_movies_df['genres']
    encoder = LabelEncoder()
    encoded_genres = encoder.fit_transform(genres)
    encoded_genres = pd.DataFrame(encoded_genres, columns=['genres'])
    edited_movies_df['genres'] = encoded_genres

    # 'year' 열에 연도 정보 추출하여 추가
    edited_movies_df['year'] = edited_movies_df['title'].str.extract(r'\((\d{4})\)')
    # 연도 정보가 포함된 제목 수정
    edited_movies_df['title'] = edited_movies_df['title'].str.replace(pat=r'[^\w\s]', repl=r'', regex=True)
    edited_movies_df['title'] = edited_movies_df['title'].str.replace(pat=r'[0-9]', repl=r'', regex=True)
    # 연대 컬럼 삽입
    edited_movies_df["year_term"] = edited_movies_df["year"].apply(lambda x : x[-4 :-1] + "0")
    # 데이터 타입 정수형으로 변경
    edited_movies_df[['year', 'year_term']] = edited_movies_df[['year', 'year_term']].astype(int)

    return edited_movies_df


# users_df 전처리
def preprocessing_users(users_df):
    # 성별을 0, 1 변환
    gender = users_df['gender']
    encoder = LabelEncoder()
    encoded_gender = encoder.fit_transform(gender)
    encoded_gender = pd.DataFrame(encoded_gender, columns=['gender'])
    users_df['gender'] = encoded_gender
    # State 컬럼 추가
    # zip_code의 첫번째 자릿수 혹은 하이픈(-) 전의 숫자가 State를 의미
    users_df['state'] = users_df['zip_code'].apply(lambda x: x.split('-')[0] if '-' in x else x[0])
    # 데이터 타입 정수형으로 변경
    users_df['state'] = users_df['state'].astype(int)

    return users_df


# ratings_df 전처리할 것 없음