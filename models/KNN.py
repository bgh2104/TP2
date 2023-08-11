import joblib
from utils.Dataloader import load_ratings,load_movies
import sys
from pathlib import Path
import os
import argparse

#콘텐츠 기반 필터링용 패키지
from gensim.models import Word2Vec
from utils.Preprocessing import tokenizer, vectorizer
import gensim.downloader as api
from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib


# 하이퍼 파라미터
# vector_size: Word2Vec 모델에서 생성할 워드 임베딩 벡터의 차원을 나타내는 정수 값입니다. 각 단어를 vector_size 차원의 벡터로 표현합니다. 기본값은 100입니다.
# pretrained: 사전 훈련된 워드 임베딩 모델의 이름을 나타내는 문자열입니다. gensim.downloader를 사용하여 사전 훈련된 워드 임베딩 모델을 로드할 때 사용됩니다.
            # 예를 들어, 'glove-twitter-100'은 Twitter 데이터로 사전 훈련된 100차원의 GloVe 워드 임베딩을 의미합니다.
# n_neighbors: K-Nearest Neighbors 모델에서 각 샘플의 이웃의 수를 나타내는 정수 값입니다. predict 메서드에서 영화 추천 시 이웃의 수로 사용됩니다. 기본값은 5입니다.

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', type=int, default=1)
    parser.add_argument('-n', '--num', type=int, default=5)

    opt = parser.parse_args()
    return opt

class KNN():
    def __init__(self, path):
        self.model = joblib.load(path)
        
    def predict(self, userid, n):
        """
        콘텐츠 정보를 기반으로 영화를 추천합니다.
        
        Args:
            userid (int) : 추천 대상 유저의 id.
            n (int) : 출력하는 추천 영화의 수. n을 입력하면 유저가 지금까지 평점을 남긴 데이터를 기반으로 비슷한 상위 n개의 영화를 추천합니다.
        """
        #data load
        cbf_data = joblib.load('models/cbf_data.joblib')
        ratings_df = load_ratings('data/')
        
        #유저가 시청했던 영화 목록 호출
        movie_list = ratings_df[ratings_df['userId']==userid]['movieId'].tolist()
        
        #입력 벡터 생성
        #입력 벡터는 유저가 본 영화의 모든 벡터의 평균을 사용
        m_vector = 0
        for m in movie_list:
            m_vector += cbf_data[m]
        
        #예측    
        return self.model.kneighbors(m_vector.reshape((1,-1)), n_neighbors=n)[1][0]

def train(movies_df=None, vector_size=100, pretrained = 'glove-twitter-100'):
    if movies_df is None:
        movies_df = load_movies('data/')
    
    print("---Tokenizing...---")
    tokens = movies_df['title'].apply(tokenizer)
    print("Tokenizing Complete.")

    print("---w2v Training...---")
    w2v = Word2Vec(sentences=tokens, vector_size = vector_size, window = 2, min_count = 1, workers = 4, sg= 0)
    w2v.save("./models/word2vec.model")
    print(w2v.wv.vectors.shape)
    print("w2v Training Complete.")

    wv = w2v.wv

    vectors = tokens.apply(vectorizer)

    print("---pre-trained w2v loading...---")
    #사전 훈련된 w2v 가중치 호출
    wv2 = api.load(f"{pretrained}")
    print("loading Complete.")


    # 장르를 전처리 안해도 되게 되어 있음ㅋㅋ
    # vector = 0: 벡터 초기화. 이 벡터는 장르 벡터를 구성하기 위해 각 장르의 벡터를 누적할 것입니다.
    # for g in sentence.split('|'):: 주어진 장르 문자열을 파이프(|)를 기준으로 분할하여 각 장르에 대해 다음 과정을 반복합니다.
    # if g.lower() == "children's": g = "children": 장르명에 'children's'라는 특정 표현이 있는 경우, 이를 'children'으로 변경합니다. 장르명에 대소문자 구분이 없도록 소문자로 변경한 후 비교합니다.
    # elif g.lower() == "film-noir": g = "noir": 장르명에 'film-noir'라는 특정 표현이 있는 경우, 이를 'noir'로 변경합니다. 마찬가지로 대소문자 구분 없이 비교 후 변경합니다.
    # vector += wv2[g.lower()]: 장르명을 해당 장르의 워드 임베딩 벡터로 변환하고, 기존의 vector에 더해줍니다. 이렇게 하면 모든 장르 벡터가 누적됩니다.
    # return vector: 최종적으로 누적된 장르 벡터를 반환합니다. 이 벡터는 주어진 장르 정보를 워드 임베딩 벡터로 변환한 결과입니다.

    def gen2vec(sentence):
        vector = 0
        for g in sentence.split('|'):
            if g.lower() == "children's":
                g = "children"
            elif g.lower() == "film-noir":
                g = "noir"

            vector += wv2[g.lower()]
        return vector

    g_vector = movies_df['genres'].apply(gen2vec)
    

    #훈련 데이터 생성

    # cbf_vectors = ((vectors.to_numpy() + g_vector.to_numpy()) / 2).tolist(): 각 영화의 제목 정보와 장르 정보를 합한 벡터를 생성합니다. 
        # 이를 위해 제목 벡터와 장르 벡터를 더한 후, 두 벡터의 평균을 구한 뒤 리스트로 변환합니다. 이것이 콘텐츠 기반 필터링의 입력 데이터로 사용됩니다.
    # cbf_data = np.zeros((movies_df['movieId'].max()+1, 100)): 영화의 수와 워드 임베딩 벡터의 차원에 맞게 초기화된 2D NumPy 배열을 생성합니다. 
        # 각 행은 영화를 나타내며, 열은 워드 임베딩 벡터의 차원을 나타냅니다.
    # for idx, vec in zip(movies_df['movieId'], cbf_vectors): 각 영화의 인덱스와 해당 영화에 대한 합성된 벡터를 반복하여 처리합니다.
    # cbf_data[idx] = vec: 영화 인덱스 idx에 대응하는 행에 합성된 벡터 vec를 할당하여 영화의 특성을 저장합니다.

    cbf_vectors = ((vectors.to_numpy() + g_vector.to_numpy()) / 2).tolist()
    cbf_data = np.zeros((movies_df['movieId'].max()+1, 100)) # 데이터셋 내의 영화 ID가 연속적이지 않거나 중간에 비어있는 경우를 고려하여 최대 ID에 1을 더한 값으로 행 개수를 설정, 벡터 사이즈가 100으로 설정되어 있어서 100임
    
    for idx, vec in zip(movies_df['movieId'], cbf_vectors):
        cbf_data[idx] = vec
    

    # joblib.dump(cbf_data, "./models/cbf_data.joblib"): 생성된 콘텐츠 기반 필터링용 데이터를 지정된 경로에 저장합니다.
    # knn = NearestNeighbors(): K-Nearest Neighbors 모델을 초기화합니다.
    # knn.fit(cbf_data): 콘텐츠 기반 필터링 데이터를 사용하여 K-Nearest Neighbors 모델을 훈련합니다.
    # joblib.dump(knn, "./models/knn.joblib"): 훈련된 K-Nearest Neighbors 모델을 지정된 경로에 저장합니다.

    joblib.dump(cbf_data, "./models/cbf_data.joblib")

    print("---knn Training...---")
    knn = NearestNeighbors()
    knn.fit(cbf_data)
    joblib.dump(knn, "./models/knn.joblib")
    print("---knn Done.---")
    



if __name__ == '__main__':
    opt = parse_opt()
    knn = KNN()
    print(knn.predict(opt.user, opt.num))