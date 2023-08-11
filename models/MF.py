from tensorflow.keras.models import load_model
import tensorflow as tf
from utils.Dataloader import load_ratings

#협업 필터링용 패키지
from utils import Dataloader
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Flatten
from sklearn.model_selection import train_test_split

# 하이퍼 파라미터
# K: 잠재 요인(latent factor)의 차원 수입니다. 사용자와 아이템 벡터의 크기를 결정하는 파라미터입니다. 모델이 학습하는 잠재적인 특성의 수를 나타냅니다.
# epochs: 모델의 훈련 에포크 수입니다. 에포크는 전체 데이터셋에 대한 한 번의 순전파와 역전파를 의미합니다. 모델이 훈련 데이터를 반복해서 학습하는 횟수를 결정합니다.
# batch_size: 미니 배치 학습에서 사용되는 배치의 크기입니다. 한 번의 업데이트마다 처리되는 데이터 포인트 수를 의미하며, 너무 작으면 학습 속도가 느려질 수 있고, 너무 크면 메모리 부하가 발생할 수 있습니다.
# validation_split: 훈련 데이터 중에서 검증 데이터로 사용할 비율입니다. 훈련 데이터의 일부를 검증 데이터로 분리하여 모델의 성능을 평가하고, 과적합을 방지하는데 활용됩니다.

class MF():
    """
    Matrix Factorization를 활용한 협업 필터링 기반 추천 모델입니다.
    """
    def __init__(self, path):
        self.model = load_model(path)
        
    def predict(self, userid, top = 10):
        """
        유저 정보를 기반으로 영화를 추천합니다.
        
        Args:
            userid (int) : 추천 대상 유저의 id.
            top (int) : 출력하는 추천 영화의 수. n을 입력하면 유저가 만족할 것으로 생각되는 상위 n개의 영화를 추천합니다.
        """
        user = self.model.get_layer('u_emb')(userid)[tf.newaxis,:]
        items = tf.transpose(self.model.get_layer('i_emb').weights[0], [1,0])
        mm = tf.matmul(user,items)
        
        return tf.argsort(mm, direction='DESCENDING').numpy().tolist()[0][:top]
    
#모델 기반 협업 필터링(Matrix Factorization)
#모델 파이프라인 생성

#user와 item 입력 레이어를 정의합니다. 이 레이어는 각각 사용자 ID와 아이템 ID를 입력으로 받습니다.
# Embedding 레이어를 사용하여 사용자와 아이템을 잠재 요인 벡터로 변환합니다. user_dim과 item_dim은 각각 사용자와 아이템의 수 또는 범주 개수입니다. 
    # K는 잠재 요인 벡터의 크기를 결정하는 파라미터입니다. 이 레이어는 사용자와 아이템을 저차원 벡터 공간으로 매핑하여 특성을 학습합니다.
# dot 레이어를 사용하여 사용자와 아이템 벡터 간의 내적을 계산합니다. 이것은 평점 예측을 위한 점수를 생성하는데 사용됩니다.
    # axes=2는 사용자와 아이템 벡터의 내적을 계산할 때, 두 번째 축(axis)을 기준으로 내적을 수행하라는 의미입니다.
# Flatten 레이어를 사용하여 내적 결과를 1차원 벡터로 평탄화합니다. 이것은 최종적인 예측 값을 만드는데 사용됩니다.
# Model 클래스를 사용하여 입력과 출력을 정의하여 MF 모델을 생성합니다.

def mf_model(user_dim, item_dim, K):
    user = Input((1,))
    item = Input((1,))
    u_emb = Embedding(user_dim, K, name='u_emb')(user)
    i_emb = Embedding(item_dim, K, name='i_emb')(item)

    R = tf.keras.layers.dot([u_emb, i_emb], axes=2)
    R = Flatten()(R)
    return Model(inputs=[user, item], outputs=R)

def train(users_df=None, movies_df=None, ratings_df=None, K=200, epochs=1, batch_size = 512, validation_split=0.2):
    
    if users_df is None:
        users_df = Dataloader.load_users('data')
    if movies_df is None:
        movies_df = Dataloader.load_movies('data')
    if ratings_df is None:
        ratings_df = Dataloader.load_ratings('data')
        
    USER_DIM = users_df['userId'].max()+1
    ITEM_DIM = movies_df['movieId'].max()+1

    #data split
    train, val = train_test_split(ratings_df, test_size=0.2) #randomstate???
    x_train = [train['userId'], train['movieId']]
    y_train = train['rating']

    #모델 선언
    mf = mf_model(USER_DIM, ITEM_DIM, K)
    mf.compile(loss="mse",
               optimizer="adam"
              )

    #모델 훈련
    mf.fit(x_train, y_train, 
           epochs=epochs,
           batch_size = batch_size,
           validation_split=validation_split)

    #모델 저장
    print("---Saving model---")
    mf.save('./models/mf.h5')
    print('Save Complete.')