# Surprise 라이브러리를 활용한 KNN_Basic + SVD Recommendation

- Surprise 라이브러리를 활용해 간단하게 만들어본 하이브리드 모델입니다.
- 하이브리드 모델을 만들때 결합방식을 사용하였으며 KNN_Basic(0.3) + SVD(0.7)을 사용하였습니다.

## 실행방법

- 먼저 터미널상에서 TP2경로로 이동한 뒤
```
python knn_svd_hybrid.py
```
- 위 코드를 실행시키면 유저아이디를 입력하라는 창이 나오고 거기에 1~6040 까지의 숫자중에 하나를 쓰고 엔터를 누르면 작동됩니다.

## 성능
* RMSE : 0.863
* Recall@10 : 0.562
* Precision@10 : 0.797