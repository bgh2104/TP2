# AIB 18기 Team Project2 레포입니다.
* 팀명 : 오늘뭐보지?!
* 팀장 : 배나연
* 팀원 : 조영재, 정영현, 성진현
* 주제 : 영화 추천 시스템 제작 프로젝트

### 팀 규칙
* 매일 오전 10시, 오후 3시 회의

### 비개인화 / 개인화 추천시스템(models폴더)
* 비개인화 MODEL : 연관규칙분석(Association_rule_analysis), EASE(Recbole 라이브러리 활용)
* 개인화 MODEL : SVDPP(협업필터링), KNN_BASIC+SVD(Hybrid), -> Surprise 라이브러리 활용
                LSTM(인공신경망 Sequential MODEL), SSE-PT(Sequential MODEL),
                TiSASRec(Sequential MODEL)

### utils
* Dataloader.py : dat파일 형식의 데이터 불러오는 파일
* Preprocessing.py : 기존 베이스코드에 있던 토큰화 함수들은 쓰지않고 전처리하는데에 필요한 함수를 추가하여 활용함.

### data
* ml-1m folder : Recbole 라이브러리에서 제공해주는 Movielens1M 파일
- .inter -> ratings, .item -> movies, .user -> users
* movies_metadata.csv : Kaggle에서 사용한 movie데이터, 비슷한데이터라 tag를 참고해보려 했으나 기존의 영화가 많이 날라감. 
- https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
* 기존 Movielens1M Data : movies.dat, ratings.dat, users,dat
* SSE-PT, TiSASRec 모델들의 경우 data폴더가 겹치기도 하고 path설정이 복잡해지는 관계로 따로 분리 하지 않고 각 모델 폴더에 담아둠.

### EDA
* 각 팀원들의 초성을 따 EDA한 ipynb파일들 업로드