레퍼런스:

Jiacheng Li, Yujie Wang, Julian McAuley (2020). Time Interval Aware Self-Attention for Sequential Recommendation. WSDM'20

이 코드는 TensorFlow 환경에서의 Window 데스크탑 (GTX 1080 Ti GPU 포함)에서 테스트되었습니다.

데이터셋
이 저장소에는 예시로 ml-1m 데이터셋이 포함되어 있습니다.

모델 학습
ml-1m 데이터셋에서 모델을 학습시키려면 (기본 하이퍼 파라미터 사용):
터미널의 통해 경로를 TISASREC에 위치시킨 후 사용해주시면 됩니다.
```
python main.py --dataset=ml-1m --train_dir=default 
```
기타
셀프 어텐션의 구현은 https://github.com/Kyubyong/transformer을 기반으로 하였습니다

Contact

질문이 있으시면 이메일 (30dayslife@gmail.com)로 연락해주세요.