# SSE-PT: Sequential Recommendation Via Personalized Transformer
We implement our code in Tensorflow and the code is tested under a server with 40-core Intel Xeon E5-2630 
v4 @ 2.20GHz CPU, 256G RAM and Nvidia GTX 1080 GPUs (with TensorFlow 1.13 and Python 3). The code is also hosted at https://github.com/wuliwei9278/SSE-PT.

## SSE-PT모델을 가져와서 사용했지만 아직 불안정한 상태입니다.
Tensorflow 1.x 버전에서 업그레이드를 시키지 못해
```
import tensorflow.compat.v1 as tf
```
로 사용중이라 warning표시가 많이 뜹니다.

SSE-PT 경로로 설정한뒤 터미널환경에
```
python3 main.py
```
로 학습하고 실행은 가능합니다.

학습한 모델의 가중치를 가져와 유저아이디에 맞게 영화리스트를 살펴보려면(Default 기준으로)
```
python3 main.py --inference_only=True --state_dict_path='ml1m_first/best_model.ckpt' --user_id= 유저아이디 입력[ex) 1, 2, 3, ..., 6040] 사이 숫자로
```

## Datasets
The preprocessed datasets are in the `data` directory (`e.g. data/ml1m.txt`). Each line of the `txt` format data contains
a `user id` and an `item id`, where both user id and item id are indexed from 1 consecutively. Each line represents one interaction between the user 
and the item. For every user, their interactions were sorted by timestamp.

## Papers
Our paper has been accepted to ACM Recommender Systems Conference 2020 for long paper and our pre-print version is on [arxiv](https://arxiv.org/abs/1908.05435) or our ICLR borderline-rejected version https://openreview.net/forum?id=HkeuD34KPH.
One can cite one of below for now:
```
@article{wu2019temporal,
  title={Temporal Collaborative Ranking Via Personalized Transformer},
  author={Wu, Liwei and Li, Shuqing and Hsieh, Cho-Jui and Sharpnack, James},
  journal={arXiv preprint arXiv:1908.05435},
  year={2019}
}
```
or
```
@misc{
wu2020ssept,
  title={{\{}SSE{\}}-{\{}PT{\}}: Sequential Recommendation Via Personalized Transformer},
  author={Liwei Wu and Shuqing Li and Cho-Jui Hsieh and James Sharpnack},
  year={2020},
  url={https://openreview.net/forum?id=HkeuD34KPH}
}
```

It is worth noting that a new regualrization technique called SSE is used. One can refer to the paper below for more details:
[Stochastic Shared Embeddings: Data-driven Regularization of Embedding Layers](https://arxiv.org/abs/1905.10630). The paper has been accepted to NeurIPS 2019. We presented the work at Vancouver, Canada. Another git repo is at https://github.com/wuliwei9278/SSE.
```
@article{wu2019stochastic,
  title={Stochastic Shared Embeddings: Data-driven Regularization of Embedding Layers},
  author={Wu, Liwei and Li, Shuqing and Hsieh, Cho-Jui and Sharpnack, James},
  journal={arXiv preprint arXiv:1905.10630},
  year={2019}
}
```

## Options
The training of the SSE-PT model is handled by the main.py script that provides the following command line arguments.
```
--dataset            STR           Name of dataset.               Default is "ml1m".
--train_dir          STR           Train directory.               Default is "default".
--batch_size         INT           Batch size.                    Default is 128.    
--lr                 FLOAT         Learning rate.                 Default is 0.001.
--maxlen             INT           Maxmum length of sequence.     Default is 50.
--user_hidden_units  INT           Hidden units of user.          Default is 50.
--item_hidden_units  INT           Hidden units of item.          Default is 50.
--num_blocks         INT           Number of blocks.              Default is 2.
--num_epochs         INT           Number of epochs to run.       Default is 2001.
--num_heads          INT           Number of heads.               Default is 1.
--dropout_rate       FLOAT         Dropout rate value.            Default is 0.5.
--threshold_user     FLOAT         SSE probability of user.       Default is 1.0.
--threshold_item     FLOAT         SSE probability of item.       Default is 1.0.
--l2_emb             FLOAT         L2 regularization value.       Default is 0.0.
--gpu                INT           Name of GPU to use.            Default is 0.
--print_freq         INT           Print frequency of evaluation. Default is 10.
--k                  INT           Top k for NDCG and Hits.       Default is 10.

# --inference_only 및 --state_dict_path 옵션 추가
--inference_only     STR2Bool      학습진행하지 않도록 하기위해 추가      Default is False
--state_dict_path    STR           가중치 저장되어있는 경로 추가         Default is None

# 유저 아이디를 입력으로 받는 옵션 추가
--user_id            INT          영화 예측 진행하기위해 유저아이디 추가   Default is None
```

## 주의할점
Warning 문제도 자주 발생하고 예측리스트를 가져오는데 있어서 유저아이디가 달라져도 같은영화 추천이 나오는경우가 있음...
model_v1.py에 있는 predict 함수를 수정해야할것 같은데 정확한 원인을 체크하지 못함.