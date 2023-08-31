import os
import time
import pickle
import joblib
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sampler import WarpSampler
from model_v1 import Model
from util import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import Dataloader

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml1m')
parser.add_argument('--train_dir', default='first')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--user_hidden_units', default=50, type=int)
parser.add_argument('--item_hidden_units', default=100, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--threshold_user', default=0.08, type=float)
parser.add_argument('--threshold_item', default=0.9, type=float)
parser.add_argument('--l2_emb', default=0.00005, type=float)
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--k', default=10, type=int)

# --inference_only 및 --state_dict_path 옵션 추가
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

# 유저 아이디를 입력으로 받는 옵션 추가
parser.add_argument('--user_id', default=None, type=int, help='User ID for movie recommendations')

args = parser.parse_args()

if not args.inference_only:
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        params = '\n'.join([str(k) + ',' + str(v) 
            for k, v in sorted(vars(args).items(), key=lambda x: x[0])])
        # print(params)
        f.write(params)
    f.close()

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu) # 맥북기준으로 gpu사용 못함 그래서 일단 코드는 냅두고 defalut값을 1로 설정해둠.

if __name__ == '__main__':
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    max_len = 0
    for u in user_train:
        cc += len(user_train[u])
        max_len = max(max_len, len(user_train[u]))
    if not args.inference_only:
        print("\nThere are {0} users {1} items \n".format(usernum, itemnum))
        print("Average sequence length: {0}\n".format(cc / len(user_train)))
        print("Maximum length of sequence: {0}\n".format(max_len))

    if not args.inference_only:
        f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
        loss_values = []  # 손실 값들을 저장할 리스트 추가

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    sampler = WarpSampler(user_train, usernum, itemnum, 
                batch_size=args.batch_size, maxlen=args.maxlen,
                threshold_user=args.threshold_user, 
                threshold_item=args.threshold_item,
                n_workers=3)
    model = Model(usernum, itemnum, args)
    sess.run(tf.global_variables_initializer()) # 초기화
    
    # 모델 가중치 로드 부분
    if args.inference_only and args.state_dict_path is not None:
        saver = tf.train.Saver()
        saver.restore(sess, args.state_dict_path)
        print("Model restored from", args.state_dict_path)

    # 유저아이디에 맞게 영화 추천리스트 출력
        user_seq = get_user_seq(dataset, args)
        user_seq_input = pad_sequence(user_seq, args)  # 시퀀스 패딩
        user_seq_input = np.expand_dims(user_seq_input, axis=0)  # 배치 차원 추가
        item_idx_array = np.array(list(range(1, itemnum+1)))
        
        # 모델로 추론
        scores = -model.predict(sess, u=np.array([args.user_id]), seq=user_seq_input, item_idx=item_idx_array)
    
        # 영화 제목 매핑
        path = './data'
        movies_df = Dataloader.load_movies(path)
        movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

        user_decoder, item_decoder = load_encoders('./data/ratings_SSE_PT.txt')
        
        # 유저가 시청한 영화
        watched_movies = set(user_train[args.user_id])

        # 추천 영화 리스트를 생성합니다. (5개 추천)
        recommended_items = []
        for item, score in enumerate(scores[0]):
            if item not in watched_movies:  # 이미 시청한 영화는 제외
                recommended_items.append((item, score))
        
        # 추천 영화를 점수 내림차순으로 정렬
        recommended_items.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개의 추천 영화 선택
        top_recommendations = recommended_items[:5]

        # 영화 제목 매핑
        recommendations = []
        for item, score in top_recommendations:
            movie_id = item_decoder.get(item, "Unknown")
            title = movie_id_to_title.get(movie_id, "Unknown")
            recommendations.append((movie_id, title))

        # 추천 결과 출력
        print("Recommended movies for user {0}:".format(args.user_id))
        for movie_id, title in recommendations:
            print("{0} - {1}".format(movie_id, title))

        # # 추천 영화 리스트를 생성합니다.(5개 추천)
        # recommended_items = scores[0].argsort()[::-1][:5]
        # recommendations = []
        # for item in recommended_items:
        #     movie_id = item_decoder.get(item, "Unknown")
        #     title = movie_id_to_title.get(movie_id, "Unknown")
        #     recommendations.append((movie_id, title))

        # # 추천 결과 출력
        # print("Recommended movies for user {0}:".format(args.user_id))
        # for movie_id, title in recommendations:
        #     print("{0} - {1}".format(movie_id, title))


    # 학습하기 위한 코드들 -> inference_only가 False(default)일경우에만
    if not args.inference_only:
        T = 0.0
        t_test = evaluate(model, dataset, args, sess)
        t_valid = evaluate_valid(model, dataset, args, sess)
        print("[0, 0.0, {0}, {1}, {2}, {3}],".format(t_valid[0], t_valid[1], t_test[0], t_test[1]))

        t0 = time.time()

        best_valid_ndcg = -1.0  # 초기 가장 좋은 성능을 낮은 값으로 설정
        best_epoch = -1

        for epoch in range(1, args.num_epochs + 1):
            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                user_emb_table, item_emb_table, attention, auc, loss, _ = sess.run([model.user_emb_table, model.item_emb_table, model.attention, model.auc, model.loss, model.train_op],
                                            {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                            model.is_training: True})

            if epoch % args.print_freq == 0:
                t1 = time.time() - t0
                T += t1
                t_test = evaluate(model, dataset, args, sess)
                t_valid = evaluate_valid(model, dataset, args, sess)
                print("[epoch : {0}, time : {1}, valid NDCG@10 : {2}, valid HR@10 : {3}, test NDCG@10 : {4}, test HR@10 : {5}],".format(epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
                f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()
                t0 = time.time()
                
                # 10에폭마다 손실 값을 저장
                loss_values.append(loss)

                # 가장 좋은 성능을 보이는 모델 가중치 저장
                if not args.inference_only and t_valid[0] > best_valid_ndcg:
                    best_valid_ndcg = t_valid[0]
                    best_epoch = epoch
                    saver = tf.train.Saver()
                    model_path = os.path.join(args.dataset + '_' + args.train_dir, f'best_model.ckpt')
                    saver.save(sess, model_path)
                    print(f"Best model saved at {model_path}")
        
        # 손실 값들을 joblib를 사용하여 파일에 저장
        loss_filename = os.path.join(args.dataset + '_' + args.train_dir, 'loss_values.joblib')
        joblib.dump(loss_values, loss_filename)

        f.close()
        sampler.close()
        print("Done")