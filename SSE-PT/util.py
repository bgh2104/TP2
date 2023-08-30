import sys
import copy
import random
import numpy as np
from collections import defaultdict
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]
        #print(predictions)
        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < args.k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 1000 == 0:
            #print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < args.k:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            #print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def get_user_seq(dataset, args):
    user_train, _, _, _, _ = dataset
    return user_train[args.user_id]

def pad_sequence(sequence, args):
    padded_sequence = sequence[:args.maxlen] + [0] * (args.maxlen - len(sequence))
    return padded_sequence

def load_encoders(file_path):
    """인코딩 했던 정보를 다시 가져오기 위한 함수"""
    df = pd.read_csv(file_path, sep=' ', header=None, names=['userId', 'movieId'])
    user_encoder = {original: idx for idx, original in enumerate(df['userId'].unique(), start=1)}
    movie_encoder = {original: idx for idx, original in enumerate(df['movieId'].unique(), start=1)}

    # decoder는 encoder의 반대입니다.
    user_decoder = {idx: original for original, idx in user_encoder.items()}
    item_decoder = {idx: original for original, idx in movie_encoder.items()}

    return user_decoder, item_decoder