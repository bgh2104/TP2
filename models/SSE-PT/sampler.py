import random
import numpy as np
from multiprocessing import Process, Queue
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen,  
                    threshold_user, threshold_item,
                    result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])

        for i in reversed(user_train[user][:-1]):
            #seq[idx] = i
            
            # SSE for user side (2 lines)
            if random.random() > threshold_item:
                i = np.random.randint(1, itemnum + 1)
                nxt = np.random.randint(1, itemnum + 1)
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        
        # SSE for item side (2 lines)
        if random.random() > threshold_user:
            user = np.random.randint(1, usernum + 1)
        # equivalent to hard parameter sharing
        #user = 1	
     
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, 
                 threshold_user=1.0, threshold_item=1.0, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            seed = np.random.randint(2e9)
            p = Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      threshold_user,
                                                      threshold_item,
                                                      self.result_queue,
                                                      seed))
            p.daemon = True
            p.start()
            self.processors.append(p)

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
