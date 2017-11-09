import random

import numpy as np

from config import config as c

def fake_reader(x):
    questions = np.random.randint(0, 90000, [c.question_size])
    context = np.random.randint(0, 90000, [c.context_size])
    pos = np.random.randint(0, 10, [c.context_size])
    ner = np.random.randint(0, 10, [c.context_size])
    context_features = np.random.normal(0, 1, [c.context_size, 4])
    target_start = np.zeros([c.context_size], dtype=float)
    target_start[0] = 1
    target_end = np.zeros([c.context_size], dtype=float)
    target_end[0] = 1
    return [(questions, context, pos, ner, context_features, target_start, target_end)]

def reader(row):
    """ Make padding and prepare data for training. """
    def padder(x, pad_val, target_size):
        pad_len = target_size - len(x)
        # print('pad_len', pad_len)
        if pad_len < 0:
            return x[:target_size]
        x = np.pad(x, (0, pad_len), 'constant',
                   constant_values=(0, pad_val))
        return x
    questions = np.array(row[5], dtype=int)
    context = np.array(row[1], dtype=int)
    pos = np.array(row[3], dtype=int)
    ner = np.array(row[4], dtype=int)
    context_features = np.array(row[2], dtype=np.float32)
    start = row[8]
    end = row[9]

    if start >= c.context_size:
        start = random.randint(0, c.context_size - 2)
    if end >= c.context_size:
        end = random.randint(0, c.context_size - 2)
    
    target_start = np.zeros([c.context_size], dtype=float)
    target_start[start] = 1
    target_end = np.zeros([c.context_size], dtype=float)
    target_end[end] = 1


    questions = padder(questions, 0, c.question_size)
    context = padder(context, 0, c.context_size)
    pos = padder(pos, c.pos_size, c.context_size)
    ner = padder(ner, c.ner_size, c.context_size)

    if c.context_size < len(context_features):
        context_features = context_features[:c.context_size]
    context_features = np.pad(context_features,
                              [(0, c.context_size-len(context_features)),(0,0)], 
                              'constant', constant_values = -1)
    return [(questions, context, pos, ner, context_features, target_start, target_end)]
     