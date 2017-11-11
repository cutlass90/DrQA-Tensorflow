import random

import numpy as np
import msgpack

from prepro import annotate_multiproc, to_id
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

def make_inf_sample(context, question):
    row = annotate_multiproc([(1, context, question, 'fake_answer', 42, 42)], True)[0]
    with open(c.path_to_meta, 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    w2id = {w:id_ for id_, w in enumerate(meta['vocab'])}
    tag2id = {tag:id_ for id_, tag in enumerate(meta['vocab_tag'])}
    ent2id = {ent:id_ for id_, ent in enumerate(meta['vocab_ent'])}
    row = to_id(row, w2id, tag2id, ent2id)
    row = list(row)
    row[-1] = 42
    row[-2] = 42
    row = reader(row)[0]
    row = [np.expand_dims(a, 0) for a in row]
    return row

def get_answer(context, question, model):
    original_row = annotate_multiproc([(1, context, question, 'fake_answer', 42, 42)], True)[0]
    context_text = original_row[-4]
    words_edges = original_row[-3]
    data = make_inf_sample(context, question)
    question = data[0]
    context = data[1]
    pos = data[2]
    ner = data[3]
    context_features = data[4]
    start_pos, end_pos = model.predict(question, context, pos, ner, context_features)
    start_pos = int(start_pos)
    end_pos = int(end_pos)
    print('start_pos, end_pos', start_pos, end_pos)
    return context_text[words_edges[start_pos][0]:words_edges[end_pos][1]]

if __name__ == '__main__':
    pass
    # with open(c.path_to_context, 'r') as f:
    #     context = f.read()
    # question = 'What was his mother\'s name?'
    # get_answer(context, question, model=1)

    with open(c.path_to_context, 'r') as f:
        context = f.read()
    question = 'What was his mother\'s name?'
    batch = make_inf_sample(context, question)
    [print(i.shape) for i in batch ]



    # import msgpack
    # from dslib.generic.measurements import Timer
    
    # with open('SQuAD1/data.msgpack', 'rb') as f:
    #     data = msgpack.load(f, encoding='utf8')
    # data = data['train'][:1000]

    # with Timer() as t:
    #     [reader(row) for row in data]
