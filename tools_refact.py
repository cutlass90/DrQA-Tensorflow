import random
import json
import re
import os
import collections
import pickle
import unicodedata
from collections import defaultdict

import csv
import pandas as pd
from tqdm import tqdm
import spacy
import numpy as np

from dslib.generic.measurements import Timer

class SingletonDecorator:
    def __init__(self, klass):
        self.klass = klass
        self.instance = None
    def __call__(self,*args, **kwds):
        if self.instance == None:
            self.instance = self.klass(*args, **kwds)
        return self.instance

class Spacer:
    def __init__(self):
        self.nlp = spacy.load('en', parser=False)
    def __call__(self, *args, **kwargs):
        return self.nlp(*args, **kwargs)

Spacer = SingletonDecorator(Spacer)

class Answer:
    """ Class that represent an answer. """
    def __init__(self, text, start_char):
        # text: str, answer text
        # start_char: int, position of answer begin in chars
        self.text = text
        self.start_char = start_char

    def set_start_end_tokens(self, context_token_span):
        def get_token_pos(char_pos, spans):
            for i, span in enumerate(spans):
                s, e = span
                if char_pos >= s and char_pos <= e:
                    return i
                if i == (len(spans)-1):
                    print('spans:', spans)
                    print('char_pos', char_pos)
                    raise ValueError('Cannot find token position')
        self.start_token = get_token_pos(self.start_char, context_token_span)
        self.end_token = get_token_pos(self.start_char+len(self.text),
                                       context_token_span) + 1

class QA:
    """ Class that represent single question and multiple answers. """
    def __init__(self, question, answers):
        # question: str, single question
        # answers: list of Answer instances
        self.question = question
        self.answers = answers

        self._process_question()
    
    def _process_question(self):
        nlp = Spacer()
        self.question_tokens = process_question(self.question, nlp)

    def get_answer(self):
        """ Return random answer from answers. """
        return random.choice(self.answers)

class Paragraph:
    """ Class represent single context and multiple question
        for it with answers. """
    def __init__(self, context, qas):
        # context: str, single context
        # qas: list of QA instances
        self.context = context
        self.qas = qas

        self._process_context()
        self._get_context_features()
        self._set_start_end_tokens()
    
    def _process_context(self):
        nlp = Spacer()
        context_tokens, context_tags, context_ents,\
            context_token_span = process_context(self.context, nlp)
        self.context_tokens = context_tokens
        self.context_tags = context_tags
        self.context_ents = context_ents
        self.context_token_span = context_token_span
    
    def _get_context_features(self):
        nlp = Spacer()
        for qa in self.qas:
            qa.context_featurs = get_context_features(self.context,
                                                      qa.question,
                                                      self.context_tokens,
                                                      qa.question_tokens,
                                                      nlp)
    
    def _set_start_end_tokens(self):
        for qa in self.qas:
            for answer in qa.answers:
                answer.set_start_end_tokens(self.context_token_span)


    def get_qa(self):
        """ Return random qa from qas. """
        return random.choice(self.qas)

####################### preprocessing ##########################################
def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def get_context_features(context, question, context_tokens, question_tokens, nlp):
    """ For each word in context return set of features:
        match_origin: bool, whether word in question
        match_lower: bool, whether word.lower() in question.lower()
        match_lemma: bool, whether word.lemma() in question.lemma()
        context_tf: float, term frequency of word in document
    """
    c_doc = nlp(re.sub(r'\s', ' ', context))
    q_doc = nlp(re.sub(r'\s', ' ', question))    
    context_tokens_lower = [w.lower() for w in context_tokens]
    question_tokens_lower_set = set([w.lower() for w in question_tokens])
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()
        for w in q_doc}
    match_origin = [w in set(question_tokens) for w in context_tokens]
    match_lower = [w in question_tokens_lower_set for w in context_tokens_lower]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower())
        in question_lemma for w in c_doc]
    # term frequency in document
    counter_ = collections.Counter(context_tokens_lower)
    total = len(context_tokens_lower)
    context_tf = [counter_[w] / total for w in context_tokens_lower]
    context_features = list(zip(match_origin, match_lower, match_lemma, context_tf))
    return context_features

def process_context(context, nlp):
    """ Return context_tokens, context_tags, context_ents, context_token_span
        for provided context.
    
    Args:
        context: str, context text
    
    Return:
        context_tokens: list of str
        context_tags: list of str
        context_ents: list of str
        context_token_span: list of tuple with start and end tokens positions
    """
    c_doc = nlp(re.sub(r'\s', ' ', context))
    context_token_span = [(w.idx, w.idx + len(w.text)) for w in c_doc]
    context_tags = [w.tag_ for w in c_doc]
    context_ents = [w.ent_type_ for w in c_doc]
    context_tokens = [normalize_text(w.text) for w in c_doc]
    return context_tokens, context_tags, context_ents, context_token_span

def process_question(question, nlp):
    """ Return list of str with tokenized question. """
    q_doc = nlp(re.sub(r'\s', ' ', question))
    question_tokens = [normalize_text(w.text) for w in q_doc]
    return question_tokens

def tokens_to_id(tokens, vocab, pad_symbol):
    vocab = defaultdict(lambda: pad_symbol, vocab)
    return np.array([vocab[t] for t in tokens])

def get_meta(c):
    glove_vectors = pd.read_table(c.path_to_embeddings, sep=" ", index_col=0,
        header=None, quoting=csv.QUOTE_NONE)
    indexes = []
    with open(c.path_to_embeddings) as f:
        for line in f:
            indexes.append(normalize_text(line.rstrip().split(' ')[0]))
    glove_vectors = pd.DataFrame(glove_vectors.as_matrix(), index=indexes)
    glove_vectors = glove_vectors[:c.dict_size-2]
    pad = pd.DataFrame(np.zeros([1, c.emb_size]), index=['<PAD>'])
    unk = pd.DataFrame(np.ones([1, c.emb_size]), index=['<UNK>'])
    glove_vectors = pd.concat([pad, unk, glove_vectors], axis=0)
    embeddings = glove_vectors.as_matrix()
    w2id = {w:i for i, w in enumerate(glove_vectors.index)}
    tag2id = {w:i for i, w in enumerate(c.vocab_tag)}
    ent2id = {w:i for i, w in enumerate(c.vocab_ent)}

    meta = {'emb':embeddings, 'w2id':w2id, 'tag2id':tag2id, 'ent2id':ent2id}
    os.makedirs(os.path.dirname(c.path_to_meta), exist_ok=True)    
    with open(c.path_to_meta, 'wb') as f:
        pickle.dump(meta, f)

    return embeddings, w2id, tag2id, ent2id
######################## gather functions ######################################

def read_set(path_to_json, path_to_save):
    """ Read dataset as list of Paragraph objects.

    Args:
        path_to_set: str, path to json file with dataset

    Return:
        list of Paragraph objects
    """
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

    paragraphs = []
    data = json.load(open(path_to_json, 'r'))
    data = [parag for title_data in data['data'] for parag in title_data['paragraphs']]
    for parag in tqdm(data):
        qas = []
        for qa in parag['qas']:
            answers = []
            for a in qa['answers']:
                answers.append(Answer(a['text'], a['answer_start']))
            qas.append(QA(qa['question'], answers))
        paragraphs.append(Paragraph(parag['context'], qas))
    with open(path_to_save, 'wb') as f:
        pickle.dump(paragraphs, f)
    return paragraphs

def load_pickled_object(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def read(paragraph, w2id, tag2id, ent2id, c):
    """ Conver to id, make padding and prepare data for training. """
    def filler(data, container, max_size, s=None):
        size = len(data)
        if size <= max_size:
            container[:size] = data
            s = None
        else:
            if s is None:
                s = random.randint(0, size - max_size)
            container[:] = data[s:s + max_size]
        return container, s
    # import pdb
    # pdb.set_trace()
    #get data
    qa = paragraph.get_qa()
    answer = qa.get_answer()
    a_s, a_e = answer.start_token, answer.end_token
    
    # convert to id
    question_id = tokens_to_id(qa.question_tokens, w2id, 1)
    context_id = tokens_to_id(paragraph.context_tokens, w2id, 1)
    pos_id = tokens_to_id(paragraph.context_tags, tag2id, 0)
    ner_id = tokens_to_id(paragraph.context_ents, ent2id, 0)
    
    # create blank
    question = np.zeros([c.question_size], dtype=int)
    context = np.zeros([c.context_size], dtype=int)
    pos = np.zeros([c.context_size], dtype=int)
    ner = np.zeros([c.context_size], dtype=int)
    context_features = -1*np.ones([c.context_size, 4], dtype=float)

    # fill blank
    # pos_size = len(c.vocab_tag)
    # ner_size = len(c.vocab_ent)
    question, _ = filler(question_id, question, c.question_size)
    context, s = filler(context_id, context, c.context_size)
    pos, _ = filler(pos_id, pos, c.context_size, s)
    ner, _ = filler(ner_id, ner, c.context_size, s)
    context_features, _ = filler(qa.context_featurs, context_features,
                                 c.context_size, s)
    print('s', s)
    if s is None:
        answer = np.zeros([c.context_size], dtype=float)
        answer[a_s:a_e] = 1
    else:
        answer = np.zeros([len(context_id)])
        answer[a_s:a_e] = 1
        answer = answer[s:s+c.context_size]
    
    return [(question, context, pos, ner, context_features, answer)]



class Reader:

    def __init__(self, w2id, tag2id, ent2id, c):
        [setattr(self, name, val) for name,val in locals().items() if name != 'self']



if __name__ == '__main__':
    from config import config as c
    # read_set(c.path_to_train_json, c.path_to_train_data)
    # read_set(c.path_to_test_json, c.path_to_est_data)
    # get_meta(c)

    w2id, tag2id, ent2id = load_pickled_object('./w_tag_ent.pkl')
    data = load_pickled_object('./SQuAD/train.pkl')
    read(data[0], w2id, tag2id, ent2id, c)