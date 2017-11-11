import random
import json
import re
import os
import collections
import pickle
import unicodedata

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
    for parag in tqdm(data[:10]):
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

if __name__ == '__main__':
    paragraphs = read_set('../../../datasets/SQuAD/dev-v1.1.json', './SQuAD/temp.pkl')
