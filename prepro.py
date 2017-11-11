import re
import os
import json
import spacy
import msgpack
import unicodedata
import numpy as np
import argparse
import collections
import multiprocessing
from multiprocessing import Pool
import logging

from config import config as c

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)
nlp = None

os.makedirs('./SQuAD', exist_ok=True)


parser = argparse.ArgumentParser(
    description='Preprocessing data files, about 10 minitues to run.')

parser.add_argument('--sort_all', action='store_true',
                    help='sort the vocabulary by frequencies of all words. '
                        'Otherwise consider question words first.')
parser.add_argument('--sample_size', type=int, default=0,
                    help='size of sample data (for debugging).')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')

args = parser.parse_args()


def flatten_json(data_file, mode):
    """Flatten each article in training data."""
    with open(data_file) as f:
        data = json.load(f)['data']
    rows = []
    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                id_, question, answers = qa['id'], qa['question'], qa['answers']
                if mode == 'train':
                    answer = answers[0]['text']  # in training data there's only one answer
                    answer_start = answers[0]['answer_start']
                    answer_end = answer_start + len(answer)
                    rows.append((id_, context, question, answer, answer_start, answer_end))
                else:  # mode == 'dev'
                    answers = [a['text'] for a in answers]
                    rows.append((id_, context, question, answers))
    return rows

########################### preprocess data ####################################

def preprocess_context_question(context, question):
    global nlp
    q_doc = nlp(clean_spaces(question))
    c_doc = nlp(clean_spaces(context))
    question_tokens = [normalize_text(w.text) for w in q_doc]
    context_tokens = [normalize_text(w.text) for w in c_doc]
    question_tokens_lower = [w.lower() for w in question_tokens]
    context_tokens_lower = [w.lower() for w in context_tokens]
    context_token_span = [(w.idx, w.idx + len(w.text)) for w in c_doc]
    context_tags = [w.tag_ for w in c_doc]
    context_ents = [w.ent_type_ for w in c_doc]
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()
        for w in q_doc}
    question_tokens_set = set(question_tokens)
    question_tokens_lower_set = set(question_tokens_lower)
    match_origin = [w in question_tokens_set for w in context_tokens]
    match_lower = [w in question_tokens_lower_set for w in context_tokens_lower]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower())
        in question_lemma for w in c_doc]
    # term frequency in document
    counter_ = collections.Counter(context_tokens_lower)
    total = len(context_tokens_lower)
    context_tf = [counter_[w] / total for w in context_tokens_lower]
    context_features = list(zip(match_origin, match_lower, match_lemma, context_tf))

    print('context_token_span', context_token_span)
    print(context[:30])
    print(context[context_token_span[1][0]:context_token_span[1][1]])

    return (context_tokens, context_features, context_tags, context_ents,
            question_tokens, context, context_token_span)

def index_answer(row):
    token_span = row[-4]
    starts, ends = zip(*token_span)
    answer_start = row[-2]
    answer_end = row[-1]
    try:
        return row[:-3] + (starts.index(answer_start), ends.index(answer_end))
    except ValueError:
        return row[:-3] + (None, None)

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def clean_spaces(text):
    """normalize spaces in a string."""
    text = re.sub(r'\s', ' ', text)
    return text

###############################################################################


def annotate(row):
    id_, context, question = row[:3]
    processed_c_q = preprocess_context_question(context, question)
    return (id_,) + processed_c_q + row[3:]

def init():
    """initialize spacy in each process"""
    global nlp
    nlp = spacy.load('en', parser=False)


def build_vocab(questions, contexts, wv_vocab):
    """
    Build vocabulary sorted by global word frequency, or consider frequencies in questions first,
    which is controlled by `args.sort_all`.
    """
    if args.sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in wv_vocab], key=counter.get,
                        reverse=True)
    else:
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in wv_vocab], key=counter_q.get,
                        reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in wv_vocab],
                        key=counter.get, reverse=True)
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab coverage {1}/{0} | OOV occurrence {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    return vocab, counter

def to_id(row, w2id, tag2id, ent2id, unk_id=1):
    context_tokens = row[1]
    context_features = row[2]
    context_tags = row[3]
    context_ents = row[4]
    question_tokens = row[5]
    question_ids = [w2id[w] if w in w2id else unk_id for w in question_tokens]
    context_ids = [w2id[w] if w in w2id else unk_id for w in context_tokens]
    tag_ids = [tag2id[w] for w in context_tags]
    ent_ids = [ent2id[w] for w in context_ents]
    return (row[0], context_ids, context_features, tag_ids, ent_ids, question_ids) + row[6:]

def get_wv_vocab(wv_file):
    """ Load vocabulary from word vector files. """
    wv_vocab = set()
    with open(wv_file) as f:
        for line in f:
            token = normalize_text(line.rstrip().split(' ')[0])
            wv_vocab.add(token)
    log.info('glove vocab loaded.')
    return wv_vocab

def annotate_multiproc(flattened, ignore_inconsistent_samples=False):
    workers = Pool(args.threads, initializer=init)
    train = workers.map(annotate, flattened)
    workers.close()
    workers.join()
    train = list(map(index_answer, train))
    initial_len = len(train)
    if not ignore_inconsistent_samples:
        train = list(filter(lambda x: x[-1] is not None, train))
    log.info('drop {} inconsistent samples.'.format(initial_len - len(train)))
    log.info('tokens generated')
    return train

def process_json(path_to_json, mode):
    flattened = flatten_json(path_to_json, mode)
    log.info('json data flattened.')
    return annotate_multiproc(flattened)

def process_local_vocab(train, dev, wv_vocab):
    full = train + dev
    vocab, counter = build_vocab([row[5] for row in full],
                                 [row[1] for row in full], wv_vocab)
    counter_tag = collections.Counter(w for row in full for w in row[3])
    vocab_tag = sorted(counter_tag, key=counter_tag.get, reverse=True)
    counter_ent = collections.Counter(w for row in full for w in row[4])
    vocab_ent = sorted(counter_ent, key=counter_ent.get, reverse=True)
    w2id = {w: i for i, w in enumerate(vocab)}
    tag2id = {w: i for i, w in enumerate(vocab_tag)}
    ent2id = {w: i for i, w in enumerate(vocab_ent)}
    log.info('Vocabulary size: {}'.format(len(vocab)))
    log.info('Found {} POS tags.'.format(len(vocab_tag)))
    log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))
    return w2id, tag2id, ent2id, vocab_ent, vocab_tag, vocab



def getting_embedding_matrix(vocab, wv_dim, path_to_wv_file, w2id):
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, wv_dim))
    embed_counts = np.zeros(vocab_size)
    embed_counts[:2] = 1  # PADDING & UNK
    with open(path_to_wv_file) as f:
        for line in f:
            elems = line.rstrip().split(' ')
            token = normalize_text(elems[0])
            if token in w2id:
                word_id = w2id[token]
                embed_counts[word_id] += 1
                embeddings[word_id] += [float(v) for v in elems[1:]]
    embeddings /= embed_counts.reshape((-1, 1))
    log.info('got embedding matrix.')
    return embeddings


def save_meta(vocab, vocab_tag, vocab_ent, embeddings, path_to_save):
    meta = {
        'vocab': vocab,
        'vocab_tag': vocab_tag,
        'vocab_ent': vocab_ent,
        'embedding': embeddings.tolist()
    }
    with open(path_to_save, 'wb') as f:
        msgpack.dump(meta, f)
    log.info('meta was saved to disk.')
    

def save_data(train, dev, path_to_save):
    result = {
        'train': train,
        'dev': dev
    }
    # train: id, context_id, context_features, tag_id, ent_id,
    #        question_id, context, context_token_span, answer_start, answer_end
    # dev:   id, context_id, context_features, tag_id, ent_id,
    #        question_id, context, context_token_span, answer
    with open(path_to_save, 'wb') as f:
        msgpack.dump(result, f)
    log.info('data was saved to disk.')

def main(args):
    log.info(vars(args))
    log.info('start data preparing...')

    train_set = process_json(c.path_to_train_json, 'train')
    val_set = process_json(c.path_to_val_json, 'train')

    wv_vocab = get_wv_vocab(c.path_to_vector_embeddings)
    w2id, tag2id, ent2id, vocab_ent, vocab_tag, vocab = process_local_vocab(
        train_set, val_set, wv_vocab)

    train_set = [to_id(row, w2id, tag2id, ent2id) for row in train_set]
    val_set = [to_id(row, w2id, tag2id, ent2id) for row in val_set]
    log.info('converted to ids.')
    save_data(train_set, val_set, c.path_to_data)

    embeddings =  getting_embedding_matrix(vocab,
                                           c.emb_size,
                                           c.path_to_vector_embeddings,
                                           w2id)
    save_meta(vocab, vocab_tag, vocab_ent, embeddings, c.path_to_meta)

if __name__ == '__main__':

    main(args)