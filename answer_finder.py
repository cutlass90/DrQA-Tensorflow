from pprint import pprint

import tensorflow as tf
import numpy as np
import pickle
import tqdm

from dslib.tf_utils.base_model import BaseModel
from dslib.tf_utils import misc, metrics
from dslib.tf_utils.tools import time_distributed

from tools import load_pickled_object

MAX_TO_KEEP = 100

class AnswerFinder(BaseModel):

    def __init__(self, config, mode, restore=True, verbose=False):
        """Create a Semantic Classifier for ECG classification.

        Args:
            config: an instance of Config
            restore: if `True`, restores a model.
            mode: str, either 'train' or 'inference'
            verbose: bool
        """

        self.config = config
        self.mode = mode
        self.model_path = self.config.model_dir
        self.verbose = verbose

        if self.verbose:
            print()
            print("*" * 80)
            print("\nCreating {}\n".format(self.config.identifier))
            print("Config:\n")
            pprint(c.parameters)

        self.graph = tf.Graph()
        with self.graph.as_default():

            self._create_placeholders()
            self._create_graph()    
            self._initialize_global_step()

            self.model_vars = misc.get_vars_by_scope("")
            self.all_vars = self.model_vars

            if mode is 'train':
                self._register_config(self.config)
                self.cost = self._create_cost(self.logits_answer, self.answer)
                self._create_metrics()
                self.merged = self._create_summaries()
                self.train_op = self._create_optimizer(self.cost)
                self.optimizer_vars = misc.get_vars_by_scope("optimizer")
                self.all_vars = self.model_vars + self.optimizer_vars
                self._create_summary_writers(self.config.summary_dir)

            self.saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP,
                                        var_list=self.all_vars)
            if restore:
                try:
                    self.load_model(verbose=False)
                except FileNotFoundError:
                    print("Model has not been restored. "
                          "Variables will be initialized.")
                    misc.initialize_vars(self.all_vars, sess=self.sess)
            else:
                misc.initialize_vars(self.all_vars, sess=self.sess)

        if self.verbose:
            print("\nNumber of parameters: {:.1f}M".format(
                    misc.number_of_parameters(self.model_vars) / 1e6))

        if mode is 'train':
            if self.verbose:
                print("Iteration step: {step}".format(step=self.get_step()))
        
    def _create_placeholders(self):
        with tf.name_scope("placeholders"):
            self.questions = tf.placeholder(tf.int32,
                                            [None, self.config.question_size],
                                            'questions')
            
            self.context = tf.placeholder(tf.int32,
                                          [None, self.config.context_size],
                                          'context')

            self.pos = tf.placeholder(tf.int32,
                                          [None, self.config.context_size],
                                          'part_of_speech')

            self.ner = tf.placeholder(tf.int32,
                                          [None, self.config.context_size],
                                          'named_entity_recognition')

            self.context_features = tf.placeholder(tf.float32,
                                                   [None, self.config.context_size, 4],
                                                   'context_features')
            
            self.answer = tf.placeholder(tf.float32,
                                         [None, self.config.context_size],
                                         'answer')
                        
            self.learn_rate = tf.placeholder(tf.float32, name="learn_rate")

            self.weight_decay = tf.placeholder(tf.float32, name='weight_decay')

            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    def _create_graph(self):
        print('Creat graph')

        self.w_embs = self.load_word_embeddings()
        self.pos_embs = self.load_pos_embeddings()
        self.ner_embs = self.load_ner_embeddings()

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)

        p = tf.nn.embedding_lookup(self.w_embs, self.context)
        p = tf.nn.dropout(p, keep_prob=self.keep_prob)
        q = tf.nn.embedding_lookup(self.w_embs, self.questions)
        q = tf.nn.dropout(q, keep_prob=self.keep_prob)
        
        aligned_emb = self.align_question_embedding(p, q)

        pos = tf.nn.embedding_lookup(self.pos_embs, self.pos)
        pos = tf.nn.dropout(pos, keep_prob=self.keep_prob)
        ner = tf.nn.embedding_lookup(self.ner_embs, self.ner)
        ner = tf.nn.dropout(ner, keep_prob=self.keep_prob)

        all_context_features = tf.concat([p, aligned_emb, pos, ner], -1)

        processed_featurs = self.create_rnn_graph(all_context_features)

        encoded_question = self.encode_question(q)
        
        self.logits_answer = bilinear_sequnce_attention(processed_featurs,
                                                        encoded_question)
        
        self.answer_probability = tf.sigmoid(self.logits_answer)

        print('Done!')

    def load_word_embeddings(self):
        print('\tLoading word embeddings...')
        meta = load_pickled_object(self.config.path_to_meta)
        embs = meta['emb']
        W = tf.get_variable(name="w_embs", shape=embs.shape,
            initializer=tf.constant_initializer(embs), trainable=False)
        return W

    def load_pos_embeddings(self):
        print('\tLoading pos embeddings...')
        embs = np.eye(len(self.config.vocab_tag))
        W = tf.get_variable(name="pos_embs", shape=embs.shape,
            initializer=tf.constant_initializer(embs), trainable=False)
        return W

    def load_ner_embeddings(self):
        print('\tLoading ner embeddings...')
        embs = np.eye(len(self.config.vocab_ent))
        W = tf.get_variable(name="ner_embs", shape=embs.shape,
            initializer=tf.constant_initializer(embs), trainable=False)
        return W

    def align_question_embedding(self, p, q):
        """ Compute Aligned question embedding.
        
        Args:
            p: context tensor, shape batch_size x context_size x emb_size
            q: question tensor, shape batch_size x question_size x emb_size
        
        Return:
            tensor of shape batch_size x context_size x hidden_size
        """
        return SeqAttnMatch(p, q)

    def create_rnn_graph(self, x):
        print('\tcreate_rnn_graph')
        outs = []
        for i in range(self.config.n_rnn_layers):
            f_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
            b_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
            res = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, x,
                dtype=tf.float32, scope='context_rnn{}'.format(i))
            x = tf.concat(res[0], axis=-1)
            outs.append(x)

        res = tf.concat(outs, axis=-1)
        return res

    def encode_question(self, q):
        print('\tencode_question')
        outs = []
        for i in range(self.config.n_rnn_layers):
            f_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
            b_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
            res = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, q,
                dtype=tf.float32, scope='question_rnn{}'.format(i))
            q = tf.concat(res[0], axis=-1)
            outs.append(q)

        res = tf.concat(outs, axis=-1)

        weights = LinearSeqAttn(res)
        weights = tf.reshape(weights, [-1, 1, self.config.question_size])
        encoded_question =  tf.einsum('ijk,ikq->ijq', weights, res) # b x 1 x 6*hidden_size
        encoded_question = tf.reshape(encoded_question,
            [-1, 2*self.config.n_rnn_layers*self.config.hidden_size])
        return encoded_question
    
    def _create_cost(self, logits_answer, answer):
        print('_create_cost')
        self.cost = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets=answer,
                                                    logits=logits_answer,
                                                    pos_weight=self.config.pos_weight))
        return self.cost

    def _create_metrics(self):
        print('_create_metrics')
        self.acc = metrics.accuracy_multilabel(labels=self.answer,
                                               logits=self.logits_answer,
                                               threshold=self.config.threshold)
        precision, recall, f1 = metrics.pr_re_f1_multilabel(
            labels=self.answer, logits=self.logits_answer,
            threshold=self.config.threshold)
        self.precision = tf.reduce_mean(precision)
        self.recall = tf.reduce_mean(recall)
        self.f1 = tf.reduce_mean(f1)



    def _create_summaries(self):
        print('_create_summaries')
        tf.summary.scalar('accuracy', self.acc)
        tf.summary.scalar('loss', self.cost)
        tf.summary.scalar('precission', self.precision)
        tf.summary.scalar('recall', self.recall)
        tf.summary.scalar('f1', self.f1)
        return tf.summary.merge_all()

    def _create_optimizer(self, cost):
        print('_create_optimizer')
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learn_rate)
            grad_var = optimizer.compute_gradients(cost)
            grad_var = [(tf.clip_by_value(g, -1, 1), v) for g,v in grad_var]
            train = optimizer.apply_gradients(grad_var)
        return train
    
    def _train_step(self, loader):
        data = loader.new_batch()

        feedDict = {
            self.questions : data[0],
            self.context : data[1],
            self.pos : data[2],
            self.ner : data[3],
            self.context_features : data[4],
            self.answer : data[5],
            self.weight_decay : self.config.weight_decay,
            self.learn_rate : self.config.learn_rate,
            self.keep_prob : self.config.keep_prob}
        self.sess.run(self.train_op, feed_dict=feedDict)

    def _save_summaries(self, loader, writer, iteration):

        data = loader.new_batch()
        feedDict = {
            self.questions : data[0],
            self.context : data[1],
            self.pos : data[2],
            self.ner : data[3],
            self.context_features : data[4],
            self.answer : data[5],
            self.keep_prob : self.config.keep_prob}

        summary = self.sess.run(self.merged, feed_dict=feedDict)
        writer.add_summary(summary, iteration)

    def fit(self,
            train_loader,
            log_interval: int,
            save_interval: int,
            validation_loader=None):
        """Fits model to data provided by `train_loader`

        Args:
            train_loader: train data provider. Should have method `new_batch()`
            log_interval: int, how frequenlty to save summary in terms
                of number of iterations
            save_interval: int, how frequently to save model in terms
                of number of iterations
            validation_loader: loader for validation data
        """
    
        for iteration in tqdm.tqdm(range(self.get_step(),
                                   self.config.iterations), ncols=75):
            self._train_step(train_loader)
            if (iteration + 1) % log_interval == 0:
                self._save_summaries(train_loader, self.train_writer, iteration)
                self._save_summaries(validation_loader, self.validation_writer,
                    iteration)
            if (iteration + 1) % save_interval == 0:
                self.save_model()
    
    def predict(self, question, context, pos, ner, context_features):
        feedDict = {
            self.questions : question,
            self.context : context,
            self.pos : pos,
            self.ner : ner,
            self.context_features : context_features,
            self.keep_prob : 1}
        return self.sess.run(self.answer_probability, feed_dict=feedDict)

def bilinear_sequnce_attention(seq, context):
    """ A bilinear attention layer over a sequence seq w.r.t context

    Args:
        seq: 3D tensor of shape b x l x h1
        context: 2D tensor of shape b x h2
    
    Return:
        tensor of shape b x l with weight coefficients
    """

    l, h1 = seq.get_shape().as_list()[1:3]
    context = tf.layers.dense(context, h1,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
    context = tf.reshape(context, [-1, h1, 1]) # b x h1 x 1
    z = tf.einsum('ijk,ikq->ijq', seq, context)
    z = tf.reshape(z, [-1, l]) # b x l
    return z

if __name__ == '__main__':
    from config import config as c
    AnswerFinder(c, 'train')


def SeqAttnMatch(x, y):
    """Given sequences x and y, match sequence y to each element in x.

    Args:
        x: tensor of shape batch x len1 x h
        y: tensor of shape batch x len2 x h
    Return:
        matched_seq = batch * len1 * h
    """
    len1, h = x.get_shape().as_list()[1:]
    len2 = y.get_shape().as_list()[1]

    x_proj = tf.layers.dense(tf.reshape(x, [-1, h]), h, activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name='proj_dense', reuse=False)
    y_proj = tf.layers.dense(tf.reshape(y, [-1, h]), h, activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name='proj_dense', reuse=True)
        
    x_proj = tf.reshape(x_proj, [-1, len1, h])
    y_proj = tf.reshape(y_proj, [-1, len2, h])
    scores = tf.einsum('ijk,ikq->ijq', x_proj, tf.transpose(y_proj, [0, 2, 1])) # b x len1 x len2
    alpha_flat = tf.nn.softmax(tf.reshape(scores, [-1, len2]))
    alpha = tf.reshape(alpha_flat, [-1, len1, len2])
    matched_seq = tf.einsum('ijk,ikq->ijq', alpha, y)
    return matched_seq

def LinearSeqAttn(x):
    """Self attention over a sequence.

    Args:
        x: tensor of shape batch x len x hdim

    Return:
        tensor of shape batch x len
    """
    len_, hdim = x.get_shape().as_list()[1:]

    x_flat = tf.reshape(x, [-1, hdim])
    scores = tf.layers.dense(x, 1,
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    scores = tf.reshape(scores, [-1, len_])
    return tf.nn.softmax(scores)