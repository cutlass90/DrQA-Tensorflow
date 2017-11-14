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
    
    def _create_graph(self):
        print('Creat graph')

        self.w_embs = self.load_word_embeddings()
        self.pos_embs = self.load_pos_embeddings()
        self.ner_embs = self.load_ner_embeddings()

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)

        p = tf.nn.embedding_lookup(self.w_embs, self.context)
        q = tf.nn.embedding_lookup(self.w_embs, self.questions)
        
        aligned_emb = self.align_question_embedding(p, q)

        pos = tf.nn.embedding_lookup(self.pos_embs, self.pos)
        ner = tf.nn.embedding_lookup(self.ner_embs, self.ner)

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
            q: question tensor, shape batch_size x question_size x emb_size
            p: context tensor, shape batch_size x context_size x emb_size
        
        Return:
            tensor of shape batch_size x context_size x hidden_size
        """
        def process_pi(x):
            """ Return pi tensor as weighted sum of question tensors. 
            
            Args:
                x: tensor of shape b x hidden_size

            Return:
                tensor of shape b x hidden_size
            """

            x = tf.reshape(x, (-1, 1, self.config.hidden_size))
            x = tf.reduce_sum(x*q, axis=2) # b x question_size
            x = tf.nn.softmax(x)
            x = tf.reduce_sum(tf.reshape(
                x, (-1, self.config.question_size, 1)) * q, axis=1)            
            return x
        print('\talign_question_embedding')
        p = tf.reshape(p, [-1, self.config.emb_size])
        p = tf.layers.dense(p, self.config.hidden_size, activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=self.regularizer, name='align_dense', reuse=False)
        p = tf.reshape(p, [-1, self.config.context_size, self.config.hidden_size])
        
        
        
        q = tf.reshape(q, [-1, self.config.emb_size])        
        q = tf.layers.dense(q, self.config.hidden_size, activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=self.regularizer, name='align_dense', reuse=True)
        q = tf.reshape(q, [-1, self.config.question_size, self.config.hidden_size])
        
        return time_distributed(process_pi, p)

    def create_rnn_graph(self, x):
        print('\tcreate_rnn_graph')

        f_cell = [tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
            for _ in range(self.config.n_rnn_layers)]
        multi_f_cell = tf.nn.rnn_cell.MultiRNNCell(f_cell)
        b_cell = [tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
            for _ in range(self.config.n_rnn_layers)]
        multi_b_cell = tf.nn.rnn_cell.MultiRNNCell(b_cell)

        res = tf.nn.bidirectional_dynamic_rnn(multi_f_cell, multi_b_cell, x,
            dtype=tf.float32)
        res = tf.concat(res[0], axis=-1)
        return res

    def encode_question(self, q):
        print('\tencode_question')
        f_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
        b_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
        res = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, q, dtype=tf.float32)
        res = tf.concat(res[0], -1)

        x = tf.reshape(res, [-1, 2*self.config.hidden_size])        
        x = tf.layers.dense(x, 1, activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=self.regularizer, name='encode_q_dense')
        x = tf.reshape(x, [-1, self.config.question_size])
        x = tf.nn.softmax(x)
        x = tf.reshape(x, [-1, self.config.question_size, 1])
        encoded_question = tf.reduce_sum(res*x, 1)
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
        self.acc = metrics.accuracy_multiclass(labels=self.answer,
                                               logits=self.logits_answer)
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
            # grad_var = [(tf.clip_by_value(g, -1, 1), v) for g,v in grad_var]
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
            self.learn_rate : self.config.learn_rate}
        self.sess.run(self.train_op, feed_dict=feedDict)

    def _save_summaries(self, loader, writer, iteration):

        data = loader.new_batch()
        feedDict = {
            self.questions : data[0],
            self.context : data[1],
            self.pos : data[2],
            self.ner : data[3],
            self.context_features : data[4],
            self.answer : data[5]}

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
            self.context_features : context_features}
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