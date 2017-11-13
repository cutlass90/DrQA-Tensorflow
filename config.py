import os

from dslib.tf_utils.config import (Config, IntegerParameter, RealParameter,
                                   StringParameter, CallableParameter, 
                                   Path, Scope, BooleanParameter,
                                   DictParameter, SequenceParameter)
# in in word vectors PAD id is 0, and vector is np.zeros
# in in word vectors UNK id is 1, and vector is np.ones

config = Config(
    scope=Scope('answer_finder_my_prepro'),
    experiments_dir=Path('./experiments'),
    path = Path(os.path.abspath(__file__)),

    path_to_embeddings = Path('../datasets/glove/glove.840B.300d.txt'),
    path_to_train_json = Path('../datasets/SQuAD/train-v1.1.json'),
    path_to_val_json = Path('../datasets/SQuAD/dev-v1.1.json'),
    path_to_train_data = Path('./SQuAD/train.pkl'),
    path_to_val_data = Path('./SQuAD/val.pkl'),
    path_to_meta = Path('./SQuAD/meta.pkl'),
    
    question_size = IntegerParameter(30),
    context_size = IntegerParameter(200),
    batch_size = IntegerParameter(64),

    hidden_size = IntegerParameter(256),
    n_rnn_layers = IntegerParameter(3),
    emb_size = IntegerParameter(300),
    dict_size = IntegerParameter(1000000),
    activation=StringParameter('tanh'),
    threshold=RealParameter(0.5),
    pos_weight=RealParameter(2.0),

    learn_rate = RealParameter(0.001),
    iterations = IntegerParameter(20000),
    weight_decay = RealParameter(0.001),
    save_interval = IntegerParameter(300),
    log_interval = IntegerParameter(1),

    path_to_context = Path('./context.txt'),

    vocab_tag = SequenceParameter(
        ['<PAD>', '""', '#', '$', ',', '-LRB-', '-PRB-', '-RRB-', '.', ':', 'ADD',
    'AFX', 'BES', 'CC', 'CD', 'DT', 'EX', 'FW', 'GW', 'HVS', 'HYPH', 'IN', 'JJ',
    'JJR', 'JJS', 'LS', 'MD', 'NFP', 'NIL', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',
    'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SP', 'SYM', 'TO', 'UH',
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', '``'],
    descriptive=False),

    vocab_ent = SequenceParameter(
        ['<PAD>',  '', 'ORG', 'DATE', 'PERSON', 'GPE', 'CARDINAL', 'NORP', 'LOC',
        'WORK_OF_ART', 'PERCENT', 'EVENT', 'ORDINAL', 'MONEY', 'FAC', 'QUANTITY',
        'LAW', 'TIME', 'LANGUAGE', 'PRODUCT'],
        descriptive=False)

)