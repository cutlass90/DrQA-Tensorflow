import os

from dslib.tf_utils.config import (Config, IntegerParameter, RealParameter,
                                   StringParameter, CallableParameter, 
                                   Path, Scope, BooleanParameter,
                                   DictParameter, SequenceParameter)
config = Config(
    scope=Scope('answer_finder_my_prepro'),
    experiments_dir=Path('./experiments'),
    path = Path(os.path.abspath(__file__)),

    path_to_vector_embeddings = Path('../datasets/glove/glove.840B.300d.txt'),
    path_to_data = Path('./SQuAD2/data.msgpack'),
    path_to_meta = Path('./SQuAD2/meta.msgpack'),
    path_to_train_json = Path('../datasets/SQuAD/train-v1.1.json'),
    path_to_val_json = Path('../datasets/SQuAD/dev-v1.1.json'),
    
    question_size = IntegerParameter(40),
    context_size = IntegerParameter(400),
    batch_size = IntegerParameter(64),

    hidden_size = IntegerParameter(256),
    n_rnn_layers = IntegerParameter(3),
    emb_size = IntegerParameter(300),
    pos_size = IntegerParameter(50), #part-of-speech (POS)
    ner_size = IntegerParameter(19), #named entity recognition (NER)
    dict_size = IntegerParameter(91187),
    activation=StringParameter('tanh'),

    learn_rate = RealParameter(0.001),
    iterations = IntegerParameter(20000),
    weight_decay = RealParameter(0.001),
    save_interval = IntegerParameter(1000),
    log_interval = IntegerParameter(1),

    path_to_context = Path('./context.txt')

)