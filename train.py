import random

import msgpack
from dslib.generic.data_providers import ParallelDataProvider

import tools
from answer_finder import AnswerFinder
from config import config as c

reader = tools.reader
with open(c.path_to_data, 'rb') as f:
    data = msgpack.load(f, encoding='utf8')

train_set, val_set = data['train'], data['dev']

with ParallelDataProvider(reader,
                          n_processes=8,
                          files=train_set,
                          capacity=4096,
                          batch_size=c.batch_size,
                          shuffle=False,
                          name="TrainLoader") as train_loader:

    with ParallelDataProvider(reader,
                              n_processes=4,
                              files=val_set,
                              capacity=512,
                              batch_size=c.batch_size,
                              shuffle=False,
                              name="ValidationLoader") as validation_loader:
        # create model
        model = AnswerFinder(config=c,
                            restore=True,
                            mode="train")
        # run model
        model.fit(train_loader=train_loader,
                  validation_loader=validation_loader,
                  log_interval=c.log_interval,
                  save_interval=c.save_interval)
