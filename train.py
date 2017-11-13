import random

import msgpack
from dslib.generic.data_providers import ParallelDataProvider

from tools import Reader, load_pickled_object, Paragraph, QA, Answer
from answer_finder import AnswerFinder
from config import config as c

reader = Reader(c)
train_set = load_pickled_object(c.path_to_train_data)
val_set = load_pickled_object(c.path_to_val_data)

with ParallelDataProvider(reader.read,
                          n_processes=2,
                          files=train_set,
                          capacity=512,
                          batch_size=c.batch_size,
                          shuffle=False,
                          name="TrainLoader") as train_loader:

    with ParallelDataProvider(reader.read,
                              n_processes=2,
                              files=val_set,
                              capacity=256,
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
