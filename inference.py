import matplotlib
matplotlib.use('Agg')

import tools
from answer_finder import AnswerFinder
from config import config as c

model = AnswerFinder(config=c,
                    restore=True,
                    mode="inference")
with open(c.path_to_context, 'r') as f:
    context = f.read()
question = 'Where did James Kirk born?'
question = 'What is James Kirk brother\'s name?'
print('Search answers ...')
answers, probs = tools.get_answer(context, question, model, c)
print('Found {} answer(s):'.format(len(answers)))
for a, p in zip(answers, probs):
    print(a, ', probability = {0:.2f}'.format(p))
