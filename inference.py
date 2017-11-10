import tools
from answer_finder import AnswerFinder
from config import config as c

model = AnswerFinder(config=c,
                    restore=True,
                    mode="inference")
with open(c.path_to_context, 'r') as f:
    context = f.read()
# question = 'What was his mother\'s name?'
# question = 'Where did James born?'
question = 'What is his brother\'s name?'
answer = tools.get_answer(context, question, model)
print('Final answer:\n', answer)