from bs4 import BeautifulSoup

import matplotlib
matplotlib.use('Agg')
from wikiapi import WikiApi
wiki = WikiApi()

import tools
from answer_finder import AnswerFinder
from config import config as c

model = AnswerFinder(config=c,
                    restore=True,
                    mode="inference")
print(chr(27) + "[2J")
print('''Hello! This is Alpha version of program for reading wikipedia to answer the question.
Program was writing basing on paper https://arxiv.org/pdf/1704.00051.pdf
For more detail cutlass900@gmail.com\n''')

while True:
    while True:
        print('What or who do you want to ask about? Example: Barak Obama')
        thing = input()
        results = wiki.find(thing)
        if len(results) > 0:
            print('Ok. I found few wiki pages about {}.'.format(thing))
            break
        else:
            print('Can\'t find any wiki pages about {}. Try another one.'.format(thing))
    while True:
        print('What is your question?')
        question = input()
        print('Your question is {}'.format(question))
        article = wiki.get_article(results[0])
        context = article.content
        print('Search answers ...')
        answers, probs = tools.get_answer(context, question, model, c)
        print('Found {} answer(s):'.format(len(answers)))
        for i, (a, p) in enumerate(zip(answers, probs)):
            print(str(i+1), a, ', probability = {0:.2f}'.format(p))
        print()
