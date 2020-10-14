from flask import Flask, render_template, request, url_for
from flask_bootstrap import Bootstrap
from flaskext.markdown import Markdown



#NLP
from textblob import Word
import spacy
from spacy import displacy
from spacymoji import Emoji
import pandas as pd
import numpy as np
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
import time


# loading in the models to predict on the data
pickle_in_1 = open('../model_pickles/Best_Model_P1.sav', 'rb')
classifier_1 = pickle.load(pickle_in_1)

pickle_in_2 = open('../model_pickles/Best_Model_BadReview_P2.sav', 'rb')
classifier_2 = pickle.load(pickle_in_2)

pickle_in_3 = open('../model_pickles/Best_Model_GoodReview_P3.sav', 'rb')
classifier_3 = pickle.load(pickle_in_3)


# defining the function which will make the prediction using
# the data which the user inputs
def prediction(reviews):

    prediction = classifier_1.predict(
        [reviews])
    print(prediction)
    return prediction

def prediction_bad(reviews):

    prediction = classifier_2.predict(
        [reviews])
    print(prediction)
    return prediction

def prediction_good(reviews):

    prediction = classifier_3.predict(
        [reviews])
    print(prediction)
    return prediction


sp = spacy.load('en_core_web_sm')
emoji = Emoji(sp, merge_spans = False)
sp.add_pipe(emoji, first = True)

app = Flask(__name__)
Bootstrap(app)
Markdown(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse', methods = ['POST'])
def analyse():
    start = time.time()
    analyser = SentimentIntensityAnalyzer()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        #NLP stuff
        sen = sp(rawtext)

        colors = {"ORG": "linear-gradient(#00FFF8, #95D0E6, #009CD7)"}
        options = {"ents": ["ORG"], "colors": colors}
        displaysen = displacy.render(sen, style = 'ent', options=options)

        received_text = sen
        number_of_tokens = len(list(sen))

        #get score from raw text
        scores = analyser.polarity_scores(str(rawtext))
        received_text = rawtext
        neg = scores['neg']
        neu = scores['neu']
        pos = scores['pos']
        compound = scores['compound']


        # predicting review and category
        result = prediction(rawtext)
        if result == 1:
            result = 'Good Review'
            final = prediction_good(rawtext)[0]
        else:
            result = 'Bad Review'
            final = prediction_bad(rawtext)[0]


        nouns = list()
        summary = list()
        final_time = list()
        for word in sen:
            if word.tag_ =='NN':
                nouns.append(word.lemma_)
                len_of_words = len(nouns)
                ran_words = random.sample(nouns,len(nouns))
                final_word = list()
                for item in ran_words:
                    word = Word(item).pluralize()
                    final_word.append(word)
        summary = final_word
        end = time.time()
        final_time.append(round(end-start, 2))

    return render_template('index.html', received_text = received_text,
                                        number_of_tokens = number_of_tokens,
                                        displaysen = displaysen,
                                        neg = neg,
                                        pos = pos,
                                        neu = neu,
                                        compound = compound,
                                        result = result,
                                        final = final,
                                        summary = summary,
                                        final_time = final_time[0])


if __name__ == '__main__':
    app.run(debug = True)
