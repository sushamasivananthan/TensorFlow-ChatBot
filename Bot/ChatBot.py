import nltk
from nltk.stem.lancaster import LancasterStemmer

import numpy as np
import tflearn
import random
import pickle
import json
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from Bot.train import getPath, getJsonPath
from keras.models import load_model
nltk.download('punkt')
import inspect, os

context = {}
def clean_up_sentence(inputString1):
    print('in cleanup .', inputString1)
    sentence_words = nltk.word_tokenize(inputString1)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words



def bow(inputString, words, show_details=False):
    inputString1=inputString
    print('b4 clean up sentense',inputString1)
    sentence_words = clean_up_sentence(inputString1)
    print(('after clean up sentense', sentence_words))
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def classify(sentence):
    inputString = sentence
    classes = pickle.load(open('classes.pkl', 'rb'))
    words = pickle.load(open('words.pkl', 'rb'))
    model = load_model('chatbot_model.h5')
    print(('b4 model', inputString))
    p = bow(inputString, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    '''results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]'''
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


class ChatBot(object):

    instance = None

    @classmethod
    def getBot(cls):
        if cls.instance is None:
            cls.instance = ChatBot()
        return cls.instance

    def getJsonPath(self):
        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        path = os.path.join(path, 'content.json').replace("\\", "/")
        return path

    def getPath(file):
        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        path = os.path.join(path, file).replace("\\", "/")
        return path

    def __init__(self):
        print("Init")
        if self.instance is not None:
            raise ValueError("Did you forgot to call getBot function ? ")

        self.stemmer = LancasterStemmer()
        data = pickle.load(open(getPath('trained_data'), "rb"))

        self.words = data['words']
        self.classes = data['classes']
        train_x = data['train_x']
        train_y = data['train_y']
        with open(getJsonPath()) as json_data:
            self.intents = json.load(json_data)
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)
        self.model = tflearn.DNN(net, tensorboard_dir=getPath('train_logs'))
        self.model.load(getPath('model.tflearn'))






    def response(self, sentence, userID='111', show_details=False):
        print('§§§§§§§§§§ inside code response:',sentence)
        results = classify(sentence)
        print('after results 77777:',results)

        if results:
            while results:
                for i in self.intents['intents']:
                    #print("intent",i)
                    if i['tag'] == results[0][0]:
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            if show_details: print('context:', i['context_set'])
                            context[userID] = i['context_set']

                        # check if this intent is contextual and applies to this user's conversation
                        if not 'context_filter' in i or \
                                (userID in context and 'context_filter' in i and i['context_filter'] == context[
                                    userID]):
                            if show_details: print('tag:', i['tag'])
                            # a random response from the intent
                            # return print(random.choice(i['responses']))
                            return random.choice(i['responses'])
                #print('resl:',results)
                results
