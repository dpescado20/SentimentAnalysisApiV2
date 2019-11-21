import operator
import pickle
import numpy as np
import pandas as pd
from text_preprocessor import SocialTextCleaner


class Classifier:
    def __init__(self):
        print('LOADING MODEL')
        self.model = pickle.load(open('model.pickle', 'rb'))
        print('LOADING VECTORIZER')
        self.vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

    def clean_text(self, text):
        stc = SocialTextCleaner()
        return stc.cleanText(text)

    def vectorize_text(self, text):
        return self.vectorizer.transform(text)

    def predict_text(self, text):
        predict_text = self.model.predict(text)
        unique, counts = np.unique(predict_text, return_counts=True)
        result = dict(zip(unique, counts))
        return max(result.items(), key=operator.itemgetter(1))[0]