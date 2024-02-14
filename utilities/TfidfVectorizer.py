from typing import Any
import numpy as np
from utilities.CountVectorizer import CountVectorizer
import math

class TfidfVectorizer:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.vocabulary_ = []
        self.idf_ = []

    def set_vocabulary(self, vocabulary):
        self.vocabulary_ = vocabulary
    
    @staticmethod
    def _smooth_algorithm(n_length, df_k):
        return math.log(n_length+1/df_k+1)
    
    def _smooth_idf(self, X):
        n_length = len(X)
        for x in list(map(list, zip(*X))):
            df_k = sum(1 for num in x if num > 0)
            self.idf_.append(self._smooth_algorithm(n_length=n_length, df_k=df_k))
    
    def _tfidf(self, X):
        X = np.array(X)
        return X * np.array(self.idf_)
    
    @staticmethod
    def _l2_normalization(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / norms
    
    def fit_transform(self, raw_documents) -> Any:
        X = self.vectorizer.fit_transform(raw_documents)
        self.set_vocabulary(self.vectorizer.vocabulary_)
        self._smooth_idf(X)
        X = self._tfidf(X)
        X = self._l2_normalization(X)
        return X

