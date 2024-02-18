from typing import Tuple, List, Optional
import numpy as np
from utilities.CountVectorizer import CountVectorizer
import math

class TfidfVectorizer:
    def __init__(self) -> None:
        self.vectorizer = CountVectorizer()
        self.vocabulary_: List[str] = []
        self.idf_: List[float] = []

    def reset(self) -> None:
        self.vectorizer.vocabulary_ = []
        self.vocabulary_ = []
        self.idf_ = []

    def set_vocabulary(self, vocabulary: List[str]) -> None:
        self.vocabulary_ = vocabulary

    @staticmethod
    def _smooth_algorithm(n_length: int, df_k: int) -> float:
        return math.log((n_length + 1) / (df_k + 1))

    def _smooth_idf(self, X: List[List[int]]) -> None:
        self.idf_ = []
        n_length = len(X)
        for x in map(list, zip(*X)):
            df_k = sum(1 for num in x if num > 0)
            self.idf_.append(self._smooth_algorithm(n_length=n_length, df_k=df_k))

    def _tfidf(self, X: np.ndarray) -> np.ndarray:
        return X * np.array(self.idf_)

    @staticmethod
    def _l2_normalization(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / norms

    def fit(self, raw_documents: List[str]) -> Tuple[np.ndarray, List[List[str]]]:
        word_sentence_list = self.vectorizer.fit(raw_documents)
        self.set_vocabulary(self.vectorizer.vocabulary_)
        X = self.vectorizer.transform(word_sentence_list)
        self._smooth_idf(X)
        return X, word_sentence_list

    def transform(self, word_sentence_list: List[List[str]], X: Optional[np.ndarray] = None) -> np.ndarray:
        if X is None:
            X = self.vectorizer.transform(word_sentence_list)
        if X is not None:
            X = self._tfidf(X)
            X = self._l2_normalization(X)
            return X

    def fit_transform(self, raw_documents: List[str]) -> Tuple[np.ndarray, List[List[str]]]:
        X, word_sentence_list = self.fit(raw_documents)
        return self.transform(word_sentence_list, X)
    