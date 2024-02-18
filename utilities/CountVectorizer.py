from typing import List, Union, Any
from pprint import pprint
import numpy as np
from collections import Counter

from utilities.WordTokenizer import WordTokenizer

class CountVectorizer:
    def __init__(self):
        self.word_tokenizer = WordTokenizer()
        self.vocabulary_: List[str] = []

    def set_vocabulary(self, vocabulary: List[str]) -> None:
        self.vocabulary_ = vocabulary
    
    def fit(self, raw_documents: List[str]) -> List[List[str]]:
        word_sentence_list = self.word_tokenizer(raw_documents, return_ws_only=True)
        word_set = set()
        for i in range(len(word_sentence_list)):
            element_counts = dict(Counter(word_sentence_list[i]))
            word_set.update(element_counts.keys())

        self.set_vocabulary(list(word_set))
        return word_sentence_list
    
    def transform(self, word_sentence_list: List[List[str]]) -> List[List[int]]:
        results = []
        for i in range(len(word_sentence_list)):
            element_counts = dict(Counter(word_sentence_list[i]))
            tmp = []
            for key in self.vocabulary_:
                tmp.append(element_counts.get(key, 0))
            results.append(tmp)

        return results
    
    def fit_transform(self, raw_documents: List[str]) -> List[List[int]]:
        word_sentence_list = self.fit(raw_documents)
        return self.transform(word_sentence_list)
