from typing import Any
from pprint import pprint
from collections import Counter
from utilities.WordTokenizer import WordTokenizer

class CountVectorizer:
    def __init__(self):
        self.word_tokenizer = WordTokenizer()
        self.vocabulary_ = []

    def set_vocabulary(self, vocabulary):
        self.vocabulary_ = vocabulary

    def _count_vectors(self, word_sentence_list):
        word_set = set()
        results = []
        for i in range(len(word_sentence_list)):
            element_counts = dict(Counter(word_sentence_list[i]))
            word_set.update(element_counts.keys())

        self.set_vocabulary(list(word_set))

        for i in range(len(word_sentence_list)):
            element_counts = dict(Counter(word_sentence_list[i]))
            tmp = []
            for key in self.vocabulary_:
                tmp.append(element_counts.get(key, 0))
            results.append(tmp)

        return results
    
    def fit_transform(self, raw_documents) -> Any:
        word_sentence_list = self.word_tokenizer(raw_documents, return_ws_only=True)
        return self._count_vectors(word_sentence_list)
