from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import time
from typing import List, Tuple

def get_tokenizer() -> Tuple[WS, POS, NER]:
    return WS("./data"), POS("./data"), NER("./data")

class WordTokenizer:
    def __init__(self) -> None:
        self.ws, self.pos, self.ner = self._initialize_tokenizer()
    
    def _initialize_tokenizer(self) -> Tuple[WS, POS, NER]:
        try:
            return get_tokenizer()
        except FileNotFoundError:
            data_utils.download_data_gdown("./")
            time.sleep(1)
            return get_tokenizer()
        
    @staticmethod
    def english_word_processing(word_sentence: List[str], pos_sentence: List[str]) -> Tuple[List[str], List[str]]:
        assert len(word_sentence) == len(pos_sentence)
        tmp = list(zip(word_sentence, pos_sentence))
        remove_idx = []
        for i, (word, pos) in enumerate(tmp):
            if pos == 'FW':
                remove_idx.append(i)
                for x in word.split(' '):
                    if x != "":
                        word_sentence.append(x)
                        pos_sentence.append('FW')
        for index in sorted(remove_idx, reverse=True):
            del word_sentence[index]
            del pos_sentence[index]

        return word_sentence, pos_sentence
    
    def get_word_seg(self, word_sentence_list: List[List[str]]) -> dict:
        output = {}
        output['ws'] = self.ws(word_sentence_list)
        output['pos'] = self.pos(output['ws'])

        for i in range(len(word_sentence_list)):
            output['ws'][i], output['pos'][i] = self.english_word_processing(output['ws'][i], output['pos'][i])
        
        return output
        
    
    def __call__(self, word_sentence_list: List[List[str]], 
                 return_ws: bool = True, return_pos: bool = True, 
                 return_ws_only: bool = False, ignore_punctuation_marks: bool = False) -> dict:
        word_sentence_list = word_sentence_list if isinstance(word_sentence_list, list) else [word_sentence_list]
        
        output = self.get_word_seg(word_sentence_list)
        if return_ws_only:
            return output['ws']
        if not return_pos:
            del output['pos']
        if not return_ws:
            del output['ws']
        return output
