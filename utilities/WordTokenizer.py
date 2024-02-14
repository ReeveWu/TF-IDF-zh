from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import time

def get_tokenizer():
    return WS("./data"), POS("./data"), NER("./data")

class WordTokenizer:
    def __init__(self):
        self.ws, self.pos, self.ner = self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        try:
            return get_tokenizer()
        except FileNotFoundError:
            data_utils.download_data_gdown("./")
            time.sleep(1)
            return get_tokenizer()
    
    def __call__(self, word_sentence_list, return_ws=True, return_pos=True, return_ner=True, return_ws_only=False, ignore_punctuation_marks=False):
        word_sentence_list = word_sentence_list if isinstance(word_sentence_list, list) else [word_sentence_list]
        if return_ws_only:
            return self.ws(word_sentence_list)
        output = {}
        if return_ws:
            output['ws'] = self.ws(word_sentence_list)
        if return_pos:
            output['pos'] = self.pos(word_sentence_list)
            if return_ner:
                output['ner'] = self.ner(word_sentence_list, output['pos'])
        return output