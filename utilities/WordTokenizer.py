from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

def get_tokenizer():
    return WS("./data"), POS("./data"), NER("./data")

class WordTokenizer:
    def __init__(self):
        try:
            self.ws, self.pos, self.ner = get_tokenizer()
        except:
            data_utils.download_data_gdown("./")
            self.ws, self.pos, self.ner = get_tokenizer()
    
    def __call__(self, *args):
        print(args[0])
        context = args[0]
        context = context if isinstance(context, list) else [context]
        return self.ws(context)