import nltk
import unidecode
import string

class Normalizaton:
    def __init__(self):
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        self.stemmer = nltk.stem.RSLPStemmer()
    
    def remove_accents(self, text):
        return unidecode.unidecode(text)
    
    def remove_ponctuations(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def tokenize_sentences(self, text):
        return self.sent_tokenizer.tokenize(text)
    
    def tokenize_words(self, text):
        return nltk.tokenize.word_tokenize(text)
    
    def lemmatize(self, text):
        return text
    
    def stemmize(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]
    