import nltk
import unidecode
import string


class Normalizer:
    def __init__(self):
        self.sent_tokenizer = nltk.data.load(
            'tokenizers/punkt/portuguese.pickle')
        self.stemmer = nltk.stem.RSLPStemmer()

    def remove_ponctuations(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_accents(self, text):
        return unidecode.unidecode(text)

    def remove_stopwords(self, tokens):
        stopwords = nltk.corpus.stopwords.words('portuguese')
        return [word for word in tokens if word not in stopwords]

    def to_lowercase(self, text):
        return text.lower()

    def stemmize(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

    def tokenize_sentences(self, text):
        return self.sent_tokenizer.tokenize(text)

    def tokenize_words(self, text):
        return nltk.tokenize.word_tokenize(text)
